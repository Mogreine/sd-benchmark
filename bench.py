import argparse
import functools
import itertools
import time
import torch.nn.functional as F
import torch.utils.checkpoint
import torch
import torch.utils.checkpoint

from torch.utils.data import Dataset
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from tqdm import tqdm
from transformers import CLIPTextModel
from diffusers import AutoencoderKL
from loguru import logger
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer


HF_TOKEN = "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"


def collate_fn(examples, with_prior_preservation, tokenizer):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch


class BenchmarkDataset(Dataset):
    def __init__(self, tokenizer: CLIPTokenizer, length: int):
        super().__init__()
        self.class_sample = torch.randn(3, 256, 256)
        self.instance_sample = torch.randn(3, 256, 256)
        self.class_text = "a photo of a man"
        self.instance_text = "a photo of a sks man"
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.tokenizer = tokenizer
        self.length = length

    def _prepare_sample(self, image, prompt):
        prompt = self.tokenizer(
            prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return image, prompt

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        example = {}
        image, prompt = self._prepare_sample(self.instance_sample, self.instance_text)
        example["instance_images"] = image
        example["instance_prompt_ids"] = prompt

        image, prompt = self._prepare_sample(self.class_sample, self.class_text)
        example["class_images"] = image
        example["class_prompt_ids"] = prompt

        return example


class DreamBoothBenchmarkPipeline:
    def __init__(self, device: str = "cuda:0", use_fp32: bool = False):
        self.device = device
        self.precision = torch.float32 if use_fp32 else torch.float16
        self.vae_path = "stabilityai/sd-vae-ft-mse"
        self.model_path = "runwayml/stable-diffusion-v1-5"

        # Loading models
        self.load_weights("vae")
        self.load_weights("unet")
        self.load_weights("text_encoder")

        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_path,
            subfolder="tokenizer",
        )

        self.noise_scheduler = DDPMScheduler.from_config(self.model_path, subfolder="scheduler", use_auth_token=HF_TOKEN)

    def load_weights(self, model_name: str):
        if model_name == "vae":
            self.vae = AutoencoderKL.from_pretrained(self.vae_path, use_auth_token=HF_TOKEN)
            self._freeze_model(self.vae)
            self.vae.to(self.device)
        elif model_name == "unet":
            self.unet = UNet2DConditionModel.from_pretrained(self.model_path, subfolder="unet", use_auth_token=HF_TOKEN)
            self.unet.to(self.device)
        elif model_name == "text_encoder":
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.model_path, subfolder="text_encoder", use_auth_token=HF_TOKEN
            )
            self.text_encoder.to(self.device)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def _freeze_model(self, model):
        model.requires_grad_(False)
        model.eval()

    def _loss(self, noise, noise_pred, noise_prior=None, noise_prior_pred=None) -> torch.Tensor:
        # Compute instance loss
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean([1, 2, 3]).mean()

        # Compute prior loss
        if noise_prior_pred is not None and noise_prior is not None:
            prior_loss = F.mse_loss(noise_prior_pred.float(), noise_prior.float(), reduction="mean")
        else:
            prior_loss = 0

        # Add the prior loss to the instance loss.
        loss = loss + prior_loss

        return loss

    def _train_step(self, batch, is_vae_latents_precalculated: bool = False):
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Calculating vae latents
        if is_vae_latents_precalculated:
            latents = batch["pixel_values"] * 0.18215
        else:
            # Convert images to latent space
            latents = self.vae.encode(batch["pixel_values"]).latent_dist.sample()
            latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        return noise, noise_pred

    def bench(self, n_steps: int, train_text_encoder: bool = True, train_unet: bool = True) -> float:
        # Freezing models
        if not train_text_encoder:
            self.text_encoder.gradient_checkpointing_disable()
            self._freeze_model(self.text_encoder)
        if not train_unet:
            self.unet.gradient_checkpointing_disable()
            self._freeze_model(self.unet)

        params_to_optimize = (
            itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
            if train_text_encoder
            else self.unet.parameters()
        )
        optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            lr=2e-6,
            weight_decay=0.01,
        )

        collate_fn_ = functools.partial(collate_fn, with_prior_preservation=True, tokenizer=self.tokenizer)
        train_dataset = BenchmarkDataset(self.tokenizer, length=n_steps)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_, pin_memory=True
        )

        lr_scheduler = get_scheduler(
            "constant",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=n_steps,
        )

        # Only show the progress bar once on each machine.
        scaler = torch.cuda.amp.GradScaler()
        if train_unet:
            self.unet.train()
        if train_text_encoder:
            self.text_encoder.train()

        start_time = time.time()
        for step, batch in tqdm(enumerate(train_dataloader), desc="Training...", total=n_steps):
            with torch.amp.autocast(device_type="cuda", dtype=self.precision):
                noise, noise_pred = self._train_step(batch, False)

                # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                noise_pred, noise_prior_pred = torch.chunk(noise_pred, 2, dim=0)
                noise, noise_prior = torch.chunk(noise, 2, dim=0)

                loss = self._loss(noise, noise_pred, noise_prior, noise_prior_pred)

            # Backward pass with loss scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        it_sec = n_steps / (time.time() - start_time)

        return it_sec


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_fp32", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = DreamBoothBenchmarkPipeline(args.device, args.use_fp32)

    clip_plus_unet_it_sec = pipeline.bench(args.n_steps, train_text_encoder=False, train_unet=True)
    unet_only_it_sec = pipeline.bench(args.n_steps, train_text_encoder=False, train_unet=True)

    logger.info(f"CLIP+Unet: {clip_plus_unet_it_sec:.2f} it/s")
    logger.info(f"Unet only: {unet_only_it_sec:.2f} it/s")


if __name__ == "__main__":
    main()
