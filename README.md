# sd-benchmark
## Usage
Run:
```
docker build -t sd-container .
docker run -it --gpus all --network=host --shm-size=10g --name sd-benchmark -v "$(pwd)":/app sd-container
python bench.py
```