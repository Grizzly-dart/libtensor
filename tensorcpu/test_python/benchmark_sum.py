import time
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"]=""

def to_us(diff: float) -> int:
    diff *= 1e6
    return int(diff)

def print_interval(start: float, end: float):
    diff = end - start
    diff *= 1e6
    print(f"took {int(diff)}us")

size: int = 2048 * 1000
a = torch.tensor([x for x in range(size)], dtype=torch.float32, device='cpu').cpu()

for i in range(3):
    c = a.sum()

average = 0.0
iterations: int = 100
for i in range(iterations):
    print(f'Iteration {i + 1}')
    start = time.perf_counter()
    c = a.sum(dtype=torch.float64)
    end = time.perf_counter()
    interval = to_us(end - start)
    print(f"took {interval}us; sum: {c}")
    average += end - start

print(f"average: {to_us(average / iterations)}us")

