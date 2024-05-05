import time
import torch
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"]=""

def to_us(diff: float) -> int:
    diff *= 1e6
    return int(diff)

def print_interval(start: float, end: float):
    diff = end - start
    diff *= 1e6
    print(f"took {int(diff)}us")

iterations: int = 100
resp = {"op": "mean", "iterations": iterations}
for withSleep in [True, False]:
    for size in [2048, 2048 * 10, 2048 * 100, 2048 * 1000]:
        a = torch.tensor([x for x in range(size)], dtype=torch.float32, device='cpu').cpu()
        average = 0.0
        print(f"============== size: {size}, withSleep: {withSleep} ==============")
        for i in range(iterations):
            # print(f'Iteration {i + 1}')
            start = time.perf_counter()
            c = a.mean()
            end = time.perf_counter()
            interval = to_us(end - start)
            # print(f"took {interval}us; sum: {c}")
            average += end - start
            if withSleep:
                time.sleep(0.1)
        print(f"average: {to_us(average / iterations)}us")
        resp[f"{size}_{withSleep}"] = {"averageTime": to_us(average / iterations), "size": size, "withSleep": withSleep}

print(json.dumps(resp, indent=2))
