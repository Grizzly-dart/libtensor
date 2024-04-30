import time
import torch


def printInterval(start: float, end: float):
    diff = end - start
    diff *= 1e6
    print(f"took {int(diff)}us")

size: int = 2048 * 10000

for i in range(10):
    a = torch.tensor([x for x in range(size)], dtype=torch.float).to('cpu')
    start = time.perf_counter()
    c = a.sum()
    printInterval(start, time.perf_counter())
    print('-----------------')

