import time
import torch


def printInterval(start: float, end: float):
    diff = end - start
    diff *= 1e6
    print(f"took {int(diff)}us")

size: int = 100

for i in range(10):
    start = time.perf_counter()
    a = torch.tensor([x for x in range(size)], dtype=torch.float).to('cpu')
    b = torch.tensor([x for x in range(size)], dtype=torch.float).to('cpu')
    c = a + b
    printInterval(start, time.perf_counter())
    print('-----------------')

