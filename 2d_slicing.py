import torch
import numpy as np

a = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])

a_sliced = a[1:3, 1:3]
print(a)
print(a_sliced)
