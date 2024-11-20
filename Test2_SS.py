import os
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import torch
import time





temp = torch.randint(0, 3, (5, 3)).to(torch.float32)

mapping = {0: (0, 0, 0), 1: (1, 1, 1), 2: (2, 2, 2)}

data = torch.nn.functional.softmax(temp, dim=-1)

print(temp)

print(data)


print(torch.argmax(data, dim=-1))
