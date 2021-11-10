import torch
a = torch.tensor([1, 1], device='cpu')
b = torch.tensor([1, 1], device='cpu')
c = torch.add(a, b)
