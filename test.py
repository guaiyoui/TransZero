import torch
from utils import f1_score_calculation
a = torch.Tensor([[0.1, 0, 0.2], [1, 1, 0]])

b = torch.Tensor([[1, 1, 0], [1, 0, 1]])

print(a, b)
print(torch.multiply(a, b))
pre = torch.sum(torch.multiply(a, b))/torch.sum(a)
rec = torch.sum(torch.multiply(a, b))/torch.sum(b)

print(pre, rec)
print(2 * pre * rec / (pre + rec))

# print(my_f1_score_calculation(a, b), f1_score_calculation(a, b))

# print(my_f1_score_calculation(a, b))

print(a.shape)
print(a[0].shape)

if len(a[0].shape) == 1:
    print("yes")

print(a[0])
print(a[0].reshape(1, -1))

print(torch.nn.functional.normalize(a, dim=1, p=1))