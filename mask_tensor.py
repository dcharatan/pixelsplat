import torch
from torch import Tensor


x = torch.tensor([0.6, 0.7, 0.3, 0.4, 0.5])

mask = x.ge(0.5)
select =  torch.masked_select(x, mask)
print(select.shape)