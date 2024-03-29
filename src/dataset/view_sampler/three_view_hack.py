import torch
from jaxtyping import Int
from torch import Tensor


def add_third_context_index(
    indices: Int[Tensor, "*batch 2"]
) -> Int[Tensor, "*batch 3"]:
    left, right = indices.unbind(dim=-1)
    return torch.stack((left, (left + right) // 2, right), dim=-1)
