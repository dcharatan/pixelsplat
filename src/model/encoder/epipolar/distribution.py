from typing import Optional

import torch
from einops import einsum
from jaxtyping import Bool, Float
from torch import Tensor, nn


class Distribution(nn.Module):
    scale: float
    to_q: nn.Linear
    to_k: nn.Linear

    def __init__(
        self,
        dim_q: int,
        dim_k: int,
        dim_inner: int = 64,
    ):
        super().__init__()
        self.scale = dim_inner**-0.5
        self.to_q = nn.Linear(dim_q, dim_inner, bias=False)
        self.to_k = nn.Linear(dim_k, dim_inner, bias=False)

    def forward(
        self,
        queries: Float[Tensor, "batch token_query dim_query"],
        keys: Float[Tensor, "batch token_key dim_key"],
        force_last_token: Optional[Bool[Tensor, " batch"]] = None,
    ) -> Float[Tensor, "batch token_query token_key"]:
        # Compute softmax attention.
        q = self.to_q(queries)
        k = self.to_k(keys)
        weights = einsum(q, k, "b q d, b k d -> b q k").softmax(dim=-1)

        if force_last_token is None:
            return weights

        # Where applicable, force the last token to be selected.
        last_token = torch.zeros(
            keys.shape[1], device=queries.device, dtype=queries.dtype
        )
        last_token[-1] = 1
        mask = force_last_token[:, None, None]
        return last_token * mask + weights * ~mask
