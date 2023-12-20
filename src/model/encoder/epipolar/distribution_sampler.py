import torch
from jaxtyping import Float, Int64, Shaped
from torch import Tensor

from ....misc.discrete_probability_distribution import (
    gather_discrete_topk,
    sample_discrete_distribution,
)


class DistributionSampler:
    def sample(
        self,
        pdf: Float[Tensor, "*batch bucket"],
        deterministic: bool,
        num_samples: int,
    ) -> tuple[
        Int64[Tensor, "*batch sample"],  # index
        Float[Tensor, "*batch sample"],  # probability density
    ]:
        """Sample from the given probability distribution. Return sampled indices and
        their corresponding probability densities.
        """
        if deterministic:
            index, densities = gather_discrete_topk(pdf, num_samples)
        else:
            index, densities = sample_discrete_distribution(pdf, num_samples)
        return index, densities

    def gather(
        self,
        index: Int64[Tensor, "*batch sample"],
        target: Shaped[Tensor, "..."],  # *batch bucket *shape
    ) -> Shaped[Tensor, "..."]:  # *batch *shape
        """Gather from the target according to the specified index. Handle the
        broadcasting needed for the gather to work. See the comments for the actual
        expected input/output shapes since jaxtyping doesn't support multiple variadic
        lengths in annotations.
        """
        bucket_dim = index.ndim - 1
        while len(index.shape) < len(target.shape):
            index = index[..., None]
        broadcasted_index_shape = list(target.shape)
        broadcasted_index_shape[bucket_dim] = index.shape[bucket_dim]
        index = index.broadcast_to(broadcasted_index_shape)

        # Add the ability to broadcast.
        if target.shape[bucket_dim] == 1:
            index = torch.zeros_like(index)

        return target.gather(dim=bucket_dim, index=index)
