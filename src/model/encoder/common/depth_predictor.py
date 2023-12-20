import torch
from einops import rearrange
from jaxtyping import Float, Int64
from torch import Tensor, nn

from ..epipolar.conversions import relative_disparity_to_depth
from ..epipolar.distribution_sampler import DistributionSampler


class DepthPredictor(nn.Module):
    sampler: DistributionSampler

    def __init__(
        self,
        use_transmittance: bool,
    ) -> None:
        super().__init__()
        self.sampler = DistributionSampler()
        self.to_pdf = nn.Softmax(dim=-1)
        self.to_offset = nn.Sigmoid()
        self.use_transmittance = use_transmittance

    def forward(
        self,
        features: Float[Tensor, "batch view ray surface depth 2"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        deterministic: bool,
        gaussians_per_pixel: int,
    ) -> tuple[
        Float[Tensor, "batch view ray surface sample"],  # depth
        Float[Tensor, "batch view ray surface sample"],  # opacity
        Int64[Tensor, "batch view ray surface sample"],  # index
    ]:
        # Convert the features into a depth distribution plus intra-bucket offsets.
        pdf_raw, offset_raw = features.unbind(dim=-1)
        pdf = self.to_pdf(pdf_raw)
        offset = self.to_offset(offset_raw)

        # Sample from the depth distribution.
        index, pdf_i = self.sampler.sample(pdf, deterministic, gaussians_per_pixel)
        offset = self.sampler.gather(index, offset)

        # Convert the sampled bucket and offset to a depth.
        *_, num_depths, _ = features.shape
        relative_disparity = (index + offset) / num_depths
        depth = relative_disparity_to_depth(
            relative_disparity,
            rearrange(near, "b v -> b v () () ()"),
            rearrange(far, "b v -> b v () () ()"),
        )

        # Compute opacity from PDF.
        if self.use_transmittance:
            partial = pdf.cumsum(dim=-1)
            partial = torch.cat(
                (torch.zeros_like(partial[..., :1]), partial[..., :-1]), dim=-1
            )
            opacity = pdf / (1 - partial + 1e-10)
            opacity = self.sampler.gather(index, opacity)
        else:
            opacity = pdf_i

        return depth, opacity, index
