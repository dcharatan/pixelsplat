from dataclasses import dataclass
from functools import partial
from typing import Optional

from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ....geometry.epipolar_lines import get_depth
from ....global_cfg import get_cfg
from ...encodings.positional_encoding import PositionalEncoding
from ...transformer.transformer import Transformer
from .conversions import depth_to_relative_disparity
from .epipolar_sampler import EpipolarSampler, EpipolarSampling
from .image_self_attention import ImageSelfAttention, ImageSelfAttentionCfg


@dataclass
class EpipolarTransformerCfg:
    self_attention: ImageSelfAttentionCfg
    num_octaves: int
    num_layers: int
    num_heads: int
    num_samples: int
    d_dot: int
    d_mlp: int
    downscale: int


class EpipolarTransformer(nn.Module):
    cfg: EpipolarTransformerCfg
    epipolar_sampler: EpipolarSampler
    depth_encoding: nn.Sequential
    transformer: Transformer
    downscaler: Optional[nn.Conv2d]
    upscaler: Optional[nn.ConvTranspose2d]
    upscale_refinement: Optional[nn.Sequential]

    def __init__(
        self,
        cfg: EpipolarTransformerCfg,
        d_in: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.epipolar_sampler = EpipolarSampler(
            get_cfg().dataset.view_sampler.num_context_views,
            cfg.num_samples,
        )
        if self.cfg.num_octaves > 0:
            self.depth_encoding = nn.Sequential(
                (pe := PositionalEncoding(cfg.num_octaves)),
                nn.Linear(pe.d_out(1), d_in),
            )
        feed_forward_layer = partial(ConvFeedForward, cfg.self_attention)
        self.transformer = Transformer(
            d_in,
            cfg.num_layers,
            cfg.num_heads,
            cfg.d_dot,
            cfg.d_mlp,
            selfatt=False,
            kv_dim=d_in,
            feed_forward_layer=feed_forward_layer,
        )

        if cfg.downscale:
            self.downscaler = nn.Conv2d(d_in, d_in, cfg.downscale, cfg.downscale)
            self.upscaler = nn.ConvTranspose2d(d_in, d_in, cfg.downscale, cfg.downscale)
            self.upscale_refinement = nn.Sequential(
                nn.Conv2d(d_in, d_in * 2, 7, 1, 3),
                nn.GELU(),
                nn.Conv2d(d_in * 2, d_in, 7, 1, 3),
            )

    def forward(
        self,
        features: Float[Tensor, "batch view channel height width"],
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
    ) -> tuple[Float[Tensor, "batch view channel height width"], EpipolarSampling,]:
        b, v, _, h, w = features.shape

        # If needed, apply downscaling.
        if self.downscaler is not None:
            features = rearrange(features, "b v c h w -> (b v) c h w")
            features = self.downscaler(features)
            features = rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)

        # Get the samples used for epipolar attention.
        sampling = self.epipolar_sampler.forward(
            features, extrinsics, intrinsics, near, far
        )

        if self.cfg.num_octaves > 0:
            # Compute positionally encoded depths for the features.
            collect = self.epipolar_sampler.collect
            depths = get_depth(
                rearrange(sampling.origins, "b v r xyz -> b v () r () xyz"),
                rearrange(sampling.directions, "b v r xyz -> b v () r () xyz"),
                sampling.xy_sample,
                rearrange(collect(extrinsics), "b v ov i j -> b v ov () () i j"),
                rearrange(collect(intrinsics), "b v ov i j -> b v ov () () i j"),
            )

            # Clip the depths. This is necessary for edge cases where the context views
            # are extremely close together (or possibly oriented the same way).
            depths = depths.maximum(near[..., None, None, None])
            depths = depths.minimum(far[..., None, None, None])
            depths = depth_to_relative_disparity(
                depths,
                rearrange(near, "b v -> b v () () ()"),
                rearrange(far, "b v -> b v () () ()"),
            )
            depths = self.depth_encoding(depths[..., None])
            q = sampling.features + depths
        else:
            q = sampling.features

        # Run the transformer.
        kv = rearrange(features, "b v c h w -> (b v h w) () c")
        features = self.transformer.forward(
            kv,
            rearrange(q, "b v () r s c -> (b v r) s c"),
            b=b,
            v=v,
            h=h // self.cfg.downscale,
            w=w // self.cfg.downscale,
        )
        features = rearrange(
            features,
            "(b v h w) () c -> b v c h w",
            b=b,
            v=v,
            h=h // self.cfg.downscale,
            w=w // self.cfg.downscale,
        )

        # If needed, apply upscaling.
        if self.upscaler is not None:
            features = rearrange(features, "b v c h w -> (b v) c h w")
            features = self.upscaler(features)
            features = self.upscale_refinement(features) + features
            features = rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)

        return features, sampling


class ConvFeedForward(nn.Module):
    def __init__(
        self,
        self_attention_cfg: ImageSelfAttentionCfg,
        d_in: int,
        d_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(d_in, d_hidden, 7, 1, 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(d_hidden, d_in, 7, 1, 3),
            nn.Dropout(dropout),
        )
        self.self_attention = ImageSelfAttention(self_attention_cfg, d_in, d_in)

    def forward(
        self,
        x: Float[Tensor, "batch token dim"],
        b: int,
        v: int,
        h: int,
        w: int,
    ) -> Float[Tensor, "batch token dim"]:
        x = rearrange(x, "(b v h w) () c -> (b v) c h w", b=b, v=v, h=h, w=w)
        x = self.layers(self.self_attention(x) + x)
        return rearrange(x, "(b v) c h w -> (b v h w) () c", b=b, v=v, h=h, w=w)
