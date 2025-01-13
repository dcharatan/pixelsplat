from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor, nn, optim

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from ..model.types import Gaussians




@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int


@dataclass
class TestCfg:
    output_path: Path


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class Model_4d(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()

    
    def mini_batch(self, batch: BatchedExample, t: int ):
        """
            return a minibatch at time t 
            example = {
                "context": {
                    "extrinsics": ex,
                    "intrinsics": Ks,
                    "image": images,
                    "near": self.get_bound("near", 2),
                    "far": self.get_bound("far", 2),
                    "index": torch.tensor([0]),
                    "mask": masks,
                    "t": length,
                },
                "target": {
                    "extrinsics": ex,
                    "intrinsics": Ks,
                    "image": images,
                    "near": self.get_bound("near", 1),
                    "far":  self.get_bound("far", 1),
                    "index": torch.tensor([0]),
                },
                "scene": scene,
            }
        """
        ex = batch["context"]["extrinsics"][0][t].unsqueeze(0)
        Ks = batch["context"]["intrinsics"][0][t].unsqueeze(0)
        images = batch["context"]["image"][0][t].unsqueeze(0)
        masks = batch["context"]["mask"][0][t].unsqueeze(0)
        
        v, _, h, w = images.shape
        images = repeat(images, "v c h w ->(m v) c h w", m=2).unsqueeze(0)
        masks = repeat(masks, "v c h w ->(m v) c h w", m=2).unsqueeze(0)
        ex = repeat(ex, "v h w ->(m v) h w", m=2).unsqueeze(0)
        Ks = repeat(Ks, "v h w ->(m v) h w", m=2).unsqueeze(0)
        
        near_c = batch["context"]["near"][0][t].unsqueeze(0)
        far_c = batch["context"]["far"][0][t].unsqueeze(0)
        
        near_t = batch["target"]["near"][0][0].unsqueeze(0)
        far_t = batch["target"]["far"][0][0].unsqueeze(0)
        
        scene = batch["scene"]
        
        
        # so we creating a new class
        # therefore, we would not affect the original batch
        example = {
            "context": {
                "extrinsics": ex,
                "intrinsics": Ks,
                "image": images,
                "near": near_c, 
                "far": far_c,
                "index": torch.tensor([0,0]).unsqueeze(0),
                "mask": masks,
                "t": 2,
            },
            "target": {
                "extrinsics": ex,
                "intrinsics": Ks,
                "image": images,
                "near": near_t,
                "far":  far_t,
                "index": torch.tensor([0]).unsqueeze(0),
            },
            "scene": scene,
        }
        return example
        
        
    def get_mask_indices(self, mask: Tensor) -> list:
        # mask b v c h w 
        #      1 2 1 176 320
        # "b v r srf spp xyz -> b (v r srf spp) xyz",
        # print(f"----shape of mask {mask.shape}")
        mask = rearrange(mask, "b v c h w -> b v (h w) c")
        mask = repeat(mask,    "b v r c ->   b (v r p) c ", p=3)

        b, n, c  = mask.shape
        mask = mask.squeeze(0)
        mask = mask.squeeze(1)
        indices = torch.nonzero(mask)
        dynamic = indices.squeeze(1)
        # dynamic = indices[:,1]  # indices for those pixel in the mask
        
        scene = torch.ones(n)
        scene[dynamic] = False
        
        static = torch.nonzero(scene).squeeze(1)
        # print(f"----shape of static {static.shape}")

        return [static, dynamic]
    
    
    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        # Run the model.
        gaussians = self.encoder(batch["context"], self.global_step, False)
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )
        target_gt = batch["target"]["image"]

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(output, batch, gaussians, self.global_step)
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss
        self.log("loss/total", total_loss)

        if self.global_rank == 0:
            print(
                f"train step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"loss = {total_loss:.6f}"
            )

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss


    def select_gaussians(self, gaussians, mask):
        gaussians.means = gaussians.means[:, mask,...]
        gaussians.covariances = gaussians.covariances[:, mask,...]  
        gaussians.harmonics =   gaussians.harmonics[:, mask,...]
        gaussians.opacities =   gaussians.opacities[:, mask]
        return gaussians  

    def combine_gaussians(self,set1, set2):
        means = torch.cat([set1.means, set2.means], dim=1)
        covariances = torch.cat([set1.covariances, set2.covariances], dim=1)
        harmonics = torch.cat([set1.harmonics, set2.harmonics], dim=1)
        opacities = torch.cat([set1.opacities, set2.opacities], dim=1)
        return Gaussians(
            means,
            covariances,
            harmonics,
            opacities,
        )

    def copy_gaussians(self, gaussians):
        return Gaussians(
            gaussians.means.clone(),
            gaussians.covariances.clone(),
            gaussians.harmonics.clone(),
            gaussians.opacities.clone(),
        )
    

    def test_step(self, batch_video, batch_idx):        
        frames = batch_video["t"]
        print(batch_video["context"]["far"].shape)
        
        
        batch: BatchedExample = self.mini_batch(batch_video, 0)
        batch = self.data_shim(batch)
        static, dynamic = self.get_mask_indices(batch["context"]["mask"])

        b, v, _, h, w = batch["target"]["image"].shape
        device = batch["target"]["image"].device
        dynamic.to(device)
        assert b == 1

        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            gaussians = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
            )
        static = self.select_gaussians(gaussians, static)
        
        for t in range(frames):
            batch: BatchedExample = self.mini_batch(batch_video, t)
            batch = self.data_shim(batch)
            _, dynamic = self.get_mask_indices(batch["context"]["mask"])

            b, v, _, h, w = batch["target"]["image"].shape
            device = batch["target"]["image"].device
            dynamic.to(device)
            assert b == 1

            # Render Gaussians.
            with self.benchmarker.time("encoder"):
                gaussians = self.encoder(
                    batch["context"],
                    self.global_step,
                    deterministic=False,
                )

            gaussians_dynamic = self.select_gaussians(gaussians, dynamic)
            # z = gaussians_dynamic.means[:,:,2]/2
            # gaussians_dynamic.means[:,:,2] = z

            gaussians_dynamic.opacities = gaussians_dynamic.opacities * 2
            gaussians = self.combine_gaussians(static, gaussians_dynamic)

                
                     
            with self.benchmarker.time("decoder", num_calls=v):
                color = []
                for i in range(0, batch["target"]["far"].shape[1], 32):
                    output = self.decoder.forward(
                        gaussians,
                        batch["target"]["extrinsics"][:1, i : i + 32],
                        batch["target"]["intrinsics"][:1, i : i + 32],
                        batch["target"]["near"][:1, i : i + 32],
                        batch["target"]["far"][:1, i : i + 32],
                        (h, w),
                    )
                    color.append(output.color)
                color = torch.cat(color, dim=1)

            # Save images.
            (scene,) = batch["scene"]
            name = get_cfg()["wandb"]["name"]
            path = self.test_cfg.output_path / name
            for index, color in zip(batch["target"]["index"][0], color[0]):
                save_image(color, path / scene / f"color/{t:0>6}.png")
            for index, color in zip(
                batch["context"]["index"][0], batch["context"]["image"][0]
            ):
                save_image(color, path / scene / f"context/{t:0>6}.png")

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(
            self.test_cfg.output_path / name / "peak_memory.json"
        )

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        gaussians_probabilistic = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=False,
        )
        output_probabilistic = self.decoder.forward(
            gaussians_probabilistic,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )
        rgb_probabilistic = output_probabilistic.color[0]
        gaussians_deterministic = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=True,
        )
        output_deterministic = self.decoder.forward(
            gaussians_deterministic,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )
        rgb_deterministic = output_deterministic.color[0]

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        for tag, rgb in zip(
            ("deterministic", "probabilistic"), (rgb_deterministic, rgb_probabilistic)
        ):
            psnr = compute_psnr(rgb_gt, rgb).mean()
            self.log(f"val/psnr_{tag}", psnr)
            lpips = compute_lpips(rgb_gt, rgb).mean()
            self.log(f"val/lpips_{tag}", lpips)
            ssim = compute_ssim(rgb_gt, rgb).mean()
            self.log(f"val/ssim_{tag}", ssim)

        # Construct comparison image.
        comparison = hcat(
            add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_probabilistic), "Target (Probabilistic)"),
            add_label(vcat(*rgb_deterministic), "Target (Deterministic)"),
        )
        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

        # Render projections and construct projection image.
        # These are disabled for now, since RE10k scenes are effectively unbounded.
        projections = vcat(
            hcat(
                *render_projections(
                    gaussians_probabilistic,
                    256,
                    extra_label="(Probabilistic)",
                )[0]
            ),
            hcat(
                *render_projections(
                    gaussians_deterministic, 256, extra_label="(Deterministic)"
                )[0]
            ),
            align="left",
        )
        self.logger.log_image(
            "projection",
            [prep_image(add_border(projections))],
            step=self.global_step,
        )

        # Draw cameras.
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=self.global_step
        )

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        # Run video validation step.
        self.render_video_interpolation(batch)
        self.render_video_wobble(batch)
        if self.train_cfg.extended_visualization:
            self.render_video_interpolation_exaggerated(batch)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step, False)
        gaussians_det = self.encoder(batch["context"], self.global_step, True)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]
        output_det = self.decoder.forward(
            gaussians_det, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_det = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_det.color[0], depth_map(output_det.depth[0]))
        ]
        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Probabilistic"),
                    add_label(image_det, "Deterministic"),
                )
            )
            for image_prob, image_det in zip(images_prob, images_det)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                # clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }
