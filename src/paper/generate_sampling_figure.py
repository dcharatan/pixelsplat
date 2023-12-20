import hydra
import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Bool, Float, install_import_hook
from lightning_fabric.utilities.apply_func import apply_to_collection
from scipy.spatial.transform import Rotation as R
from torch import Tensor
from torch.utils.data import default_collate
from tqdm import tqdm

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset import get_dataset
    from src.dataset.view_sampler.view_sampler_arbitrary import ViewSamplerArbitraryCfg
    from src.geometry.projection import (
        get_world_rays,
        homogenize_points,
        project,
        sample_image_grid,
    )
    from src.global_cfg import set_cfg
    from src.misc.image_io import save_image
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.decoder.cuda_splatting import render_cuda_orthographic
    from src.model.encoder import get_encoder
    from src.model.encoder.epipolar.conversions import depth_to_relative_disparity
    from src.model.model_wrapper import ModelWrapper
    from src.visualization.drawing.cameras import unproject_frustum_corners
    from src.visualization.drawing.lines import draw_lines
    from src.visualization.drawing.points import draw_points


SCENES = (
    # scene, context 1, context 2, far plane
    ("fc60dbb610046c56", 0, 115, 9.0),
)
FIGURE_WIDTH = 500
MARGIN = 4
LINE_WIDTH = 3
LINE_COLOR = [0, 0, 0]
POINT_DENSITY = 1.0
ANGLE = 30
RESOLUTION = 1536
SAMPLES_PER_RAY = 2048


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="main",
)
def generate_point_cloud_figure(cfg_dict):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)
    device = torch.device("cuda:0")

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    model_wrapper = ModelWrapper.load_from_checkpoint(
        checkpoint_path,
        optimizer_cfg=cfg.optimizer,
        test_cfg=cfg.test,
        train_cfg=cfg.train,
        encoder=encoder,
        encoder_visualizer=encoder_visualizer,
        decoder=get_decoder(cfg.model.decoder, cfg.dataset),
        losses=[],
        step_tracker=None,
    )
    model_wrapper.eval()

    for scene, *context_indices, far in SCENES:
        # Create a dataset that always returns the desired scene.
        view_sampler_cfg = ViewSamplerArbitraryCfg(
            "arbitrary",
            2,
            2,
            context_views=list(context_indices),
            target_views=list(context_indices),
        )
        cfg.dataset.view_sampler = view_sampler_cfg
        cfg.dataset.overfit_to_scene = scene

        # Get the scene.
        dataset = get_dataset(cfg.dataset, "test", None)
        example = default_collate([next(iter(dataset))])
        example = apply_to_collection(example, Tensor, lambda x: x.to(device))

        pdf = None
        offset = None

        def hook(module, input, output):
            nonlocal pdf
            pdf = output

        def hook_offset(module, input, output):
            nonlocal offset
            offset = output

        # Register hook to grab PDF.
        encoder.depth_predictor.to_pdf.register_forward_hook(hook)
        encoder.depth_predictor.to_offset.register_forward_hook(hook_offset)

        # Generate the Gaussians.
        gaussians = encoder.forward(example["context"], False)

        # Figure out which Gaussians to mask off/throw away.
        _, _, _, h, w = example["context"]["image"].shape

        # Get the Gaussian means.
        means = rearrange(
            gaussians.means, "() (v h w spp) xyz -> h w spp v xyz", v=2, h=h, w=w
        )

        # Screw with the intrinsics a bit to make the result look clearer.
        k = example["context"]["intrinsics"][0]
        k[..., :2, :2] *= 256 / (256 - 16)

        # Compute frustum corners for the context views.
        frustum_corners = unproject_frustum_corners(
            example["context"]["extrinsics"][0],
            k,
            torch.ones((2,), dtype=torch.float32, device=device) * far,
        )
        camera_origins = example["context"]["extrinsics"][0, :, :3, 3]

        def is_in_frustum(point: Float[Tensor, "*batch 3"]) -> Bool[Tensor, " *batch"]:
            is_in_frustum_mask = torch.ones_like(point[..., 0], dtype=torch.bool)

            # Drop Gaussians that are outside the first frustum.
            for i in range(4):
                ab = frustum_corners[0, i - 1] - frustum_corners[0, i]
                ac = camera_origins[0] - frustum_corners[0, i]
                plane = ab.cross(ac)
                test_vector = camera_origins[0] - point
                dot = einsum(test_vector, plane, "... xyz, xyz -> ...")
                is_in_frustum_mask &= dot > 0

            # Drop Gaussians that are too far away.
            ab = frustum_corners[0, 0] - frustum_corners[0, 1]
            ac = frustum_corners[0, 2] - frustum_corners[0, 1]
            plane = ac.cross(ab)
            test_vector = frustum_corners[0, 0] - point
            dot = einsum(test_vector, plane, "... xyz, xyz -> ...")
            is_in_frustum_mask &= dot > 0

            return is_in_frustum_mask

        # Create a mask to filter the Gaussians.
        mask = is_in_frustum(means)

        def trim(element):
            element = rearrange(
                element, "() (v h w spp) ... -> h w spp v ...", v=2, h=h, w=w
            )
            return element[mask][None]

        # Define the pose we render from.
        pose = torch.eye(4, dtype=torch.float32, device=device)
        rotation = R.from_euler("xyz", [-15, ANGLE - 90, 0], True).as_matrix()
        pose[:3, :3] = torch.tensor(rotation, dtype=torch.float32, device=device)
        translation = torch.eye(4, dtype=torch.float32, device=device)
        # visual balance, 0.5x pyramid/frustum volume
        translation[2, 3] = far * (0.5 ** (1 / 3))
        pose = translation @ pose

        ones = torch.ones((1,), dtype=torch.float32, device=device)
        render_args = {
            "extrinsics": example["context"]["extrinsics"][0, :1] @ pose,
            "width": ones * far * 2,
            "height": ones * far * 2,
            "near": ones * 0,
            "far": ones * far,
            "image_shape": (RESOLUTION, RESOLUTION),
            "background_color": torch.zeros((1, 3), dtype=torch.float32, device=device),
            "gaussian_means": trim(gaussians.means),
            "gaussian_covariances": trim(gaussians.covariances),
            "gaussian_sh_coefficients": trim(gaussians.harmonics),
            "gaussian_opacities": trim(gaussians.opacities),
            "fov_degrees": 0.1,
        }

        # Render alpha (opacity).
        dump = {}
        alpha_args = {
            **render_args,
            "gaussian_sh_coefficients": torch.ones_like(
                render_args["gaussian_sh_coefficients"][..., :1]
            ),
            "use_sh": False,
        }
        alpha = render_cuda_orthographic(**alpha_args, dump=dump)[0]

        # Render (premultiplied) color.
        color = render_cuda_orthographic(**render_args)[0]

        # Render depths. Without modifying the renderer, we can only render
        # premultiplied depth, then hackily transform it into straight alpha depth,
        # which is needed for sorting.
        depth = render_args["gaussian_means"] - dump["extrinsics"][0, :3, 3]
        depth = depth.norm(dim=-1)
        depth_args = {
            **render_args,
            "gaussian_sh_coefficients": repeat(depth, "() g -> () g c ()", c=3),
            "use_sh": False,
        }
        depth_premultiplied = render_cuda_orthographic(**depth_args)
        depth = (depth_premultiplied / alpha).nan_to_num(posinf=1e10, nan=1e10)[0]

        # Save the rendering for later depth-based alpha compositing.
        layers = [(color, alpha, depth)]

        # Figure out the intrinsics from the FOV.
        fx = 0.5 / (0.5 * dump["fov_x"]).tan()
        fy = 0.5 / (0.5 * dump["fov_y"]).tan()
        dump_intrinsics = torch.eye(3, dtype=torch.float32, device=device)
        dump_intrinsics[0, 0] = fx
        dump_intrinsics[1, 1] = fy
        dump_intrinsics[:2, 2] = 0.5

        # Generate the 3D lines that have to be computed.
        lines = []
        for corners, origin in zip(frustum_corners, camera_origins):
            for i in range(4):
                lines.append((corners[i], corners[i - 1]))
                lines.append((corners[i], origin))

            # add the extra line we want to draw
            # dx = 0.5
            # dy = 0.4
            # x00 = corners[0]
            # x01 = corners[1]
            # x10 = corners[2]
            # x11 = corners[3]
            # x0 = x00 * dx + x01 * (1 - dx)
            # x1 = x10 * dx + x11 * (1 - dx)
            # xx = x0 * dy + x1 * (1 - dy)
            # lines.append((xx, origin))

            # Only draw one frustum
            break

        # Generate an alpha compositing layer for each line.
        only_lines = []
        for line_idx, (a, b) in enumerate(lines):
            special = False
            lw = LINE_WIDTH * 4 if special else LINE_WIDTH
            lc = torch.tensor(LINE_COLOR, dtype=torch.float32, device=device)
            if special:
                lc = torch.tensor((66, 99, 235), device=device) / 255

            # Start with the point whose depth is further from the camera.
            a_depth = (dump["extrinsics"].inverse() @ homogenize_points(a))[..., 2]
            b_depth = (dump["extrinsics"].inverse() @ homogenize_points(b))[..., 2]
            start = a if (a_depth > b_depth).all() else b
            end = b if (a_depth > b_depth).all() else a

            # Create the alpha mask (this one is clean).
            start_2d = project(start, dump["extrinsics"], dump_intrinsics)[0][0]
            end_2d = project(end, dump["extrinsics"], dump_intrinsics)[0][0]
            alpha = draw_lines(
                torch.zeros_like(color),
                start_2d[None],
                end_2d[None],
                (1, 1, 1),
                lw,
                x_range=(0, 1),
                y_range=(0, 1),
            )

            # Create the color.
            color = draw_lines(
                torch.zeros_like(color),
                start_2d[None],
                end_2d[None],
                lc,
                lw,
                x_range=(0, 1),
                y_range=(0, 1),
            )

            # Create the depth. We just individually render points.
            wh = torch.tensor(
                (RESOLUTION, RESOLUTION), dtype=torch.float32, device=device
            )
            delta = (wh * (start_2d - end_2d)).norm()
            num_points = delta / POINT_DENSITY
            t = torch.linspace(0, 1, int(num_points) + 1, device=device)
            xyz = start[None] * t[:, None] + end[None] * (1 - t)[:, None]
            depth = (xyz - dump["extrinsics"][0, :3, 3]).norm(dim=-1)
            depth = repeat(depth, "p -> p c", c=3)
            xy = project(xyz, dump["extrinsics"], dump_intrinsics)[0]
            depth = draw_points(
                torch.ones_like(color) * 1e10,
                xy,
                depth,
                lw,  # makes it 2x as wide as line
                x_range=(0, 1),
                y_range=(0, 1),
            )

            layers.append((color, alpha, depth))
            only_lines.append((color.clone(), alpha.clone(), depth.clone()))

        # Do the alpha compositing.
        canvas = torch.ones_like(color)
        colors = torch.stack([x for x, _, _ in layers])
        alphas = torch.stack([x for _, x, _ in layers])
        depths = torch.stack([x for _, _, x in layers])
        index = depths.argsort(dim=0)
        colors = colors.gather(index=index, dim=0)
        alphas = alphas.gather(index=index, dim=0)
        t = (1 - alphas).cumprod(dim=0)
        t = torch.cat([torch.ones_like(t[:1]), t[:-1]], dim=0)
        image = (t * colors).sum(dim=0)
        total_alpha = (t * alphas).sum(dim=0)
        image = total_alpha * image + (1 - total_alpha) * canvas

        print("Saving Gaussians!")
        save_image(image, "sampling_figure/gaussians.png")

        # Get rid of junk. This is trash-tier code but it's a throwaway
        del gaussians
        del a
        del image
        del colors
        del alphas
        del index
        del t
        del lines
        del alpha
        del delta
        del start_2d
        del end_2d
        del alpha_args
        del a_depth
        del b
        del b_depth
        del canvas
        del corners
        del depth
        del depth_premultiplied
        del depths
        del encoder
        del encoder_visualizer
        del end

        # Generate camera rays.
        xy, _ = sample_image_grid((RESOLUTION, RESOLUTION), device)
        origins, directions = get_world_rays(xy, dump["extrinsics"][0], dump_intrinsics)
        origins = rearrange(origins, "h w xyz -> (h w) xyz")
        directions = rearrange(directions, "h w xyz -> (h w) xyz")
        t = torch.linspace(0, 1, SAMPLES_PER_RAY, device=device)
        far = dump["far"] + 10
        near = dump["near"] - 10
        t = t * (far - near) + near

        # preprocess the pdf
        SUBDIVISION = 32
        pdf = rearrange(pdf[0, 0], "(h w) () d -> h w d", h=h, w=w)
        offset = rearrange(offset[0, 0], "(h w) () d -> h w d", h=h, w=w)
        chosen = (offset * SUBDIVISION).type(torch.int64).clip(max=SUBDIVISION - 1)
        pdf = repeat(pdf, "h w d -> h w (d sd)", sd=SUBDIVISION)
        msk = torch.zeros_like(repeat(offset, "h w d -> sd h w d", sd=SUBDIVISION))
        d = msk.shape[-1]
        ih = repeat(torch.arange(h, device=device), "h -> h w d", h=h, w=w, d=d)
        iw = repeat(torch.arange(w, device=device), "w -> h w d", h=h, w=w, d=d)
        id = repeat(torch.arange(d, device=device), "d -> h w d", h=h, w=w, d=d)
        msk[chosen, ih, iw, id] = 1
        pdf = pdf * rearrange(msk, "sd h w d -> h w (d sd)")

        rendered_density = []
        rendered_depth = []
        for r_o, r_d in zip(tqdm(origins.split(512)), directions.split(512)):
            # Generate sample locations.
            xyz = r_o[:, None] + r_d[:, None] * t[:, None]

            # test: just render the frustum itself
            # density = is_in_frustum(xyz).float() * 100

            # Get the actual densities.
            # First, get XYZ in camera space.
            extr = example["context"]["extrinsics"][0, 0]
            intr = example["context"]["intrinsics"][0, 0]
            dpt = (xyz - extr[:3, 3]).norm(dim=-1)
            dpt = depth_to_relative_disparity(
                dpt, example["context"]["near"][0, 0], example["context"]["far"][0, 0]
            )
            img_xy, vv = project(xyz, extr, intr)
            valid = (
                vv
                & (0 < dpt)
                & (dpt < 1)
                & (img_xy > 0).all(dim=-1)
                & (img_xy < 1).all(dim=-1)
            )
            n = pdf.shape[-1]
            dpt_bucket = (dpt * n).type(torch.int64).clip(min=0, max=n - 1)
            x_bucket = (img_xy[..., 0] * w).type(torch.int64).clip(min=0, max=w - 1)
            y_bucket = (img_xy[..., 1] * h).type(torch.int64).clip(min=0, max=h - 1)
            sampled_pdf = pdf[y_bucket, x_bucket, dpt_bucket]
            density = sampled_pdf * valid * is_in_frustum(xyz)

            # volume rendering
            # segment_length = (t[1:] - t[:-1]).mean()
            # alpha = 1 - torch.exp(-density * segment_length)
            # tm = (1 - alpha).cumprod(dim=-1)
            # tm = torch.cat([torch.ones_like(tm[:, :1]), tm[:, :-1]], dim=-1)
            # weight = alpha * tm
            # result = weight.sum(dim=-1)

            # just take the max lol
            result, i_depth = density.max(dim=-1)
            depth = t[i_depth] + 0.01  # add offset so line wins
            depth[result < 0.05] = 1e20

            rendered_density.append(result)
            rendered_depth.append(depth)

        rendered_density = torch.cat(rendered_density)
        rendered_density = repeat(
            rendered_density, "(h w) -> c h w", h=RESOLUTION, w=RESOLUTION, c=3
        )

        rendered_depth = torch.cat(rendered_depth)
        rendered_depth = repeat(
            rendered_depth, "(h w) -> c h w", h=RESOLUTION, w=RESOLUTION, c=3
        )

        # Do the alpha compositing.
        ccc = torch.tensor((80, 80, 80), device=device) / 255
        only_lines.insert(
            0,
            (
                torch.ones_like(rendered_density) * ccc[:, None, None],
                rendered_density,
                rendered_depth,
            ),
        )
        canvas = torch.ones_like(color)
        colors = torch.stack([x for x, _, _ in only_lines])
        alphas = torch.stack([x for _, x, _ in only_lines])
        depths = torch.stack([x for _, _, x in only_lines])
        index = depths.argsort(dim=0)
        colors = colors.gather(index=index, dim=0)
        alphas = alphas.gather(index=index, dim=0)
        t = (1 - alphas).cumprod(dim=0)
        t = torch.cat([torch.ones_like(t[:1]), t[:-1]], dim=0)
        image = (t * colors).sum(dim=0)
        total_alpha = (t * alphas).sum(dim=0)
        image = total_alpha * image + (1 - total_alpha) * canvas

        print("Saving Composite!")
        save_image(image, "sampling_figure/density.png")

        a = 1


if __name__ == "__main__":
    with torch.no_grad():
        generate_point_cloud_figure()
