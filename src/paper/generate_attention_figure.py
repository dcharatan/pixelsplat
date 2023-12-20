from pathlib import Path

import hydra
import svg
import torch
from einops import rearrange, reduce
from jaxtyping import Float, install_import_hook
from lightning_fabric.utilities.apply_func import apply_to_collection
from torch import Tensor
from torch.utils.data import default_collate

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset import get_dataset
    from src.dataset.view_sampler.view_sampler_arbitrary import ViewSamplerArbitraryCfg
    from src.global_cfg import set_cfg
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper
    from src.paper.common import encode_image, save_svg
    from src.visualization.colors import get_distinct_color


SCENE = "2177ca3a775a9ee9"
CONTEXT_INDICES = (135, 195)
QUERIES = (
    # x, y
    (238, 168),  # sofa pillow corner
    (238, 80),  # painting corner
    (159, 195),  # plant leaves
    (227, 277),  # carpet corner
    (300, 80),  # random spot on wall
)
QUERIES = tuple((x / 400, y / 400) for x, y in QUERIES)
LAYER = 1
HEAD = 2
IMAGE_SHAPE = (256, 256)
FIGURE_WIDTH = 240
MARGIN = 4
LINE_WIDTH = 4
RAY_RADIUS = 2
RAY_BACKER_RADIUS = 2.5


def to_hex(color: Float[Tensor, "3"]) -> str:
    r, g, b = color.tolist()
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="main",
)
def generate_attention_figure(cfg_dict):
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
    dataset = iter(get_dataset(cfg.dataset, "test", None))

    # Create a dataset that always returns the desired scene.
    view_sampler_cfg = ViewSamplerArbitraryCfg(
        "arbitrary",
        2,
        2,
        context_views=list(CONTEXT_INDICES),
        target_views=list(CONTEXT_INDICES),
    )
    cfg.dataset.view_sampler = view_sampler_cfg
    cfg.dataset.overfit_to_scene = SCENE

    # Get the scene.
    dataset = get_dataset(cfg.dataset, "test", None)
    example = default_collate([next(iter(dataset))])
    example = apply_to_collection(example, Tensor, lambda x: x.to(device))

    # Run the encoder with hooks to capture the attention output.
    softmax_weights = []

    def hook(module, input, output):
        softmax_weights.append(output)

    handles = [
        layer[0].fn.attend.register_forward_hook(hook)
        for layer in encoder.epipolar_transformer.transformer.layers
    ]
    visualization_dump = {}
    encoder.forward(example["context"], False, visualization_dump=visualization_dump)
    for handle in handles:
        handle.remove()

    attention = torch.stack(softmax_weights)
    sampling = visualization_dump["sampling"]
    context_images = example["context"]["image"]

    # Pick a random batch element, view, and other view.
    _, _, _, h, w = context_images.shape
    ds = cfg.model.encoder.epipolar_transformer.downscale
    wh = torch.tensor((w // ds, h // ds), dtype=torch.float32, device=device)
    queries = torch.tensor(QUERIES, dtype=torch.float32, device=device) * wh
    col, row = queries.type(torch.int64).unbind(dim=-1)
    rr = row * (w // ds) + col

    b, v, _, r, s, _ = sampling.xy_sample.shape
    rb = 0
    rv = 0
    rov = 0

    # Visualize attention in the sample view.
    attention = rearrange(attention, "l (b v r) hd () s -> l b v r hd s", b=b, v=v, r=r)
    attention = attention[:, rb, rv, rr, :, :]

    # Create colors according to attention.
    color = [get_distinct_color(i) for i, _ in enumerate(rr)]
    color = torch.tensor(color, device=attention.device)
    color = rearrange(color, "r c -> r () c")
    attn = rearrange(attention[LAYER, :, HEAD], "r s -> r s ()")
    attn = attn / reduce(attn, "r s () -> r () ()", "max")

    left_image = context_images[rb, rv]
    right_image = context_images[rb, encoder.sampler.index_v[rv, rov]]

    # Generate the SVG.
    # Create an SVG canvas.
    image_width = (FIGURE_WIDTH - MARGIN) / 2
    image_height = image_width * IMAGE_SHAPE[0] / IMAGE_SHAPE[1]
    fig = svg.SVG(
        width=FIGURE_WIDTH,
        height=image_height,
        elements=[],
        viewBox=svg.ViewBoxSpec(0, 0, FIGURE_WIDTH, image_height),
    )

    # Draw the left image.
    left_image = svg.Image(
        width=image_width,
        height=image_height,
        href=encode_image(left_image, "jpeg"),
    )
    fig.elements.append(left_image)

    # Draw the right image.
    right_image = svg.Image(
        width=image_width,
        height=image_height,
        href=encode_image(right_image, "jpeg"),
        x=image_width + MARGIN,
    )
    fig.elements.append(right_image)

    # Create a mask for the epipolar line.
    mask = svg.Mask(
        elements=[svg.Rect(width=FIGURE_WIDTH, height=image_height, fill="white")],
        id="mask",
        maskUnits="userSpaceOnUse",
    )
    fig.elements.append(mask)

    scale = torch.tensor(
        (image_width, image_height), dtype=torch.float32, device=device
    )
    for rrv_idx, rrv in enumerate(rr):
        # Draw the sample.
        ray_xy = (sampling.xy_ray[rb, rv, rrv] * scale).tolist()
        ray = svg.Circle(
            cx=ray_xy[0],
            cy=ray_xy[1],
            r=RAY_BACKER_RADIUS,
            fill="#000000",
        )
        fig.elements.append(ray)
        ray = svg.Circle(
            cx=ray_xy[0],
            cy=ray_xy[1],
            r=RAY_RADIUS,
            fill=to_hex(color[rrv_idx, 0]),
        )
        fig.elements.append(ray)

        # Draw the epipolar line.
        start = (sampling.xy_sample_near[rb, rv, rov, rrv, 0] * scale).tolist()
        start[0] += image_width + MARGIN
        end = (sampling.xy_sample_far[rb, rv, rov, rrv, -1] * scale).tolist()
        end[0] += image_width + MARGIN
        epipolar_line = svg.Line(
            x1=2 * start[0] - end[0],  # extra length that gets clipped
            y1=2 * start[1] - end[1],  # extra length that gets clipped
            x2=end[0],
            y2=end[1],
            stroke="#000000",
            stroke_width=LINE_WIDTH,
            mask="url(#mask)",
        )
        fig.elements.append(epipolar_line)

        # Draw lines for attention.
        for sv in range(s):
            start = (sampling.xy_sample_near[rb, rv, rov, rrv, sv] * scale).tolist()
            start[0] += image_width + MARGIN
            end = (sampling.xy_sample_far[rb, rv, rov, rrv, sv] * scale).tolist()
            end[0] += image_width + MARGIN
            epipolar_line = svg.Line(
                x1=start[0],
                y1=start[1],
                x2=end[0],
                y2=end[1],
                stroke=to_hex((color * attn)[rrv_idx, sv]),
                stroke_width=LINE_WIDTH,
                mask="url(#mask)",
            )
            fig.elements.append(epipolar_line)

    save_svg(fig, Path("attention.svg"))


if __name__ == "__main__":
    generate_attention_figure()
