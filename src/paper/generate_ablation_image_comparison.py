from pathlib import Path

import hydra
import svg
import torch
from jaxtyping import Float, install_import_hook
from omegaconf import DictConfig
from torch import Tensor
from tqdm import tqdm

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_config
    from src.misc.image_io import load_image
    from src.paper.common import MARGIN, encode_image, save_svg
    from src.scripts.compute_metrics import RootCfg

FIGURE_WIDTH = 240
LINE_WIDTH = 0.5
TEXT_SIZE = 10


def generate_image_grid(
    images: list[list[Float[Tensor, "3 height width"] | None]],
    method_names: list[str],
):
    num_rows = len(images)
    num_cols = len(images[0])

    # There are two extra margins for the vertical lines.
    image_width = (FIGURE_WIDTH - (num_cols - 1) * MARGIN) / num_cols
    figure_height = num_rows * image_width + (num_rows - 1) * MARGIN + TEXT_SIZE

    # Setting width and height seems to be broken, so we manually set them here.
    fig = svg.SVG(
        width=FIGURE_WIDTH,
        height=figure_height,
        elements=[],
        viewBox=svg.ViewBoxSpec(0, 0, FIGURE_WIDTH, figure_height),
    )

    for row, row_images in enumerate(images):
        for col, image in enumerate(row_images):
            if image is None:
                image = torch.ones((3, 128, 128), dtype=torch.float32) * 0.5

            # For now, assume square images.
            _, h, w = image.shape
            assert h == w

            # Compute values needed to determine the image's position.
            offset = image_width + MARGIN
            image = svg.Image(
                width=image_width,
                height=image_width,
                href=encode_image(image, "jpeg"),
                x=offset * col,
                y=offset * row + TEXT_SIZE,
            )
            fig.elements.append(image)

    # Draw the method names.
    for i, method_name in enumerate(method_names):
        text = svg.Text(
            x=image_width * (0.5 + i) + MARGIN * i,
            y=TEXT_SIZE * 0.65,
            elements=[method_name],
            font_size=TEXT_SIZE,
            text_anchor="middle",
        )
        fig.elements.append(text)

    save_svg(fig, Path("ablation.svg"))


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="compute_metrics",
)
def generate_image_comparison(cfg_dict: DictConfig):
    cfg = load_typed_config(cfg_dict, RootCfg)
    rows = []
    for highlighted in tqdm(cfg.evaluation.highlighted, "Loading data"):
        scene = highlighted.scene
        target_index = highlighted.target_index

        # Add the rendered frames to the grid.
        row = []
        for method in cfg.evaluation.methods:
            try:
                image = load_image(
                    method.path / scene / f"color/{target_index:0>6}.png"
                )
            except FileNotFoundError:
                image = None

            row.append(image)
        rows.append(row)

    generate_image_grid(rows, [method.name for method in cfg.evaluation.methods])


if __name__ == "__main__":
    generate_image_comparison()
