import codecs
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Literal

import svg
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from src.misc.image_io import prep_image

# Constants
MARGIN = 2

Shape = tuple[int | float, int | float]


def encode_image(
    image: Float[Tensor, "3 height width"],
    image_format: Literal["png", "jpeg"] = "png",
) -> str:
    stream = BytesIO()
    Image.fromarray(prep_image(image)).save(stream, image_format)
    stream.seek(0)
    base64str = codecs.encode(stream.read(), "base64").rstrip()
    return f"data:image/{image_format};base64,{base64str.decode('ascii')}"


def save_svg(fig: svg.SVG, path: Path) -> None:
    with path.open("w") as f:
        # This hack makes embedded images work.
        f.write(
            str(fig)
            .replace("href", "xlink:href")
            .replace("<svg", '<svg xmlns:xlink="http://www.w3.org/1999/xlink"')
        )
    actual_width = float(
        subprocess.check_output(f"inkscape -D {path} --query-width".split(" "))
        .decode()
        .strip()
    )
    print(
        "When importing this SVG figure, make sure to multiply the width by "
        f"{actual_width / fig.width}"
    )
