import json
from pathlib import Path

import hydra
import numpy as np
from jaxtyping import install_import_hook
from omegaconf import DictConfig

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_config
    from src.paper.table import make_latex_table
    from src.scripts.compute_metrics import RootCfg


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="compute_metrics",
)
def generate_table_comparison(cfg_dict: DictConfig):
    cfg = load_typed_config(cfg_dict, RootCfg)
    table = {}
    for method in cfg.evaluation.methods:
        # Add the rendering time metrics.
        try:
            with (method.path / "benchmark.json").open("r") as f:
                benchmark = json.load(f)
            encoder_time = np.mean(benchmark.get("encoder", [0]))
            decoder_time = np.mean(benchmark.get("decoder", [0]))
            if np.isclose(decoder_time, 0):
                decoder_time = None
        except FileNotFoundError:
            print(f"Warning: Could not load benchmark for {method.key}.")
            encoder_time = None
            decoder_time = None

        # Add memory consumption metric.
        try:
            with (method.path / "peak_memory.json").open("r") as f:
                peak_memory = json.load(f) / 1e9
        except FileNotFoundError:
            print(f"Warning: Could not load peak memory for {method.key}.")
            peak_memory = None

        table[method.name] = [encoder_time, decoder_time, peak_memory]

    table = make_latex_table(
        table,
        ["Encoding (s)", "Decoding (s)", "VRAM (GB)"],
        [3, 3, 3],
        [-1, -1, -1],
    )

    with Path("table.tex").open("w") as f:
        f.write(table)


if __name__ == "__main__":
    generate_table_comparison()
