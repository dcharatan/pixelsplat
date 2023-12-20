import json
from pathlib import Path

import hydra
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

METRICS = (
    ("psnr", "PSNR", 1),
    ("ssim", "SSIM", 1),
    ("lpips", "LPIPS", -1),
)


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="compute_metrics",
)
def generate_comparison_table(cfg_dict: DictConfig):
    cfg = load_typed_config(cfg_dict, RootCfg)
    with cfg.output_metrics_path.open("r") as f:
        metrics = json.load(f)

    table = {
        method.name: [
            metrics.get(f"{metric_key}_{method.key}") for metric_key, _, _ in METRICS
        ]
        for method in cfg.evaluation.methods
    }

    table = make_latex_table(
        table,
        [metric_name for _, metric_name, _ in METRICS],
        [2, 3, 3],
        [metric_order for _, _, metric_order in METRICS],
    )

    with Path("table.tex").open("w") as f:
        f.write(table)


if __name__ == "__main__":
    generate_comparison_table()
