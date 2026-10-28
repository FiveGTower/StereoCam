
"""Run Trinocular and automatically evaluate MAE/RMSE."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


# ================================================
# ONLY CHANGE THIS PATH FOR EACH EXPERIMENT
# ================================================

CASE_PARAMETERS = (
    r"D:/course_4/project_geoscan/StereoCam/Scenes"
    r"/fov835_baseline_02/Parameters.yaml"
)

# Blender GT convention: verify before final evaluation.
# "range": distance along viewing ray
# "axial": depth along camera optical axis
GT_DEPTH_TYPE = "range"

# ================================================


def resolve_path(base, path_string):
    p = Path(path_string).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=CASE_PARAMETERS
    )

    parser.add_argument(
        "--gt-depth-type",
        default=GT_DEPTH_TYPE,
        choices=["range", "axial"]
    )

    args = parser.parse_args()

    config_path = Path(args.config).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Parameters.yaml not found: {config_path}"
        )

    case_dir = config_path.parent

    cfg = yaml.safe_load(
        config_path.read_text(encoding="utf-8-sig")
    )

    if not isinstance(cfg, dict):
        raise ValueError("Invalid Parameters.yaml")

    # Check three input images.
    input_keys = {
        "L": "center",
        "R": "right",
        "SL": "down",
    }

    for camera, key in input_keys.items():
        image_path = resolve_path(
            case_dir,
            cfg["inputs"][key]
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Camera {camera}: {image_path}"
            )

    # Find result directory from YAML.
    result_dir = resolve_path(
        case_dir,
        cfg["experiment"]["output_dir"]
    )

    # Find ground truth from YAML.
    gt_cfg = cfg.get("ground_truth") or {}

    groundtruth_path = resolve_path(
        case_dir,
        gt_cfg.get(
            "depth_exr_path"
        ) or "./groundtruth.exr"
    )

    if not groundtruth_path.is_file():
        raise FileNotFoundError(
            f"Ground truth not found: {groundtruth_path}"
        )

    if not cfg["output"].get("save_preview_png", False):
        raise ValueError(
            "Set output.save_preview_png: true"
        )

    if not cfg["output"].get("save_exr", True):
        raise ValueError(
            "Set output.save_exr: true"
        )

    print("\n==============================")
    print("CASE:", case_dir.name)
    print("CONFIG:", config_path)
    print("GROUND TRUTH:", groundtruth_path)
    print("==============================\n")

    # 1. RUN EXISTING TRINOCULAR ALGORITHM
    # Original run.py has been renamed to run_core.py.
    from run_core import run as run_trinocular

    run_trinocular(config_path)

    # 2. CREATE REQUESTED OUTPUT FILENAMES
    aliases = {
        "trinocular_depth_preview.png": "depthmap.png",
        "trinocular_disparity_preview.png": "disparitymap.png",
        "trinocular_depth_m.exr": "depthmap.exr",
    }

    for old_name, new_name in aliases.items():
        source = result_dir / old_name
        destination = result_dir / new_name

        if not source.is_file():
            raise FileNotFoundError(
                f"Missing algorithm output: {source}"
            )

        shutil.copy2(source, destination)

        print("Saved:", destination)

    # 3. EVALUATE AGAINST GROUND TRUTH
    evaluate_script = (
        Path(__file__).resolve().parent
        / "evaluate_depth.py"
    )

    if not evaluate_script.is_file():
        raise FileNotFoundError(
            f"Missing evaluate_depth.py: {evaluate_script}"
        )

    command = [
        sys.executable,
        str(evaluate_script),
        "--results",
        str(result_dir),
        "--gt",
        str(groundtruth_path),
        "--gt-depth-type",
        args.gt_depth_type,
    ]

    subprocess.run(command, check=True)

    # 4. CHECK EVALUATION REPORT
    report_path = (
        result_dir / "evaluation_metrics.txt"
    )

    if not report_path.is_file():
        raise FileNotFoundError(
            "evaluation_metrics.txt was not created"
        )

    print("\n==============================")
    print("EXPERIMENT COMPLETED")
    print("==============================")
    print("Depth PNG:", result_dir / "depthmap.png")
    print("Disparity PNG:", result_dir / "disparitymap.png")
    print("Depth EXR:", result_dir / "depthmap.exr")
    print("Metrics:", report_path)


if __name__ == "__main__":
    main()
