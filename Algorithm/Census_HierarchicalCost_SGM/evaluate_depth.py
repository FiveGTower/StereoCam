"""Evaluate trinocular predicted depth against Blender Z/Depth EXR.

Requires: numpy, opencv-python, PyYAML; OpenEXR>=3.3 for multilayer GT.
Results are stored beside predicted depth as evaluation_metrics.txt.

Example from project folder:
  py evaluate_depth.py --results results/fov835_baseline01 --gt ../trinocular_dataset_fov835_baseline01/Groundtruth.exr --gt-depth-type range
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Enable OpenEXR BEFORE importing cv2.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np
import yaml


def abs_path(s: str | Path, base: Path) -> Path:
    p = Path(s).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def read_exr_depth(path: Path, gt: bool) -> tuple[np.ndarray, str]:
    """Use actual EXR channel names; never interpret arbitrary RGB as depth."""
    if not path.is_file():
        raise FileNotFoundError(f"EXR not found: {path}")
    try:
        import OpenEXR
    except ImportError:
        OpenEXR = None

    if OpenEXR is not None and hasattr(OpenEXR, "File"):
        file = OpenEXR.File(str(path), separate_channels=True)
        for part in file.parts:
            channels = part.channels
            if gt:
                # Blender multi-layer: ViewLayer.Depth.Z (or ViewLayer.Depth).
                ranked = []
                for name in channels:
                    low = name.lower()
                    if "denois" in low or "mist" in low:
                        continue
                    if low in ("z", "depth") or low.endswith(".depth.z"):
                        rank = 3
                    elif low.endswith(".depth"):
                        rank = 2
                    elif low.endswith(".z"):
                        rank = 1
                    else:
                        continue
                    ranked.append((rank, name))
                if ranked:
                    selected = max(ranked)[1]
                elif len(file.parts) == 1 and len(channels) == 1:
                    selected = next(iter(channels))
                else:
                    continue
            else:
                if len(channels) != 1:
                    raise ValueError(f"Predicted EXR must have one channel: {list(channels)}")
                selected = next(iter(channels))
            pixels = np.asarray(channels[selected].pixels, dtype=np.float32)
            if pixels.ndim != 2:
                raise ValueError(f"EXR channel {selected} is not 2D: {pixels.shape}")
            return pixels, selected
        raise ValueError(f"No Z/Depth channel in {path}. Check Blender Depth Pass")

    # OpenCV fallback for single-channel EXRs (e.g. existing L_0001_depth.exr).
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 2 or not np.issubdtype(img.dtype, np.floating):
        raise RuntimeError("Cannot identify raw depth channel. Install: py -m pip install OpenEXR>=3.3")
    return img.astype(np.float32), "single channel (OpenCV)"


def camera_intrinsics(results: Path):
    """Read the actual intrinsics from run.py's config snapshot when present."""
    snapshot = results / "config_snapshot.yaml"
    if not snapshot.is_file():
        return 896.32, 896.32, 800.0, 650.0, "default for 1600x1300 FOV=83.5"
    cfg = yaml.safe_load(snapshot.read_text(encoding="utf-8-sig"))
    reference = cfg["rig"].get("reference_camera", "L")
    cam = cfg["cameras"][reference]
    return (float(cam["fx_px"]), float(cam["fy_px"]),
            float(cam["cx_px"]), float(cam["cy_px"]),
            f"{snapshot.name} / {reference}")


def range_to_axial(z_range, fx, fy, cx, cy):
    """Convert camera-to-surface ray distance into depth along optical axis."""
    h, w = z_range.shape
    x = (np.arange(w, dtype=np.float64) + 0.5 - cx) / fx
    y = (np.arange(h, dtype=np.float64) + 0.5 - cy) / fy
    factor = np.sqrt(1.0 + x[None, :] ** 2 + y[:, None] ** 2)
    return (z_range / factor).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Evaluate MAE/RMSE from Blender Groundtruth EXR")
    parser.add_argument("--results", default="results/fov835_baseline01")
    parser.add_argument("--gt", required=True, help="Groundtruth.exr or L_0001_depth.exr")
    parser.add_argument("--gt-depth-type", choices=("range", "axial"), default="range")
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=100.0)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    results = abs_path(args.results, base)
    gt_path = abs_path(args.gt, base)
    if not results.is_dir():
        raise FileNotFoundError(f"Results folder not found: {results}")
    if not (0 <= args.min_depth < args.max_depth):
        raise ValueError("Invalid min/max depth range")

    gt, gt_channel = read_exr_depth(gt_path, gt=True)
    pred_exr = results / "trinocular_depth_m.exr"
    pred_npy = results / "trinocular_depth_m.npy"
    if pred_exr.is_file():
        pred, pred_channel = read_exr_depth(pred_exr, gt=False)
        pred_path = pred_exr
    elif pred_npy.is_file():
        pred = np.load(pred_npy, allow_pickle=False).astype(np.float32)
        pred_path, pred_channel = pred_npy, "NumPy depth array"
        print("INFO: EXR prediction not found; using existing .npy depth instead")
    else:
        raise FileNotFoundError(f"Neither {pred_exr.name} nor {pred_npy.name} exists in {results}")
    if gt.shape != pred.shape:
        raise ValueError(f"Depth size mismatch! GT={gt.shape}, pred={pred.shape}; do not resize")

    fx, fy, cx, cy, intr_source = camera_intrinsics(results)
    if args.gt_depth_type == "range":
        gt = range_to_axial(gt, fx, fy, cx, cy)

    valid_gt = np.isfinite(gt) & (gt >= args.min_depth) & (gt <= args.max_depth)
    valid_pred = np.isfinite(pred) & (pred >= args.min_depth) & (pred <= args.max_depth)
    valid = valid_gt & valid_pred
    count = int(np.count_nonzero(valid))
    if count == 0:
        raise ValueError("No overlapping valid pixels! Check EXR channel and depth units")

    err = pred[valid].astype(np.float64) - gt[valid].astype(np.float64)
    mean_bias = float(np.mean(err))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    fraction = 100.0 * count / gt.size
    gt_fraction = 100.0 * np.count_nonzero(valid_gt) / gt.size
    pred_fraction = 100.0 * np.count_nonzero(valid_pred) / pred.size

    report = (
        "TRINOCULAR DEPTH EVALUATION\n"
        "==========================\n"
        f"Ground Truth    : {gt_path}\n"
        f"GT channel      : {gt_channel}\n"
        f"GT depth type   : {args.gt_depth_type}\n"
        f"Prediction      : {pred_path}\n"
        f"Pred channel    : {pred_channel}\n"
        f"Resolution      : {gt.shape[1]} x {gt.shape[0]}\n"
        f"Intrinsics      : fx={fx:.5f}, fy={fy:.5f}, cx={cx:.2f}, cy={cy:.2f}\n"
        f"Intrinsics from : {intr_source}\n"
        f"GT valid        : {gt_fraction:.2f}%\n"
        f"Prediction valid: {pred_fraction:.2f}%\n"
        f"Compared pixels : {count}/{gt.size} ({fraction:.2f}%)\n"
        f"MAE             : {mae:.6f} m = {mae*100:.3f} cm\n"
        f"RMSE            : {rmse:.6f} m = {rmse*100:.3f} cm\n"
        f"Mean Bias: {mean_bias:.6f} m\n"
        f"Mean Bias: {mean_bias * 100:.3f} cm\n"        
    )
    output_path = results / "evaluation_metrics.txt"
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
