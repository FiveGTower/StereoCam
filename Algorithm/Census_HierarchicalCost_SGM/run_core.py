"""Dense Trinocular Stereo runner for Blender cameras L, R, SL.

YAML inputs: center -> L, right -> R, down -> SL.
Outputs: depth/disparity .npy, 32-bit .exr, preview PNG, metadata.
Stereo matching algorithm remains in trinocular_algorithm.py.
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from trinocular_algorithm import trinocular_disparity


CAMERAS = ("L", "R", "SL")
INPUT_KEYS = {"L": "center", "R": "right", "SL": "down"}


def path_from_config(base: Path, value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected nonempty path in config.yaml")
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def section(cfg: dict, key: str) -> dict:
    result = cfg.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"Missing YAML section: {key}")
    return result


def load_calibration(cfg: dict, width: int, height: int) -> dict:
    rig = section(cfg, "rig")
    cameras = section(cfg, "cameras")
    if rig.get("reference_camera") != "L":
        raise ValueError("Set rig.reference_camera: L")
    if rig.get("horizontal_camera", "R") != "R":
        raise ValueError("Set rig.horizontal_camera: R")
    if rig.get("vertical_camera", "SL") != "SL":
        raise ValueError("Set rig.vertical_camera: SL")
    if rig.get("images_already_rectified") is not True:
        raise ValueError("Images must be parallel and already rectified")

    intrinsics = {}
    poses = {}
    for name in CAMERAS:
        cam = cameras.get(name)
        if not isinstance(cam, dict):
            raise ValueError(f"Missing cameras.{name} in YAML")
        if (int(cam["width_px"]), int(cam["height_px"])) != (width, height):
            raise ValueError(f"Camera {name}: image resolution differs from YAML")
        if cam.get("model") != "perspective" or cam.get("distortion_model") != "none":
            raise ValueError(f"Camera {name}: expected perspective without distortion")
        intr = {key: float(cam[key]) for key in ("fx_px", "fy_px", "cx_px", "cy_px")}
        if not all(np.isfinite(x) for x in intr.values()) or intr["fx_px"] <= 0 or intr["fy_px"] <= 0:
            raise ValueError(f"Camera {name}: invalid intrinsics")
        intrinsics[name] = intr
        xyz = cam.get("position_world_xyz_m")
        if isinstance(xyz, (list, tuple)) and len(xyz) == 3 and all(x is not None for x in xyz):
            poses[name] = np.asarray(xyz, dtype=float)
            if not np.all(np.isfinite(poses[name])):
                raise ValueError(f"Camera {name}: invalid world position")

    for name in ("R", "SL"):
        for key in ("fx_px", "fy_px", "cx_px", "cy_px"):
            if abs(intrinsics[name][key] - intrinsics["L"][key]) > 1e-3:
                raise ValueError(f"cameras.{name}.{key} must equal cameras.L.{key}")

    bx = float(rig["baseline_horizontal_m"])
    by = float(rig["baseline_vertical_m"])
    if not all(np.isfinite(x) and x > 0 for x in (bx, by)):
        raise ValueError("Baselines must be positive finite distances in meters")
    if len(poses) == 3:
        tol = float(rig.get("pose_baseline_tolerance_m", 0.0005))
        actual_x = float(np.linalg.norm(poses["R"] - poses["L"]))
        actual_y = float(np.linalg.norm(poses["SL"] - poses["L"]))
        if abs(bx - actual_x) > tol or abs(by - actual_y) > tol:
            raise ValueError(
                f"Baseline mismatch: YAML ({bx:.4f}, {by:.4f}) m; "
                f"camera positions ({actual_x:.4f}, {actual_y:.4f}) m. "
                "Check Blender positions and rerender if they changed."
            )

    sign_x = int(rig.get("horizontal_shift_sign", -1))
    sign_y = int(rig.get("vertical_shift_sign", -1))
    if sign_x not in (-1, 1) or sign_y not in (-1, 1):
        raise ValueError("horizontal_shift_sign and vertical_shift_sign must be -1 or +1")
    return {
        **intrinsics["L"],
        "baseline_horizontal_m": bx,
        "baseline_vertical_m": by,
        "horizontal_shift_sign": sign_x,
        "vertical_shift_sign": sign_y,
    }


def preprocess(img: np.ndarray, cfg: dict) -> np.ndarray:
    equalize = str(cfg["equalize"]).lower()
    if equalize == "clahe":
        clahe = cv2.createCLAHE(
            clipLimit=float(cfg["clahe_clip_limit"]),
            tileGridSize=(int(cfg["clahe_tile_grid"]),) * 2,
        )
        img = clahe.apply(img)
    elif equalize != "none":
        raise ValueError("preprocess.equalize must be none or clahe")
    kernel = int(cfg["gaussian_blur_ksize"])
    if kernel:
        if kernel < 3 or kernel % 2 == 0:
            raise ValueError("gaussian_blur_ksize must be 0 or odd >= 3")
        img = cv2.GaussianBlur(img, (kernel, kernel), 0)
    return img


def write_exr(path: Path, data: np.ndarray) -> None:
    options = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]
    if not cv2.imwrite(str(path), data.astype(np.float32), options):
        raise OSError(f"Cannot write EXR: {path}")
    print(f"Saved EXR: {path}", flush=True)


def write_preview(data: np.ndarray, path: Path, low: float, high: float,
                  invert: bool = False) -> None:
    if high <= low:
        raise ValueError("Preview high must exceed low")
    valid = np.isfinite(data)
    normalized = np.clip((np.where(valid, data, low) - low) / (high - low), 0, 1)
    if invert:
        normalized = 1 - normalized
    image = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[~valid] = 0
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write PNG: {path}")


def run(config_file: Path) -> None:
    config_file = config_file.resolve()
    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict):
        raise ValueError("Invalid YAML root")
    for key in ("experiment", "inputs", "rig", "cameras", "preprocess",
                "search", "trinocular", "output"):
        section(cfg, key)
    base = config_file.parent

    inputs = cfg["inputs"]
    images = {}
    for name in CAMERAS:
        key = INPUT_KEYS[name]
        if key not in inputs:
            raise ValueError(f"Missing inputs.{key} for camera {name}")
        p = path_from_config(base, inputs[key])
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read camera {name} image: {p}")
        images[name] = img
        print(f"Input {name}: {p}", flush=True)
    if images["L"].shape != images["R"].shape or images["L"].shape != images["SL"].shape:
        raise ValueError("L, R and SL images must have identical resolution")
    height, width = images["L"].shape
    calib = load_calibration(cfg, width, height)
    print(f"Resolution {width}x{height}; fx={calib['fx_px']:.2f} px; "
          f"Bx={calib['baseline_horizontal_m']:.3f} m; "
          f"By={calib['baseline_vertical_m']:.3f} m", flush=True)

    pre = cfg["preprocess"]
    scale = float(pre["processing_scale"])
    if not (0.1 <= scale <= 1.0):
        raise ValueError("processing_scale must be between 0.1 and 1.0")
    w2, h2 = max(32, round(width * scale)), max(32, round(height * scale))
    sx, sy = w2 / width, h2 / height
    if abs(sx - sy) > 0.005:
        raise ValueError("Image scaling mismatch in X and Y")
    proc = {}
    for name in CAMERAS:
        img = preprocess(images[name], pre)
        proc[name] = (cv2.resize(img, (w2, h2), interpolation=cv2.INTER_AREA)
                      if (w2, h2) != (width, height) else img)

    search = cfg["search"]
    dmin = math.floor(float(search["min_disparity_px"]) * sx)
    dmax = math.ceil(float(search["max_disparity_px"]) * sx)
    if not (0 <= dmin < dmax < w2):
        raise ValueError("Invalid disparity search interval")
    min_valid = float(search["min_valid_disparity_px"]) * sx
    max_depth = float(search["max_valid_depth_m"])
    if min_valid <= 0 or max_depth <= 0:
        raise ValueError("Invalid minimum disparity or maximum depth")
    ratio_v = (
        calib["fy_px"] * calib["baseline_vertical_m"] * sy
        / (calib["fx_px"] * calib["baseline_horizontal_m"] * sx)
    )
    algorithm_cfg = dict(cfg["trinocular"])
    algorithm_cfg["_min_valid_processed"] = min_valid

    started = time.perf_counter()
    small = trinocular_disparity(
        proc["L"], proc["R"], proc["SL"], dmin, dmax, ratio_v,
        {
            "horizontal_shift_sign": calib["horizontal_shift_sign"],
            "vertical_shift_sign": calib["vertical_shift_sign"],
        }, algorithm_cfg,
    )
    disparity = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    disparity = (disparity / sx).astype(np.float32)
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 0)
    depth[valid] = calib["fx_px"] * calib["baseline_horizontal_m"] / disparity[valid]
    depth[depth > max_depth] = np.nan
    disparity[~np.isfinite(depth)] = np.nan
    elapsed = time.perf_counter() - started

    out = path_from_config(base, cfg["experiment"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    outcfg = cfg["output"]
    if outcfg["save_npy"]:
        np.save(out / "trinocular_disparity_px.npy", disparity)
        np.save(out / "trinocular_depth_m.npy", depth)
    if outcfg.get("save_exr", True):
        write_exr(out / "trinocular_disparity_px.exr", disparity)
        write_exr(out / "trinocular_depth_m.exr", depth)
    if outcfg["save_preview_png"]:
        write_preview(disparity, out / "trinocular_disparity_preview.png",
                      0.0, float(outcfg["preview_max_disparity_px"]))
        write_preview(depth, out / "trinocular_depth_preview.png",
                      float(outcfg["preview_min_depth_m"]),
                      float(outcfg["preview_max_depth_m"]), invert=True)
    if outcfg["save_metadata_json"]:
        (out / "config_snapshot.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        metadata = {
            "algorithm": "dense_trinocular_only",
            "reference_camera": "L",
            "horizontal_camera": "R",
            "vertical_camera": "SL",
            "image_resolution_wh": [width, height],
            "processed_resolution_wh": [w2, h2],
            "camera_intrinsics_and_baselines": calib,
            "vertical_to_horizontal_disparity_ratio": ratio_v,
            "runtime_seconds": round(elapsed, 3),
            "valid_depth_fraction": float(np.isfinite(depth).mean()),
            "disparity_unit": "pixels_in_original_L_image",
            "depth_definition": "axial_Z_from_L_m",
        }
        (out / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE: {out}; valid depth={np.isfinite(depth).mean():.1%}; "
          f"runtime={elapsed:.3f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run L/R/SL dense Trinocular")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("config.yaml"))
    run(parser.parse_args().config)
