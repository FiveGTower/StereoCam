
import os
import cv2
import OpenEXR
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG: sua cac gia tri nay theo scene Blender THUC TE
# ============================================================

ROOT = os.path.dirname(os.path.abspath(__file__))

CENSUS_FILE = "disparity_map.npy"
SGBM_FILE = "disparity_trinocular.npy"
GT_FILE = "groundtruth.exr"

# Camera Blender: 1600 x 1300, lens 2.8 mm, sensor 5 mm
FX_ORIG = 896.0
FY_ORIG = 896.0

BASELINE_M = 0.1  # Phai xac nhan baseline thuc te
MAX_DISP = 96
MAX_DEPTH_M = 100.0

# Blender Z Pass: optical-axis depth
GT_IS_RAY_DISTANCE = False

# Cac khoang cach can danh gia
RANGES = [
    ("< 3 m", 0.0, 3.0),
    ("2.5 - 3.5 m", 2.5, 3.5),
    ("3 - 5 m", 3.0, 5.0),
]


# ============================================================
# 1. LOAD GROUND TRUTH EXR
# ============================================================

def load_exr_depth(path):

    exr = OpenEXR.File(path)
    channels = exr.channels()

    print("EXR channels:", list(channels.keys()))

    keys = [
        k for k in channels
        if k == "Depth.V"
        or k == "Z"
        or k.endswith(".Z")
        or k.endswith(".Depth")
    ]

    if not keys:
        raise ValueError(
            "Khong tim thay kenh Depth/Z trong EXR"
        )

    depth = np.asarray(
        channels[keys[0]].pixels,
        dtype=np.float32
    )

    if depth.ndim != 2:
        raise ValueError(
            f"Depth channel khong phai anh 2D: {depth.shape}"
        )

    print("Using depth channel:", keys[0])

    return depth


# ============================================================
# 2. CALCULATE METRICS
# ============================================================

def metrics_for_mask(disp, mask, gt_depth, gt_disp, fx_b):

    n_gt = int(mask.sum())

    pred_valid = (
        np.isfinite(disp)
        & (disp > 0)
        & (disp < MAX_DISP - 1)
    )

    valid = mask & pred_valid
    n_valid = int(valid.sum())

    coverage = (
        100.0 * n_valid / n_gt
        if n_gt else np.nan
    )

    result = {
        "GT pixels": n_gt,
        "Valid pixels": n_valid,
        "Coverage (%)": coverage,

        "Disp MAE (px)": np.nan,
        "Disp RMSE (px)": np.nan,

        "Bad-1 valid (%)": np.nan,
        "Bad-2 valid (%)": np.nan,
        "Bad-2 all (%)": np.nan,

        "Depth MAE (m)": np.nan,
        "Depth RMSE (m)": np.nan,
    }

    if n_gt == 0:
        return result

    if n_valid == 0:
        result["Bad-2 all (%)"] = 100.0
        return result

    # ------------------------------
    # DISPARITY ERROR
    # ------------------------------

    de = np.abs(
        disp[valid].astype(np.float64)
        - gt_disp[valid].astype(np.float64)
    )

    disp_mae = float(np.mean(de))

    disp_rmse = float(
        np.sqrt(np.mean(de ** 2))
    )

    bad1 = float(
        100.0 * np.mean(de > 1)
    )

    bad2 = float(
        100.0 * np.mean(de > 2)
    )

    # ------------------------------
    # DEPTH ERROR
    # ------------------------------

    pred_depth = (
        fx_b / disp[valid].astype(np.float64)
    )

    ze = (
        pred_depth
        - gt_depth[valid].astype(np.float64)
    )

    depth_mae = float(
        np.mean(np.abs(ze))
    )

    depth_rmse = float(
        np.sqrt(np.mean(ze ** 2))
    )

    # Invalid prediction duoc tinh la bad pixel
    bad2_all = float(
        100.0 * (
            (n_gt - n_valid)
            + np.count_nonzero(de > 2)
        ) / n_gt
    )

    result.update({
        "Disp MAE (px)": disp_mae,
        "Disp RMSE (px)": disp_rmse,

        "Bad-1 valid (%)": bad1,
        "Bad-2 valid (%)": bad2,
        "Bad-2 all (%)": bad2_all,

        "Depth MAE (m)": depth_mae,
        "Depth RMSE (m)": depth_rmse,
    })

    return result


# ============================================================
# 3. SAVE ERROR MAP - GRAYSCALE
# ============================================================

def save_error_map(name, disp, eval_mask, gt_disp):

    pred_valid = (
        np.isfinite(disp)
        & (disp > 0)
        & (disp < MAX_DISP - 1)
    )

    valid = eval_mask & pred_valid

    out = np.zeros(
        disp.shape,
        dtype=np.uint8
    )

    # Sai so 0 px = den
    # Sai so >= 5 px = trang
    out[valid] = np.clip(
        np.abs(disp[valid] - gt_disp[valid])
        / 5.0 * 255.0,
        0, 255
    ).astype(np.uint8)

    # Pixel khong co du doan = xam
    out[eval_mask & ~pred_valid] = 127

    path = os.path.join(
        ROOT,
        f"error_{name}.png"
    )

    cv2.imwrite(path, out)

    print("Saved:", path)


# ============================================================
# 4. PRINT METRICS
# ============================================================

def show_metrics(label, data):

    print(f"\n  {label}")

    for key, value in data.items():

        if isinstance(value, (int, np.integer)):
            print(f"    {key}: {value}")

        else:
            if np.isfinite(value):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: N/A")


# ============================================================
# 5. THEORETICAL DEPTH ERROR
# ============================================================

def plot_theory(fx_b):

    # Khoang cach 2 m den 5 m
    z = np.linspace(2.0, 5.0, 200)

    # Delta Z ~= Z^2 / (f * B) * Delta d
    # Delta d = 1 pixel
    dz = z ** 2 / fx_b

    points = np.array([
        2.0, 2.5, 3.0, 3.5, 4.0, 5.0
    ])

    errors = points ** 2 / fx_b

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        z,
        dz,
        linewidth=2.5,
        label=(
            r"$|\Delta Z| \approx "
            r"\frac{Z^2}{fB}|\Delta d|,"
            r"\ |\Delta d|=1\ \mathrm{px}$"
        )
    )

    ax.scatter(
        points,
        errors,
        s=40
    )

    for x, y in zip(points, errors):
        ax.annotate(
            f"{y:.3f} m",
            (x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9
        )

    ax.set_title(
        "Theoretical depth error for 1 px disparity error\n"
        f"f = {fx_b / BASELINE_M:.1f} px, "
        f"B = {BASELINE_M:g} m"
    )

    ax.set_xlabel(
        "Ground-truth depth Z (m)"
    )

    ax.set_ylabel(
        "Approximate absolute depth error (m)"
    )

    ax.set_xticks(points)
    ax.grid(True, alpha=0.4, linestyle="--")
    ax.legend()

    fig.tight_layout()

    path = os.path.join(
        ROOT,
        "theoretical_depth_error.png"
    )

    fig.savefig(path, dpi=200)
    plt.close(fig)

    print("Saved:", path)


# ============================================================
# 6. MAIN
# ============================================================

def main():

    # ------------------------------
    # LOAD TWO DISPARITY MAPS
    # ------------------------------

    census = np.load(
        os.path.join(ROOT, CENSUS_FILE)
    ).astype(np.float32)

    sgbm = np.load(
        os.path.join(ROOT, SGBM_FILE)
    ).astype(np.float32)

    if census.shape != sgbm.shape:
        raise ValueError(
            f"Disparity shapes differ: "
            f"{census.shape}, {sgbm.shape}"
        )

    H, W = census.shape

    # ------------------------------
    # LOAD GROUND TRUTH
    # ------------------------------

    gt_depth = load_exr_depth(
        os.path.join(ROOT, GT_FILE)
    )

    H0, W0 = gt_depth.shape

    # ------------------------------
    # CAMERA CALIBRATION
    # ------------------------------

    fx = FX_ORIG * W / W0
    fy = FY_ORIG * H / H0

    fx_b = fx * BASELINE_M

    print(
        f"\nDisparity size: {W} x {H}"
        f"\nfx = {fx:.3f} px"
        f"\nfy = {fy:.3f} px"
        f"\nBaseline = {BASELINE_M} m"
    )

    # ------------------------------
    # RESIZE GROUND TRUTH
    # ------------------------------

    gt_depth = cv2.resize(
        gt_depth,
        (W, H),
        interpolation=cv2.INTER_NEAREST
    )

    # Chi chuyen neu GT la ray distance
    if GT_IS_RAY_DISTANCE:

        yy, xx = np.indices(
            (H, W),
            dtype=np.float32
        )

        cx = W / 2.0
        cy = H / 2.0

        gt_depth = gt_depth / np.sqrt(
            1.0
            + ((xx - cx) / fx) ** 2
            + ((yy - cy) / fy) ** 2
        )

    # ------------------------------
    # GROUND TRUTH DISPARITY
    # ------------------------------

    gt_valid = (
        np.isfinite(gt_depth)
        & (gt_depth > 0)
        & (gt_depth < MAX_DEPTH_M)
    )

    gt_disp = np.full(
        gt_depth.shape,
        np.nan,
        dtype=np.float32
    )

    gt_disp[gt_valid] = (
        fx_b / gt_depth[gt_valid]
    )

    # Vung GT hop le cho ca 2 thuat toan
    eval_mask = (
        gt_valid
        & np.isfinite(gt_disp)
        & (gt_disp >= 1)
        & (gt_disp < MAX_DISP - 1)
    )

    if not np.any(eval_mask):
        raise ValueError(
            "Khong co GT pixel hop le. "
            "Kiem tra fx, baseline, do phan giai."
        )

    print(
        "\nGT evaluation pixels:",
        int(eval_mask.sum())
    )

    # ------------------------------
    # EVALUATE EACH ALGORITHM
    # ------------------------------

    maps = {
        "census": census,
        "sgbm": sgbm
    }

    for name, disp in maps.items():

        print("\n" + "=" * 55)
        print(name.upper())

        # Toan bo khoang depth hop le
        show_metrics(
            "FULL GT RANGE",
            metrics_for_mask(
                disp,
                eval_mask,
                gt_depth,
                gt_disp,
                fx_b
            )
        )

        # Danh gia theo 3 khoang cach
        for label, low, high in RANGES:

            mask = (
                eval_mask
                & (gt_depth >= low)
                & (gt_depth < high)
            )

            show_metrics(
                label,
                metrics_for_mask(
                    disp,
                    mask,
                    gt_depth,
                    gt_disp,
                    fx_b
                )
            )

        # Save grayscale error
        save_error_map(
            name,
            disp,
            eval_mask,
            gt_disp
        )

    # ========================================================
    # 7. FAIR COMPARISON: COMMON VALID PIXELS
    # ========================================================

    common_valid = (
        eval_mask

        & np.isfinite(census)
        & (census > 0)
        & (census < MAX_DISP - 1)

        & np.isfinite(sgbm)
        & (sgbm > 0)
        & (sgbm < MAX_DISP - 1)
    )

    print("\n" + "=" * 55)
    print(
        "COMMON VALID PIXELS "
        "(fair MAE/RMSE comparison)"
    )

    print(
        "Common pixels:",
        int(common_valid.sum())
    )

    common_ranges = [
        ("Full", 0.0, MAX_DEPTH_M)
    ] + RANGES

    for label, low, high in common_ranges:

        mask = (
            common_valid
            & (gt_depth >= low)
            & (gt_depth < high)
        )

        print(
            f"\n{label}: "
            f"{int(mask.sum())} common pixels"
        )

        if not np.any(mask):
            continue

        for name, disp in maps.items():

            res = metrics_for_mask(
                disp,
                mask,
                gt_depth,
                gt_disp,
                fx_b
            )

            print(
                f"  {name:7s}: "
                f"Depth MAE = "
                f"{res['Depth MAE (m)']:.4f} m, "
                f"RMSE = "
                f"{res['Depth RMSE (m)']:.4f} m, "
                f"Disp MAE = "
                f"{res['Disp MAE (px)']:.4f} px"
            )

    # ------------------------------
    # DRAW THEORETICAL GRAPH
    # ------------------------------

    plot_theory(fx_b)

    print("\n" + "=" * 55)
    print("EVALUATION FINISHED")

    print(
        "\nBad-2 all counts invalid predictions as bad."
    )

    print(
        "MAE/RMSE include valid predictions only."
    )

    print("\nSaved files:")
    print("  error_census.png")
    print("  error_sgbm.png")
    print("  theoretical_depth_error.png")


if __name__ == "__main__":
    main()
