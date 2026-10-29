
import os
import cv2
import numpy as np
from numba import njit

# ==========================================
# 1. CONFIG
# ==========================================

ROOT = os.path.dirname(os.path.abspath(__file__))

SCALE = 0.5
MAX_DISP = 96

SIGN_H = 1
SIGN_V = 1

P1 = 8.0
P2 = 64.0

UNIQUENESS = 0.05


# ==========================================
# 2. CENSUS TRANSFORM 5x5
# ==========================================

@njit
def census5(im):
    h, w = im.shape
    out = np.zeros((h, w), dtype=np.uint32)

    for y in range(2, h - 2):
        for x in range(2, w - 2):

            bits = np.uint32(0)

            for dy in range(-2, 3):
                for dx in range(-2, 3):

                    if dy == 0 and dx == 0:
                        continue

                    bits = (
                        (bits << np.uint32(1))
                        | np.uint32(
                            im[y + dy, x + dx] < im[y, x]
                        )
                    )

            out[y, x] = bits

    return out


# ==========================================
# 3. HAMMING MATCHING COST
# ==========================================

@njit
def matching_cost(a, b, dmax, axis, sign):
    h, w = a.shape

    cost = np.full(
        (h, w, dmax),
        36.0,
        dtype=np.float32
    )

    for y in range(2, h - 2):
        for x in range(2, w - 2):
            for d in range(dmax):

                if axis == 0:
                    xx = x - sign * d
                    yy = y
                else:
                    xx = x
                    yy = y - sign * d

                if (
                    xx < 2 or xx >= w - 2
                    or yy < 2 or yy >= h - 2
                ):
                    continue

                v = a[y, x] ^ b[yy, xx]

                n = 0
                while v:
                    v &= v - np.uint32(1)
                    n += 1

                cost[y, x, d] = float(n)

    return cost


# ==========================================
# 4. HIERARCHICAL COST
# Full + Half + Quarter
# ==========================================

def hierarchical(a, b, D, axis, sign):
    full = None

    H, W = a.shape

    for s in (1, 2, 4):

        if s == 1:
            aa, bb = a, b
        else:
            aa = cv2.resize(
                a,
                (W // s, H // s),
                interpolation=cv2.INTER_AREA
            )

            bb = cv2.resize(
                b,
                (W // s, H // s),
                interpolation=cv2.INTER_AREA
            )

        ca = census5(aa)
        cb = census5(bb)

        c = matching_cost(
            ca, cb,
            (D + s - 1) // s,
            axis, sign
        )

        if full is None:
            full = c
        else:
            for d in range(D):

                idx = d / s
                lo = int(idx)
                hi = min(
                    lo + 1,
                    c.shape[2] - 1
                )

                t = idx - lo

                plane = (
                    (1 - t) * c[:, :, lo]
                    + t * c[:, :, hi]
                )

                full[:, :, d] += cv2.resize(
                    plane,
                    (W, H),
                    interpolation=cv2.INTER_LINEAR
                )

    return full


# ==========================================
# 5. SGM - 8 DIRECTIONS
# ==========================================

@njit
def sgm8(cost, p1, p2):
    H, W, D = cost.shape

    total = np.zeros(
        (H, W, D),
        dtype=np.float32
    )

    directions = (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1)
    )

    for dx, dy in directions:

        L = np.zeros(
            (H, W, D),
            dtype=np.float32
        )

        if dy >= 0:
            ys = range(H)
        else:
            ys = range(H - 1, -1, -1)

        if dx >= 0:
            xs = range(W)
        else:
            xs = range(W - 1, -1, -1)

        for y in ys:
            for x in xs:

                py = y - dy
                px = x - dx

                if (
                    py < 0 or py >= H
                    or px < 0 or px >= W
                ):
                    for d in range(D):
                        L[y, x, d] = cost[y, x, d]

                else:
                    prev_min = np.float32(1e10)

                    for d in range(D):
                        prev_min = min(
                            prev_min,
                            L[py, px, d]
                        )

                    for d in range(D):

                        v = L[py, px, d]

                        if d > 0:
                            v = min(
                                v,
                                L[py, px, d - 1] + p1
                            )

                        if d + 1 < D:
                            v = min(
                                v,
                                L[py, px, d + 1] + p1
                            )

                        v = min(
                            v,
                            prev_min + p2
                        )

                        L[y, x, d] = (
                            cost[y, x, d]
                            + v
                            - prev_min
                        )

                for d in range(D):
                    total[y, x, d] += L[y, x, d]

    return total


# ==========================================
# 6. MAIN
# ==========================================

def main():

    # --------------------------------------
    # Read 3 images
    # --------------------------------------

    names = (
        "left_up.png",
        "right_up.png",
        "left_down.png"
    )

    images = []

    for name in names:

        path = os.path.join(ROOT, name)

        im = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE
        )

        if im is None:
            raise FileNotFoundError(path)

        im = cv2.resize(
            im,
            None,
            fx=SCALE,
            fy=SCALE,
            interpolation=cv2.INTER_AREA
        )

        images.append(im)

    left, right, down = images

    if not (
        left.shape == right.shape == down.shape
    ):
        raise ValueError(
            "Image sizes must match"
        )

    # --------------------------------------
    # Hierarchical Census Costs
    # --------------------------------------

    print(
        "Computing hierarchical Census costs...",
        flush=True
    )

    Ch = hierarchical(
        left, right,
        MAX_DISP,
        axis=0,
        sign=SIGN_H
    )

    Cv = hierarchical(
        left, down,
        MAX_DISP,
        axis=1,
        sign=SIGN_V
    )

    # --------------------------------------
    # Weighted Trinocular Cost Fusion
    # --------------------------------------

    gx = np.abs(
        cv2.Sobel(
            left, cv2.CV_32F,
            1, 0, ksize=3
        )
    )

    gy = np.abs(
        cv2.Sobel(
            left, cv2.CV_32F,
            0, 1, ksize=3
        )
    )

    wh = (gx + 8) / (gx + gy + 16)

    C = np.ascontiguousarray(
        wh[..., None] * Ch
        + (1 - wh[..., None]) * Cv,
        dtype=np.float32
    )

    del Ch, Cv

    # --------------------------------------
    # SGM 8 Directions
    # --------------------------------------

    print(
        "Running 8-direction SGM...",
        flush=True
    )

    S = sgm8(C, P1, P2)

    # --------------------------------------
    # Winner Takes All
    # --------------------------------------

    disp = np.argmin(
        S,
        axis=2
    ).astype(np.float32)

    # --------------------------------------
    # Uniqueness Check
    # --------------------------------------

    best = np.take_along_axis(
        S,
        disp.astype(np.int32)[..., None],
        axis=2
    )[..., 0]

    for offset in (-1, 0, 1):

        index = np.clip(
            disp.astype(np.int32) + offset,
            0,
            MAX_DISP - 1
        )

        yy, xx = np.indices(
            index.shape
        )

        S[yy, xx, index] = np.inf

    second = np.min(S, axis=2)

    good = (
        (disp > 0)
        & (disp < MAX_DISP - 1)
        & (
            (second - best)
            > UNIQUENESS
            * np.maximum(best, 1.0)
        )
    )

    # Remove image boundaries
    good[:4] = False
    good[-4:] = False
    good[:, :4] = False
    good[:, -4:] = False

    disp[~good] = np.nan

    # --------------------------------------
    # Save raw disparity
    # --------------------------------------

    np.save(
        os.path.join(
            ROOT, "disparity_map.npy"
        ),
        disp
    )

    # ==========================================
    # DISPARITY MAP - GRAYSCALE ONLY
    # ==========================================

    valid = np.isfinite(disp) & (disp > 0)

    gray = np.zeros(disp.shape, dtype=np.uint8)

    if np.any(valid):
        low, high = np.percentile(disp[valid], [2, 98])

        gray[valid] = np.clip(
            255 * (disp[valid] - low) / max(high - low, 1e-6),
            0, 255
        ).astype(np.uint8)

    # Luu anh trang den
    cv2.imwrite(
        os.path.join(ROOT, "disparity_map.png"),
        gray
    )

    # Hien thi bang matplotlib
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    print("Saved: disparity_map.png")


if __name__ == "__main__":
    main()
