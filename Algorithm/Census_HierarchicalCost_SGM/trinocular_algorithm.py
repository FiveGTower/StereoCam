"""Dense trinocular stereo algorithm ONLY: Census, joint matching cost, SGM.

Research implementation inspired by Keller & Scherer (2019), §§ III-IV.
Not a claimed exact reproduction of the authors' unpublished implementation.
No StereoSGBM, wire semantic segmentation or polarimetry.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from numba import njit

INVALID_COST = np.float32(1.0e4)


def census(im: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns packed Census uint32/uint64, validity mask, #bits.

    5x5 yields 24 Census bits, 7x7 yields 48 Census bits.
    Border descriptors are masked invalid (BORDER_REFLECT_101 for construction).
    """
    if size % 2 == 0 or size < 3 or size > 7:
        raise ValueError("census_window must be an odd number between 3 and 7")
    h, w = im.shape
    r = size // 2
    padded = cv2.copyMakeBorder(im, r, r, r, r, cv2.BORDER_REFLECT_101)
    desc = np.zeros((h, w), dtype=np.uint32 if size <= 5 else np.uint64)
    bits = 0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            neighbor = padded[r + dy:r + dy + h, r + dx:r + dx + w]
            bit = (neighbor < im).astype(desc.dtype)
            desc |= (bit << bits)
            bits += 1
    valid = np.ones((h, w), dtype=np.bool_)
    valid[:r, :] = False
    valid[-r:, :] = False
    valid[:, :r] = False
    valid[:, -r:] = False
    return desc, valid, bits


@njit(cache=True)
def _hamming_costs(a, b, valid_a, valid_b, dmin, dmax, hxsign, vysign,
                   ratio_v, w_h, w_v, both_pairs, nbits):
    """Build C(y,x,d) directly in a single Numba kernel to limit allocations.

    R matching: (x+hxsign*d,y); U matching: (x,y+vysign*round(d*ratio_v)).
    Integer vertical positions are an approximation when vertical bases differ.
    """
    h, w = a.shape
    dcount = dmax - dmin + 1
    out = np.full((h, w, dcount), INVALID_COST, dtype=np.float32)
    for di in range(dcount):
        d = dmin + di
        dx = hxsign * d
        dy = vysign * int(np.floor(d * ratio_v + 0.5))
        for y in range(h):
            uy = y + dy
            for x in range(w):
                if not valid_a[y, x]:
                    continue
                rx = x + dx
                rh_ok = 0 <= rx < w and valid_b[0, y, rx]
                uv_ok = 0 <= uy < h and valid_b[1, uy, x]
                if both_pairs and not (rh_ok and uv_ok):
                    continue
                if not rh_ok and not uv_ok:
                    continue
                ch = 0.0
                cv = 0.0
                if rh_ok:
                    z = np.uint64(a[y, x]) ^ np.uint64(b[0, y, rx])
                    # Brian Kernighan popcount, max 48 iterations.
                    n = 0
                    while z != 0:
                        z = z & (z - np.uint64(1))
                        n += 1
                    ch = n / nbits
                if uv_ok:
                    z = np.uint64(a[y, x]) ^ np.uint64(b[1, uy, x])
                    n = 0
                    while z != 0:
                        z = z & (z - np.uint64(1))
                        n += 1
                    cv = n / nbits
                if rh_ok and uv_ok:
                    out[y, x, di] = w_h[y, x] * ch + w_v[y, x] * cv
                elif rh_ok:
                    out[y, x, di] = ch
                else:
                    out[y, x, di] = cv
    return out


def gradient_weights(im: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    h, w = im.shape
    if not cfg["enabled"]:
        return (np.full((h, w), 0.5, np.float32),
                np.full((h, w), 0.5, np.float32))
    floor = float(cfg["floor"])
    if not 0.0 <= floor <= 0.5:
        raise ValueError("gradient_weights.floor must be between 0 and 0.5")
    gx = np.abs(cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3))
    eps = float(cfg["epsilon"])
    gamma = float(cfg["gamma"])
    # Horizontal matching is more informative on vertically oriented edges (|gx|).
    ax = np.power(gx + eps, gamma)
    ay = np.power(gy + eps, gamma)
    p_h = ax / (ax + ay)
    wh = np.asarray(floor + (1.0 - 2.0 * floor) * p_h, np.float32)
    return wh, (1.0 - wh).astype(np.float32)


def build_cost(center: np.ndarray, right: np.ndarray, upper: np.ndarray,
               dmin: int, dmax: int, ratio_vertical: float, geom: dict,
               tc: dict) -> np.ndarray:
    sz = int(tc["census_window"])
    a, ma, bits = census(center, sz)
    b, mb, _ = census(right, sz)
    c, mc, _ = census(upper, sz)
    weights = gradient_weights(center, tc["gradient_weights"])
    return _hamming_costs(a, np.stack((b, c), axis=0), ma,
                          np.stack((mb, mc), axis=0),
                          dmin, dmax,
                          int(geom["horizontal_shift_sign"]),
                          int(geom["vertical_shift_sign"]),
                          ratio_vertical, weights[0], weights[1],
                          bool(tc["require_both_pairs"]), bits)


def hierarchical_joint_cost(C: np.ndarray, R: np.ndarray, U: np.ndarray,
                            dmin: int, dmax: int, ratio_v: float,
                            geom: dict, tc: dict) -> np.ndarray:
    """Coarse->fine accumulation of matching COSTS, not disparity maps.

    This is a practical scale-space approximation of Algorithm 1 of the paper.
    """
    levels = int(tc["hierarchy"]["levels"])
    if not 1 <= levels <= 5:
        raise ValueError("hierarchy.levels must be 1..5")
    coarse_weight = float(tc["hierarchy"]["coarse_cost_weight"])
    if coarse_weight < 0:
        raise ValueError("coarse_cost_weight must be >=0")
    full_h, full_w = C.shape
    pyramid = []
    for lev in range(levels):
        factor = 2 ** lev
        w = max(16, round(full_w / factor))
        h = max(16, round(full_h / factor))
        images = tuple(cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
                       if factor > 1 else im for im in (C, R, U))
        scale_x = w / full_w
        ldmin = max(0, int(math.floor(dmin * scale_x)))
        ldmax = max(ldmin + 1, int(math.ceil(dmax * scale_x)))
        pyramid.append((images, ldmin, ldmax))

    previous = None
    previous_min = 0
    previous_w = 0
    for lev in range(levels - 1, -1, -1):
        (imc, imr, imu), dm, dM = pyramid[lev]
        print(f"  Pyramid L{lev}: {imc.shape[1]}x{imc.shape[0]} "
              f"disparity [{dm}, {dM}], Census and joint costs", flush=True)
        cost = build_cost(imc, imr, imu, dm, dM, ratio_v, geom, tc)
        if previous is not None and coarse_weight > 0:
            h, w, nd = cost.shape
            up = cv2.resize(previous, (w, h), interpolation=cv2.INTER_LINEAR)
            if up.ndim == 2:
                up = up[:, :, None]
            factor = previous_w / w
            for idx in range(nd):
                coarse_d = (dm + idx) * factor - previous_min
                lo = int(np.floor(coarse_d))
                hi = lo + 1
                frac = coarse_d - lo
                if lo < 0 or hi >= up.shape[2]:
                    # Out-of-bounds for highest disparity: nearest valid sample.
                    lo = max(0, min(up.shape[2] - 1, int(round(coarse_d))))
                    hi = lo
                    frac = 0.0
                lower = up[:, :, lo]
                upper = up[:, :, hi]
                inherited = (1.0 - frac) * lower + frac * upper
                valid = (cost[:, :, idx] < INVALID_COST / 2) & (
                    lower < INVALID_COST / 2) & (upper < INVALID_COST / 2)
                cost[:, :, idx] += np.where(valid, coarse_weight * inherited, 0.0)
            del up
        previous = cost
        previous_min = dm
        previous_w = imc.shape[1]
    return previous


@njit(cache=True)
def _aggregate_one_direction(cost, aggregated, dy, dx, P1, P2):
    h, w, dn = cost.shape
    for start_y in range(h):
        for start_x in range(w):
            # Path start is a pixel whose predecessor is off the image.
            pred_y = start_y - dy
            pred_x = start_x - dx
            if 0 <= pred_y < h and 0 <= pred_x < w:
                continue
            prev = np.empty(dn, dtype=np.float32)
            curr = np.empty(dn, dtype=np.float32)
            for di in range(dn):
                prev[di] = cost[start_y, start_x, di]
                aggregated[start_y, start_x, di] += prev[di]
            y = start_y + dy
            x = start_x + dx
            while 0 <= y < h and 0 <= x < w:
                min_prev = np.min(prev)
                for di in range(dn):
                    a = prev[di]
                    b = prev[di - 1] + P1 if di > 0 else np.float32(1e9)
                    c = prev[di + 1] + P1 if di < dn - 1 else np.float32(1e9)
                    d = min_prev + P2
                    t = min(a, b, c, d)
                    v = cost[y, x, di] + t - min_prev
                    curr[di] = v
                    aggregated[y, x, di] += v
                prev, curr = curr, prev
                y += dy
                x += dx


def sgm_aggregate(cost: np.ndarray, cfg: dict) -> np.ndarray:
    paths = int(cfg["paths"])
    if paths not in (4, 8):
        raise ValueError("sgm.paths must be 4 or 8")
    p1, p2 = float(cfg["P1"]), float(cfg["P2"])
    if not 0 < p1 < p2:
        raise ValueError("Expected 0 < trinocular.sgm.P1 < P2")
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    if paths == 8:
        directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    agg = np.zeros_like(cost, dtype=np.float32)
    for i, (dy, dx) in enumerate(directions, 1):
        print(f"  SGM direction {i}/{paths}: ({dy},{dx})", flush=True)
        _aggregate_one_direction(cost, agg, dy, dx, p1, p2)
    return agg


@njit(cache=True)
def _pick_disparity(agg, cost, dmin, uniqueness_ratio, subpixel,
                    min_valid):
    h, w, dn = agg.shape
    disp = np.full((h, w), np.nan, np.float32)
    for y in range(h):
        for x in range(w):
            best = np.float32(1e30)
            k = -1
            for di in range(dn):
                if agg[y, x, di] < best:
                    best = agg[y, x, di]
                    k = di
            if k < 0 or cost[y, x, k] >= INVALID_COST / 2:
                continue
            d = np.float32(dmin + k)
            if d < min_valid:
                continue
            if uniqueness_ratio > 0:
                second = np.float32(1e30)
                for di in range(dn):
                    if abs(di - k) <= 1:
                        continue
                    if agg[y, x, di] < second:
                        second = agg[y, x, di]
                if second < 1e29:
                    relative_gap = (second - best) / max(abs(best), 1e-6)
                    if relative_gap < uniqueness_ratio:
                        continue
            if subpixel and 0 < k < dn - 1:
                l = agg[y, x, k - 1]
                m = agg[y, x, k]
                r = agg[y, x, k + 1]
                den = l - 2 * m + r
                if den > 1e-6:
                    off = 0.5 * (l - r) / den
                    if -1 <= off <= 1:
                        d += off
            if d >= min_valid:
                disp[y, x] = d
    return disp


def median_valid_filter(disp: np.ndarray, ksize: int) -> np.ndarray:
    if not ksize:
        return disp
    if ksize % 2 == 0 or ksize < 3:
        raise ValueError("median_ksize must be 0 or odd >=3")
    finite = np.isfinite(disp)
    # Filter only valid pixels; never hallucinate into missing regions.
    src = np.where(finite, disp, 0).astype(np.float32)
    out = cv2.medianBlur(src, ksize)
    return np.where(finite, out, np.nan).astype(np.float32)


def trinocular_disparity(C, R, U, dmin, dmax, ratio_v, camcfg, cfg):
    if not 0 <= dmin < dmax:
        raise ValueError("Require 0 <= min disparity < max disparity")
    if ratio_v <= 0:
        raise ValueError("Vertical baseline/focal ratio must be positive")
    print("Trinocular: Census -> hierarchical joint cost -> SGM -> disparity")
    cost = hierarchical_joint_cost(C, R, U, dmin, dmax, ratio_v, camcfg, cfg)
    agg = sgm_aggregate(cost, cfg["sgm"])
    print("  Selecting dense disparity (uniqueness/subpixel)", flush=True)
    result = _pick_disparity(agg, cost, dmin,
                             float(cfg["selection"]["uniqueness_ratio"]),
                             bool(cfg["selection"]["subpixel"]),
                             max(0.001, float(cfg["_min_valid_processed"])))
    del cost, agg
    result = median_valid_filter(result,
                                int(cfg["postprocess"]["median_ksize"]))
    return result


