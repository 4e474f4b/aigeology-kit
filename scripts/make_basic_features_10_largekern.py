#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_basic_features_10_largekern.py

単一DEMから「比高・標準偏差・傾斜・曲率・開度・TRI」など
地形分類用の基本特徴量を、マルチスケールで一括生成するスクリプト。

make_basic_features_10_scaleaware.py との違い
----------------------------------------------
傾斜・斜面方位・平均曲率・ラプラシアンを「スムージング版（平滑化DEM→3×3）」
ではなく「大窓版（10mDEMのまま d ピクセル離れた中心差分）」で計算する。

  比高 R=30m → k=3 → d=1 → 隣接1ピクセル（10m）差分
  比高 R=90m → k=9 → d=4 → 4ピクセル（40m）離れた差分
  比高 R=300m → k=31 → d=15 → 15ピクセル（150m）離れた差分

これにより比高・標準偏差・TPI・TRIと同じ考え方
（10mDEMをそのまま使い、窓サイズでスケールを決める）で
全10特徴量をマルチスケール出力できる。

"""

import os
import sys
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import (
    uniform_filter,
    maximum_filter,
    minimum_filter,
)


# =============== ユーティリティ ===============

def ask_path(prompt, must_exist=True, default=None):
    while True:
        s = input(f"{prompt}{' [' + default + ']' if default else ''}: ").strip()
        if not s and default:
            s = default
        if not s:
            print("  入力が空です。")
            continue
        p = Path(s.strip('"'))
        try:
            p = p.resolve(strict=False)
        except Exception:
            pass
        if must_exist and not p.exists():
            print(f"  見つかりません: {p}")
            continue
        return str(p)


def ask_float(prompt, default):
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if not s:
            return float(default)
        try:
            return float(s)
        except ValueError:
            print("  数値で入力してください。")


def _r_to_k_list(relief_r_list, px):
    """R[m] -> k（窓サイズ, 奇数）に変換する。"""
    k_list = []
    for R in relief_r_list:
        if R is None:
            continue
        if R <= 0:
            raise ValueError(f"R must be > 0: {R}")
        k = int(round(R / px))
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1
        k_list.append(k)
    return k_list


def ask_int(prompt, default):
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if not s:
            return int(default)
        try:
            return int(s)
        except ValueError:
            print("  整数で入力してください。")


def ask_float_list(prompt, default_list):
    default_str = ",".join(str(v) for v in default_list)
    while True:
        s = input(f"{prompt} [{default_str}]: ").strip()
        if not s:
            return list(default_list)
        try:
            s2 = s.strip().strip("[]")
            vals = [float(x.strip()) for x in s2.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except ValueError:
            print("  カンマ区切りの数値で入力してください（例: 10,20,30）。")


def _winpix_from_meters(r_m: float, px: float, *, min_win: int = 1, odd: bool = True) -> int:
    if not np.isfinite(r_m) or r_m <= 0:
        k = min_win
    else:
        k = int(round(float(r_m) / float(px)))
        k = max(min_win, k)
    if odd and (k % 2 == 0):
        k += 1
    return k


def _apply_valid_mask(arr: np.ndarray, win_pix: int):
    """外縁 floor(k/2) ピクセルを NaN にする。"""
    h, w = arr.shape
    k = int(max(1, win_pix))
    r = k // 2
    mask = np.ones_like(arr, dtype=bool)
    mask[r:h-r, r:w-r] = False
    out = arr.copy()
    out[mask] = np.nan
    return out


# =============== 比高 / 標準偏差 ===============

def local_relief(arr, win_pix: int):
    """局所比高：窓内の max - min"""
    valid = np.isfinite(arr)
    a_maxin = np.where(valid, arr, -np.inf)
    k = int(max(1, win_pix))
    loc_max = maximum_filter(a_maxin, size=k, mode="nearest")
    a_minin = np.where(valid, arr, +np.inf)
    loc_min = minimum_filter(a_minin, size=k, mode="nearest")
    out = (loc_max - loc_min).astype(np.float32)
    out[~valid] = np.nan
    return out


def local_stddev(arr, win_pix: int):
    """局所標準偏差（NaN 対応）"""
    k = int(max(1, win_pix))
    valid = np.isfinite(arr).astype(np.float32)
    a = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    n = uniform_filter(valid, size=k, mode="nearest")
    sum1 = uniform_filter(a, size=k, mode="nearest") * n
    sum2 = uniform_filter(a * a, size=k, mode="nearest") * n
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sum1 / n
        mean_sq = sum2 / n
    var = mean_sq - mean * mean
    var[var < 0] = 0.0
    std = np.sqrt(var).astype(np.float32)
    std[n == 0] = np.nan
    return std


def local_mean_nan(arr, win_pix: int):
    """局所平均（NaN 対応）"""
    k = int(max(1, win_pix))
    valid = np.isfinite(arr).astype(np.float32)
    a = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    n = uniform_filter(valid, size=k, mode="nearest")
    s = uniform_filter(a, size=k, mode="nearest")
    out = np.full_like(a, np.nan, dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[n > 0] = (s[n > 0] / n[n > 0]).astype(np.float32)
    out[~np.isfinite(arr)] = np.nan
    return out


def local_tpi(arr, win_pix: int):
    """TPI = z - mean(neighborhood)（NaN 対応）"""
    m = local_mean_nan(arr, win_pix)
    out = (arr.astype(np.float32) - m).astype(np.float32)
    out[~np.isfinite(arr)] = np.nan
    return out


def _shift_edge(arr, dy: int, dx: int):
    """端を edge で埋めるシフト。result[i,j] = arr[i-dy, j-dx]"""
    h, w = arr.shape
    py, px = abs(dy), abs(dx)
    a = np.pad(arr, ((py, py), (px, px)), mode="edge")
    y0 = py - dy
    x0 = px - dx
    return a[y0:y0+h, x0:x0+w]


def local_tri(arr, win_pix: int):
    """TRI（8近傍差分型, d=(k-1)//2 を採用）"""
    k = int(max(1, win_pix))
    d = max(1, (k - 1) // 2)
    z = arr.astype(np.float32)
    c_valid = np.isfinite(z)
    shifts = [(-d,0),(d,0),(0,-d),(0,d),(-d,-d),(-d,d),(d,-d),(d,d)]
    sumdiff = np.zeros_like(z, dtype=np.float32)
    cnt = np.zeros_like(z, dtype=np.float32)
    for dy, dx in shifts:
        nb = _shift_edge(z, dy, dx)
        v = c_valid & np.isfinite(nb)
        sumdiff[v] += np.abs(z[v] - nb[v])
        cnt[v] += 1.0
    out = np.full_like(z, np.nan, dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[cnt > 0] = (sumdiff[cnt > 0] / cnt[cnt > 0]).astype(np.float32)
    out[~c_valid] = np.nan
    return out


# =============== 大窓版 勾配 / 方位 / ラプラシアン / 平均曲率 ===============

def _gradients_d(arr: np.ndarray, px: float, d: int = 1):
    """
    間隔 d ピクセルの中心差分で勾配ベクトル (gx, gy) を計算。
    d=1 → 3×3 相当, d=4 → 9×9 相当（中心と端点のみ使用）。
    """
    mask = ~np.isfinite(arr)
    z = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    step = d * px

    east  = _shift_edge(z, 0, -d)   # z[i, j+d]
    west  = _shift_edge(z, 0, +d)   # z[i, j-d]
    north = _shift_edge(z, +d, 0)   # z[i-d, j]（行インデックス小=北）
    south = _shift_edge(z, -d, 0)   # z[i+d, j]

    gx = (east  - west)  / (2.0 * step)
    gy = (north - south) / (2.0 * step)

    gx[mask] = np.nan
    gy[mask] = np.nan
    return gx, gy


def slope_deg_d(arr: np.ndarray, px: float, d: int = 1) -> np.ndarray:
    """勾配角 [deg]（間隔 d ピクセルの中心差分）"""
    gx, gy = _gradients_d(arr, px, d)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    slope[~np.isfinite(arr)] = np.nan
    return slope.astype(np.float32)


def aspect_deg_d(arr: np.ndarray, px: float, d: int = 1) -> np.ndarray:
    """斜面方位 [deg]（0=東, 90=北、間隔 d ピクセルの中心差分）"""
    gx, gy = _gradients_d(arr, px, d)
    asp = np.degrees(np.arctan2(gy, gx))
    asp = np.mod(90.0 - asp, 360.0)
    asp[~np.isfinite(arr)] = np.nan
    return asp.astype(np.float32)


def laplacian_d(arr: np.ndarray, px: float, d: int = 1) -> np.ndarray:
    """ラプラシアン（間隔 d ピクセルの中心差分）"""
    mask = ~np.isfinite(arr)
    z = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    step = d * px

    east  = _shift_edge(z, 0, -d)
    west  = _shift_edge(z, 0, +d)
    north = _shift_edge(z, +d, 0)
    south = _shift_edge(z, -d, 0)

    out = (east + west + north + south - 4.0 * z) / (step * step)
    out[mask] = np.nan
    return out.astype(np.float32)


def mean_curvature_d(arr: np.ndarray, px: float, d: int = 1) -> np.ndarray:
    """平均曲率（間隔 d ピクセルの中心差分）"""
    mask = ~np.isfinite(arr)
    z = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    step = d * px

    east  = _shift_edge(z, 0, -d)
    west  = _shift_edge(z, 0, +d)
    north = _shift_edge(z, +d, 0)
    south = _shift_edge(z, -d, 0)
    ne    = _shift_edge(z, +d, -d)
    nw    = _shift_edge(z, +d, +d)
    se    = _shift_edge(z, -d, -d)
    sw    = _shift_edge(z, -d, +d)

    gx  = (east  - west)  / (2.0 * step)
    gy  = (north - south) / (2.0 * step)
    zxx = (east  - 2.0 * z + west)  / (step * step)
    zyy = (north - 2.0 * z + south) / (step * step)
    zxy = (ne - nw - se + sw)       / (4.0 * step * step)

    zx2 = gx * gx
    zy2 = gy * gy
    denom = 2.0 * np.power(1.0 + zx2 + zy2, 1.5)
    num   = (1.0 + zy2) * zxx - 2.0 * gx * gy * zxy + (1.0 + zx2) * zyy

    with np.errstate(invalid="ignore", divide="ignore"):
        H = num / denom
    H[mask] = np.nan
    return H.astype(np.float32)


# =============== 開度（Python 内蔵版） ===============

def _unit_dirs(n_dirs: int):
    """0〜πの半円方向に n_dirs 本の単位ベクトル"""
    thetas = np.linspace(0.0, math.pi, n_dirs, endpoint=False)
    return np.cos(thetas), np.sin(thetas)


def openness_pair(dem, px: float, r_open: float, n_dirs: int = 8, step_stride: int = 1):
    """正開度 / 負開度 を Python のみで計算する簡易実装。"""
    h, w = dem.shape
    cx = np.arange(w)
    cy = np.arange(h)
    X, Y = np.meshgrid(cx, cy)

    cos_t, sin_t = _unit_dirs(n_dirs)
    pos_list = []
    neg_list = []
    max_step = max(1, int(r_open / px))
    valid = np.isfinite(dem)

    for k in range(n_dirs):
        dx = cos_t[k]
        dy = sin_t[k]
        alphas = []
        betas  = []

        for step in range(1, max_step + 1, step_stride):
            x_f = X + dx * step
            y_f = Y - dy * step
            x0 = np.floor(x_f).astype(int)
            y0 = np.floor(y_f).astype(int)
            x1 = x0 + 1
            y1 = y0 + 1
            inside = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
            if not inside.any():
                continue
            wx = x_f - x0
            wy = y_f - y0
            x0c = np.clip(x0, 0, w - 1)
            x1c = np.clip(x1, 0, w - 1)
            y0c = np.clip(y0, 0, h - 1)
            y1c = np.clip(y1, 0, h - 1)
            z00 = dem[y0c, x0c]
            z10 = dem[y0c, x1c]
            z01 = dem[y1c, x0c]
            z11 = dem[y1c, x1c]
            z_top = (z00*(1-wx)*(1-wy) + z10*wx*(1-wy) +
                     z01*(1-wx)*wy    + z11*wx*wy)
            dist = step * px
            with np.errstate(invalid="ignore"):
                dz    = z_top - dem
                alpha = np.degrees(np.arctan2(dz, dist))
                beta  = np.degrees(np.arctan2(-dz, dist))
            alpha[~inside] = np.nan
            beta[~inside]  = np.nan
            alphas.append(alpha)
            betas.append(beta)

        if not alphas:
            continue
        alpha_stack = np.stack(alphas, axis=0)
        beta_stack  = np.stack(betas,  axis=0)
        with np.errstate(invalid="ignore"):
            max_alpha = np.nanmax(alpha_stack, axis=0)
            max_beta  = np.nanmax(beta_stack,  axis=0)
        pos_list.append(90.0 - max_alpha)
        neg_list.append(90.0 - max_beta)

    if pos_list:
        pos = np.nanmean(np.stack(pos_list, axis=0), axis=0).astype(np.float32)
        neg = np.nanmean(np.stack(neg_list, axis=0), axis=0).astype(np.float32)
    else:
        pos = np.full_like(dem, np.nan, dtype=np.float32)
        neg = np.full_like(dem, np.nan, dtype=np.float32)
    pos[~valid] = np.nan
    neg[~valid] = np.nan
    return pos, neg


# =============== SAGA 開度 ===============

def confirm_saga_openness_tool(saga_cmd_path: str, tool_id: int = 5) -> bool:
    try:
        result = subprocess.run(
            [saga_cmd_path, "ta_lighting", "--help"],
            capture_output=True, text=True, timeout=15,
        )
        output = result.stdout + result.stderr
        if f"{tool_id}" in output or "Topographic Openness" in output:
            return True
        print(f"\n=== SAGA ta_lighting モジュールの一覧を確認します... ===")
        result2 = subprocess.run(
            [saga_cmd_path, "ta_lighting"],
            capture_output=True, text=True, timeout=15,
        )
        print(result2.stdout[:2000])
        print(result2.stderr[:500])
        return True
    except Exception as e:
        print(f"  SAGA 確認中にエラー: {e}")
        return False


def compute_openness_with_saga(
    saga_cmd_path: str,
    dem_path: Path,
    out_dir: Path,
    stem: str,
    openness_r_list,
    n_dirs: int,
    nodata,
    meta: dict,
    px: float,
    global_mask_win: int,
):
    """SAGA ta_lighting 5 で正開度 / 負開度を計算する。"""
    for r_open in openness_r_list:
        rtag = int(round(r_open))
        tag_open = f"r{rtag}m_nd{n_dirs}dir"
        pos_tif = out_dir / f"{stem}_openness_pos_{tag_open}.tif"
        neg_tif = out_dir / f"{stem}_openness_neg_{tag_open}.tif"

        tmp_pos = out_dir / f"_tmp_pos_{rtag}.sdat"
        tmp_neg = out_dir / f"_tmp_neg_{rtag}.sdat"

        radius_px = max(1, int(round(r_open / px)))

        cmd = [
            saga_cmd_path, "ta_lighting", "5",
            "-DEM",        str(dem_path),
            "-POS",        str(tmp_pos),
            "-NEG",        str(tmp_neg),
            "-RADIUS",     str(radius_px),
            "-DIRECTIONS", str(n_dirs),
            "-METHOD",     "0",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"  [WARN] SAGA 終了コード={result.returncode}")
                print(result.stderr[:500])

            for tmp, out_tif, label in [
                (tmp_pos, pos_tif, "openness_pos"),
                (tmp_neg, neg_tif, "openness_neg"),
            ]:
                sdat = tmp.with_suffix(".sdat")
                if sdat.exists():
                    with rasterio.open(sdat) as src:
                        data = src.read(1).astype(np.float32)
                    data[~np.isfinite(data)] = np.nan
                    data = np.degrees(data)
                    data = _apply_valid_mask(data, global_mask_win)
                    arr = np.where(np.isfinite(data), data, nodata).astype(np.float32)
                    with rasterio.open(out_tif, "w", **meta) as dst:
                        dst.write(arr, 1)
                    print(f"    [{label}] → {out_tif.name}")
                    for ext in (".sdat", ".sgrd", ".prj", ".mgrd"):
                        p = tmp.with_suffix(ext)
                        if p.exists():
                            p.unlink()
                else:
                    print(f"  [WARN] SAGA 出力 {sdat} が見つかりません。")
        except Exception as e:
            print(f"  [ERROR] SAGA openness 失敗: {e}")


# =============== メイン処理 ===============

def main():
    print("=== DEM → 10特徴量 一括出力（大窓マルチスケール・全特徴量同一スケール） ===")

    dem_path_str = ask_path("入力DEM GeoTIFF のパス", must_exist=True)
    dem_path = Path(dem_path_str)
    stem = dem_path.stem

    default_out_dir = str(dem_path.with_suffix("").parent / (stem + "_features"))
    out_dir_str = ask_path(f"出力フォルダ [{default_out_dir}]", must_exist=False, default=default_out_dir)
    out_dir = Path(out_dir_str)

    try:
        is_file = out_dir.exists() and out_dir.is_file()
    except Exception:
        is_file = False
    if is_file or out_dir.suffix.lower() in (".tif", ".tiff", ".vrt"):
        print(f"[WARN] 出力フォルダにファイルが指定されています: {out_dir}")
        print(f"       → {default_out_dir} に補正します。")
        out_dir = Path(default_out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is None:
            nodata = -9999.0
        dem = np.where(np.isfinite(dem), dem, np.nan).astype(np.float32)
        transform: Affine = src.transform
        px = transform.a
        meta = src.meta.copy()
        meta.update(
            dtype="float32",
            nodata=nodata,
            compress="lzw",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            count=1,
        )

    h, w = dem.shape
    print(f"\n---情報--- DEM サイズ: {w} × {h} ピクセル, ピクセルサイズ: {px:.3f} m")

    # SAGA 判定
    saga_cmd_path = shutil.which("saga_cmd")
    has_saga = saga_cmd_path is not None

    # 開度の計算方法
    print("\n=== 開度の計算方法 ===")
    if has_saga:
        print(f"  saga_cmd が見つかりました: {saga_cmd_path}")
        print("  1: SAGA (ta_lighting 5) で開度を計算する（大規模DEM向け推奨）")
        print("  2: Python 内蔵の開度計算を使う（小〜中規模DEM向け）")
        print("  3: 開度は計算しない")
        openness_mode = ask_int("番号を選んでください", 1)
        if openness_mode == 1:
            ok = confirm_saga_openness_tool(saga_cmd_path, tool_id=5)
            if not ok:
                print("  → SAGA 開度は使用せず、Python 内蔵開度モードに切り替えます。")
                openness_mode = 2
    else:
        print("  saga_cmd が見つかりませんでした。")
        print("  1: Python 内蔵の開度計算を使う（小〜中規模DEM向け）")
        print("  2: 開度は計算しない")
        openness_mode_raw = ask_int("番号を選んでください", 1)
        openness_mode = 2 if openness_mode_raw == 1 else 3

    # スケール指定
    print("\n[スケール指定（全10特徴量共通）]\n")
    print("空間スケール R[m] を指定してください（カンマ区切り可）。")
    print("  例) px=10m のとき R=30 → k=3(d=1), R=90 → k=9(d=4), R=300 → k=31(d=15)")
    print("  比高・標準偏差・TPI・TRI・傾斜・方位・曲率・ラプラシアンすべて同一スケールで出力します。\n")

    relief_r_list = ask_float_list("R[m]（例: 30,50,90）", [30.0, 50.0, 90.0])
    relief_r_in_list = list(relief_r_list)
    k_list = _r_to_k_list(relief_r_in_list, px)
    adopted_r_list = [float(k) * float(px) for k in k_list]

    print("\n  → 入力Rに対して採用される実効R（=k×px）と差分距離 d:")
    for Rin, k, Radopt in zip(relief_r_in_list, k_list, adopted_r_list):
        d = max(1, (k - 1) // 2)
        print(f"    R_in={Rin:.3f} m → k={k}（{k}×{k}窓） → R_adopt={Radopt:.3f} m, d={d}px（{d*px:.1f}m）")

    # 開度スケール（比高と同一スケールで自動設定）
    print("  1: 外接円（正方形窓の外接円） R_open = √2 × R_eff")
    print("  2: 面積等価円（k×k窓と同面積） R_open = (2/√π) × R_eff")
    print("  3: 内接円（正方形窓の内接円） R_open = R_eff")
    match_mode = ask_int("番号を選んでください", 2)

    if match_mode == 2:
        coeff = 2.0 / math.sqrt(math.pi)
        mode_label = "面積等価円 (2/√π × R_eff)"
    elif match_mode == 3:
        coeff = 1.0
        mode_label = "内接円 (= R_eff)"
    else:
        coeff = math.sqrt(2.0)
        mode_label = "外接円 (√2 × R_eff)"

    print(f"  → 採用方式: {mode_label}")

    win_pix_list   = []
    r_eff_list     = []
    r_open_match_list = []
    for Rin, k in zip(relief_r_in_list, k_list):
        win_pix_list.append(k)
        R_eff = ((k - 1) / 2.0) * px
        r_eff_list.append(R_eff)
        R_open_match = coeff * R_eff
        r_open_match_list.append(R_open_match)
        print(f"  R_in={Rin:.3f}m → k={k}, R_eff≒{R_eff:.3f}m, R_open_match≒{R_open_match:.3f}m")

    mask_win_candidates = list(win_pix_list)

    # 開度パラメータ
    openness_r_list = []
    n_dirs = None
    step_stride = 1

    if openness_mode != 3:
        default_open_list = [float(f"{v:.3f}") for v in r_open_match_list]
        info_pairs = ", ".join(
            f"R={R:.3g}m→R_open_match≒{Ropen:.3g}m"
            for R, Ropen in zip(relief_r_list, r_open_match_list)
        )
        openness_r_list = ask_float_list(
            "開度の最大距離 R_open[m]（カンマ区切り可, Enterで比高と同スケール推奨値）"
            f"\n  スケール対応の目安: {info_pairs}",
            default_open_list,
        )
        openness_r_list = [float(x) for x in openness_r_list if float(x) > 0.0]
        _seen = set()
        _tmp = []
        for x in openness_r_list:
            key = round(x, 6)
            if key in _seen:
                continue
            _seen.add(key)
            _tmp.append(x)
        openness_r_list = _tmp

        n_dirs = ask_int("開度の方向数 n_dirs（8 or 16 推奨）", 8)

        if openness_mode == 2:
            step_stride = ask_int(
                "開度のサンプリング間隔（ピクセル, Python 内蔵のみ）"
                "\n  1: 1,2,3,...ピクセルごと / 2: 2,4,6,... / 3: 3,6,9,...",
                1,
            )
            for r_open in openness_r_list:
                n_px = max(1, int(round(float(r_open) / float(px))))
                print(f"    - R_open={r_open}m → 1〜{n_px}px（step_stride={step_stride}）")
        elif openness_mode == 1:
            for r_open in openness_r_list:
                n_px = max(1, int(round(float(r_open) / float(px))))
                print(f"    - R_open={r_open}m → 1〜{n_px}px（n_dirs={n_dirs}）")

        for r_open in openness_r_list:
            win_pix_open = _winpix_from_meters(r_open, px, min_win=1, odd=True)
            mask_win_candidates.append(win_pix_open)

    if openness_mode == 2:
        total_px = h * w
        if total_px > 50_000_000:
            print(
                f"\n---注意--- DEM の総ピクセル数は約 {total_px/1e6:.1f} 百万ピクセルです。"
                "\n       Python 内蔵の開度計算は非常に時間がかかる可能性があります。"
            )

    # 外縁マスク窓（全指標で共通）
    if mask_win_candidates:
        global_mask_win = max(int(k) for k in mask_win_candidates)
    else:
        global_mask_win = 3
    print(f"\n---情報--- 外縁マスクは共通窓 k={global_mask_win} を使用します。")

    def write_feature(name: str, data: np.ndarray):
        arr = data.astype(np.float32)
        arr = np.where(np.isfinite(arr), arr, nodata).astype(np.float32)
        out_path = out_dir / name
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(arr, 1)

    # ─────────────────────────────────────────────────────────────
    # マルチスケールループ：全 10 特徴量を同一スケールで計算
    # ─────────────────────────────────────────────────────────────
    for Rin, win_pix in zip(relief_r_in_list, k_list):
        R_eff = ((win_pix - 1) / 2.0) * px
        d = max(1, (win_pix - 1) // 2)
        tag = f"rin{int(round(Rin))}m_k{win_pix}_reff{int(round(R_eff))}m"

        print(f"\n  > スケール R={Rin:.3g}m, k={win_pix}, d={d}px ...")

        # 比高・標準偏差
        rel = local_relief(dem, win_pix)
        write_feature(f"{stem}_relief_{tag}.tif", _apply_valid_mask(rel, global_mask_win))

        std = local_stddev(dem, win_pix)
        write_feature(f"{stem}_stddev_{tag}.tif", _apply_valid_mask(std, global_mask_win))

        # TPI・TRI
        tpi = local_tpi(dem, win_pix)
        write_feature(f"{stem}_tpi_{tag}.tif", _apply_valid_mask(tpi, global_mask_win))

        tri = local_tri(dem, win_pix)
        write_feature(f"{stem}_tri_{tag}.tif", _apply_valid_mask(tri, global_mask_win))

        # 傾斜・方位・ラプラシアン・平均曲率（大窓中心差分）
        slope = slope_deg_d(dem, px, d)
        write_feature(f"{stem}_slope_deg_{tag}.tif", _apply_valid_mask(slope, global_mask_win))

        aspect = aspect_deg_d(dem, px, d)
        write_feature(f"{stem}_aspect_deg_{tag}.tif", _apply_valid_mask(aspect, global_mask_win))

        lap = laplacian_d(dem, px, d)
        write_feature(f"{stem}_laplacian_{tag}.tif", _apply_valid_mask(lap, global_mask_win))

        meancurv = mean_curvature_d(dem, px, d)
        write_feature(f"{stem}_mean_curvature_{tag}.tif", _apply_valid_mask(meancurv, global_mask_win))

        print(f"     比高・標準偏差・TPI・TRI・傾斜・方位・ラプラシアン・平均曲率 → 完了")

    # 開度（マルチスケール）
    if openness_mode == 3:
        print("\n  > 開度の計算はスキップされました。")
    elif openness_mode == 1:
        compute_openness_with_saga(
            saga_cmd_path, dem_path, out_dir, stem,
            openness_r_list, n_dirs, nodata, meta, px, global_mask_win,
        )
    elif openness_mode == 2:
        for r_open in openness_r_list:
            max_step = max(1, int(r_open / px))
            print(f"\n  > 開度 (Python 内蔵) R_open={r_open}m → 1〜{max_step}px（stride={step_stride}）")
            pos, neg = openness_pair(dem, px, r_open, n_dirs=n_dirs, step_stride=step_stride)
            pos_m = _apply_valid_mask(pos, global_mask_win)
            neg_m = _apply_valid_mask(neg, global_mask_win)
            rtag_open = int(round(r_open))
            tag_open = f"r{rtag_open}m_nd{n_dirs}dir"
            print("    - 地上開度 (openness_pos) ...")
            write_feature(f"{stem}_openness_pos_{tag_open}.tif", pos_m)
            print("    - 地下開度 (openness_neg) ...")
            write_feature(f"{stem}_openness_neg_{tag_open}.tif", neg_m)

    print("\n=== DONE: 全 10 特徴量（マルチスケール大窓版）の出力が完了しました。 ===")


if __name__ == "__main__":
    main()
