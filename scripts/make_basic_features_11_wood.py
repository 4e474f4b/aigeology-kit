#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_basic_features_11_wood.py

単一DEMから地形特徴量画像をマルチスケールで一括生成。

v10 (make_basic_features_10_scaleaware.py) からの変更点:
  1. スケール R を全指標で共通化
     - 比高 / 標準偏差 / TPI / TRI / 勾配 / 曲率 / 開度 すべて同一 R[m] を使用
     - 外接円 / 面積等価円 / 内接円 の変換選択を廃止
  2. 勾配・曲率を Wood (1996) / Evans (1979) 多項式フィッティングに変更
     - 「平滑化DEM → Horn」方式を廃止
     - slope / aspect / plan_curv / prof_curv / mean_curv をスケール別に出力
  3. 開度: Python 内蔵のみ（SAGA 依存を廃止）

出力（1スケールあたり）:
  {stem}_relief_{tag}.tif          比高
  {stem}_stddev_{tag}.tif          標準偏差
  {stem}_tpi_{tag}.tif             地形位置指数
  {stem}_tri_{tag}.tif             地形起伏指数
  {stem}_slope_{tag}.tif           勾配 [deg]
  {stem}_aspect_{tag}.tif          斜面方位 [deg] (0=北, 時計回り)
  {stem}_mean_curv_{tag}.tif       平均曲率 [1/m] (正=凸/丘, 負=凹/谷) ← 常時出力
  {stem}_plan_curv_{tag}.tif       平面曲率 [1/m] (Florinsky 1998 規則) ← オプション
  {stem}_prof_curv_{tag}.tif       縦断曲率 [1/m] (Florinsky 1998 規則) ← オプション
  {stem}_openness_pos_{tag}.tif    地上開度 [deg]
  {stem}_openness_neg_{tag}.tif    地下開度 [deg]
"""

import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import (
    uniform_filter,
    maximum_filter,
    minimum_filter,
    convolve,
    correlate,
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
            vals = [float(x.strip()) for x in s2.split(",") if x.strip()]
            if not vals:
                raise ValueError
            return vals
        except ValueError:
            print("  カンマ区切りの数値で入力してください（例: 30,50,90）。")


def _dedupe_preserve_order(items, key=None):
    if key is None:
        key = lambda x: x
    seen = set()
    out = []
    for x in items:
        k = key(x)
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def _r_to_k_list(r_list, px):
    """R[m] → 窓サイズ k（奇数, 最小3）"""
    k_list = []
    for R in r_list:
        if R <= 0:
            raise ValueError(f"R は正の値が必要: {R}")
        k = int(round(R / px))
        k = max(3, k)
        if k % 2 == 0:
            k += 1
        k_list.append(k)
    return k_list


def _winpix_from_meters(r_m, px, *, min_win=1, odd=True):
    if not np.isfinite(r_m) or r_m <= 0:
        k = min_win
    else:
        k = int(round(float(r_m) / float(px)))
        k = max(min_win, k)
    if odd and (k % 2 == 0):
        k += 1
    return k


def _apply_valid_mask(arr, win_pix):
    """外縁 win_pix//2 ピクセルを NaN にする（近傍が足りない領域を除去）"""
    h, w = arr.shape
    r = int(max(1, win_pix)) // 2
    mask = np.ones_like(arr, dtype=bool)
    mask[r:h-r, r:w-r] = False
    out = arr.copy()
    out[mask] = np.nan
    return out


# =============== 比高 / 標準偏差 / TPI / TRI ===============

def local_relief(arr, win_pix):
    """局所比高: max - min"""
    valid = np.isfinite(arr)
    k = int(max(1, win_pix))
    loc_max = maximum_filter(np.where(valid, arr, -np.inf), size=k, mode="nearest")
    loc_min = minimum_filter(np.where(valid, arr, +np.inf), size=k, mode="nearest")
    out = (loc_max - loc_min).astype(np.float32)
    out[~valid] = np.nan
    return out


def local_stddev(arr, win_pix):
    """局所標準偏差（NaN 対応）"""
    k = int(max(1, win_pix))
    valid = np.isfinite(arr).astype(np.float32)
    a = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    n = uniform_filter(valid, size=k, mode="nearest")
    s1 = uniform_filter(a, size=k, mode="nearest") * n
    s2 = uniform_filter(a * a, size=k, mode="nearest") * n
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n
        mean_sq = s2 / n
    var = mean_sq - mean * mean
    var[var < 0] = 0.0
    std = np.sqrt(var).astype(np.float32)
    std[n == 0] = np.nan
    return std


def local_mean_nan(arr, win_pix):
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


def local_tpi(arr, win_pix):
    """TPI = z - 局所平均（NaN 対応）"""
    m = local_mean_nan(arr, win_pix)
    out = (arr.astype(np.float32) - m).astype(np.float32)
    out[~np.isfinite(arr)] = np.nan
    return out


def _shift_edge(arr, dy, dx):
    """端を edge で埋めるシフト"""
    h, w = arr.shape
    py, px = abs(dy), abs(dx)
    a = np.pad(arr, ((py, py), (px, px)), mode="edge")
    y0 = py - dy
    x0 = px - dx
    return a[y0:y0+h, x0:x0+w]


def local_tri(arr, win_pix):
    """TRI: 8近傍差分の平均絶対値（スケール可変）"""
    k = int(max(1, win_pix))
    d = max(1, (k - 1) // 2)
    z = arr.astype(np.float32)
    c_valid = np.isfinite(z)
    shifts = [(-d,0),(d,0),(0,-d),(0,d),(-d,-d),(-d,d),(d,-d),(d,d)]
    sumdiff = np.zeros_like(z)
    cnt = np.zeros_like(z)
    for dy, dx in shifts:
        nb = _shift_edge(z, dy, dx)
        v = c_valid & np.isfinite(nb)
        sumdiff[v] += np.abs(z[v] - nb[v])
        cnt[v] += 1.0
    out = np.full_like(z, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[cnt > 0] = sumdiff[cnt > 0] / cnt[cnt > 0]
    out[~c_valid] = np.nan
    return out.astype(np.float32)


# =============== Wood (1996) 多項式フィッティング ===============
#
# k×k 窓に 2 次多項式 z = A*x² + B*x*y + C*y² + D*x + E*y + F を最小二乗フィット。
# x = East [m], y = North [m]（GeoTIFF の行方向を反転して North 正）
#
# 中心セル (x=0, y=0) における各偏微分:
#   p = dz/dx = D        (East 方向勾配)
#   q = dz/dy = E        (North 方向勾配)
#   r = d²z/dx² = 2A
#   s = d²z/dxdy = B
#   t = d²z/dy² = 2C
#
# 参照: Wood (1996), Evans (1979), Florinsky (1998)

_wood_kernel_cache: dict = {}


def _wood_kernels(k: int, px: float, py: float = None) -> dict:
    """
    Wood 多項式フィッティング用 k×k 畳み込みカーネルを計算してキャッシュする。

    数値安定性のため正規化座標（整数オフセット）で OLS を解いた後、
    物理スケール（m）に変換する。

    Returns: dict キー='A','B','C','D','E','F', 各値は shape (k,k) の float32 配列
    """
    if py is None:
        py = px
    cache_key = (k, round(px, 8), round(py, 8))
    if cache_key in _wood_kernel_cache:
        return _wood_kernel_cache[cache_key]

    n = k // 2
    j_vals = np.arange(-n, n + 1, dtype=np.float64)  # 列オフセット (East 正)
    i_vals = np.arange(-n, n + 1, dtype=np.float64)  # 行オフセット (South 正)
    J, I = np.meshgrid(j_vals, i_vals)

    # 正規化座標: xn = j (East), yn = -i (North)
    xn = J.ravel()
    yn = -I.ravel()

    # 設計行列 [xn², xn*yn, yn², xn, yn, 1]
    design = np.column_stack([xn**2, xn * yn, yn**2, xn, yn, np.ones_like(xn)])

    # OLS: K = (D'D)^{-1} D'  →  shape (6, k²)
    try:
        DtD = design.T @ design
        K = np.linalg.solve(DtD, design.T)
    except np.linalg.LinAlgError:
        K, _, _, _ = np.linalg.lstsq(design, np.eye(len(xn)), rcond=None)
        K = K.T

    # 正規化 → 物理スケール変換
    # 正規化空間: z = A'*xn² + ...  (xn = x/px)
    # 物理空間:   z = A*x² + ...   (x = xn*px)
    # → A = A'/px², B = B'/(px*py), C = C'/py², D = D'/px, E = E'/py, F = F'
    scale = np.array([
        1.0 / px**2,
        1.0 / (px * py),
        1.0 / py**2,
        1.0 / px,
        1.0 / py,
        1.0,
    ])

    names = ["A", "B", "C", "D", "E", "F"]
    kernels = {
        name: (K[i] * scale[i]).reshape(k, k).astype(np.float32)
        for i, name in enumerate(names)
    }

    _wood_kernel_cache[cache_key] = kernels
    return kernels


def wood_terrain_params(arr: np.ndarray, px: float, k: int, py: float = None) -> dict:
    """
    Wood (1996) 多項式フィッティングによる地形パラメータを一括計算。

    Parameters
    ----------
    arr  : DEM 配列（NaN = NoData）
    px   : x 方向ピクセルサイズ [m]
    k    : 窓サイズ（奇数）
    py   : y 方向ピクセルサイズ [m]（省略時 px と同値）

    Returns
    -------
    dict:
      slope     : 勾配角 [deg]
      aspect    : 斜面方位 [deg]  (0=北, 90=東, 時計回り)
      plan_curv : 平面曲率 [1/m]  (符号: Florinsky 1998 規則)
      prof_curv : 縦断曲率 [1/m]  (符号: Florinsky 1998 規則)
      mean_curv : 平均曲率 [1/m]  (正=凸/丘, 負=凹/谷 ← Florinsky 1998 規則)
    """
    if py is None:
        py = px
    kernels = _wood_kernels(k, px, py)
    nan_mask = ~np.isfinite(arr)
    filled = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)

    # 係数の計算（correlate: カーネル反転なし → 正しい勾配方向を保持）
    p = correlate(filled, kernels["D"], mode="nearest")  # dz/dx (East)
    q = correlate(filled, kernels["E"], mode="nearest")  # dz/dy (North)
    A = correlate(filled, kernels["A"], mode="nearest")
    B = correlate(filled, kernels["B"], mode="nearest")
    C = correlate(filled, kernels["C"], mode="nearest")

    r = 2.0 * A  # d²z/dx²
    s = B        # d²z/dxdy
    t = 2.0 * C  # d²z/dy²

    p2 = p * p
    q2 = q * q
    pq = p * q
    _EPS = 1e-10  # 水平面（p²+q²→0）での発散防止

    # 勾配 [deg]
    slope = np.degrees(np.arctan(np.hypot(p, q))).astype(np.float32)

    # 斜面方位 [deg]: 北=0, 東=90, 時計回り（Horn と同一規則）
    asp = np.degrees(np.arctan2(q, p))
    aspect = np.mod(90.0 - asp, 360.0).astype(np.float32)

    # 平面曲率 (plan curvature): 等高線方向の曲率
    # κ_h = -(q²r - 2pqs + p²t) / ((p²+q²) * √(1+p²+q²))
    with np.errstate(invalid="ignore", divide="ignore"):
        plan_curv = -(q2 * r - 2 * pq * s + p2 * t) / (
            (p2 + q2 + _EPS) * np.sqrt(1.0 + p2 + q2)
        )
    plan_curv = plan_curv.astype(np.float32)

    # 縦断曲率 (profile curvature): 傾斜方向の曲率
    # κ_v = -(p²r + 2pqs + q²t) / ((p²+q²) * (1+p²+q²)^1.5)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof_curv = -(p2 * r + 2 * pq * s + q2 * t) / (
            (p2 + q2 + _EPS) * np.power(1.0 + p2 + q2, 1.5)
        )
    prof_curv = prof_curv.astype(np.float32)

    # 平均曲率 (mean curvature)
    # H = -((1+q²)r - 2pqs + (1+p²)t) / (2*(1+p²+q²)^1.5)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_curv = -((1 + q2) * r - 2 * pq * s + (1 + p2) * t) / (
            2.0 * np.power(1.0 + p2 + q2, 1.5)
        )
    mean_curv = mean_curv.astype(np.float32)

    # NaN マスク適用
    for out in (slope, aspect, plan_curv, prof_curv, mean_curv):
        out[nan_mask] = np.nan

    return {
        "slope": slope,
        "aspect": aspect,
        "plan_curv": plan_curv,
        "prof_curv": prof_curv,
        "mean_curv": mean_curv,
    }


# =============== 開度（Python 内蔵版） ===============

def _unit_dirs(n_dirs):
    """0〜π の半円方向に n_dirs 本の単位ベクトル"""
    thetas = np.linspace(0.0, math.pi, n_dirs, endpoint=False)
    return np.cos(thetas), np.sin(thetas)


def openness_pair(dem, px, r_open, n_dirs=8, step_stride=1):
    """
    正開度 / 負開度を Python のみで計算。
    r_open[m] まで各方向にレイを飛ばし、最大仰角 / 最大俯角を評価する。
    """
    h, w = dem.shape
    cx = np.arange(w)
    cy = np.arange(h)
    X, Y = np.meshgrid(cx, cy)
    cos_t, sin_t = _unit_dirs(n_dirs)
    max_step = max(1, int(r_open / px))
    valid = np.isfinite(dem)
    pos_list, neg_list = [], []

    for k in range(n_dirs):
        dx, dy = cos_t[k], sin_t[k]
        alphas, betas = [], []

        for step in range(1, max_step + 1, step_stride):
            x_f = X + dx * step
            y_f = Y - dy * step
            x0 = np.floor(x_f).astype(int)
            y0 = np.floor(y_f).astype(int)
            x1, y1 = x0 + 1, y0 + 1
            inside = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
            if not inside.any():
                continue
            wx = x_f - x0
            wy = y_f - y0
            x0c = np.clip(x0, 0, w - 1)
            x1c = np.clip(x1, 0, w - 1)
            y0c = np.clip(y0, 0, h - 1)
            y1c = np.clip(y1, 0, h - 1)
            z_top = (
                dem[y0c, x0c] * (1 - wx) * (1 - wy)
                + dem[y0c, x1c] * wx * (1 - wy)
                + dem[y1c, x0c] * (1 - wx) * wy
                + dem[y1c, x1c] * wx * wy
            )
            d = step * px
            with np.errstate(invalid="ignore"):
                dz = z_top - dem
                alpha = np.degrees(np.arctan2(dz, d))
                beta = np.degrees(np.arctan2(-dz, d))
            alpha[~inside] = np.nan
            beta[~inside] = np.nan
            alphas.append(alpha)
            betas.append(beta)

        if not alphas:
            continue
        alpha_stack = np.stack(alphas, axis=0)
        beta_stack = np.stack(betas, axis=0)
        with np.errstate(invalid="ignore"):
            pos_list.append(90.0 - np.nanmax(alpha_stack, axis=0))
            neg_list.append(90.0 - np.nanmax(beta_stack, axis=0))

    if pos_list:
        pos = np.nanmean(np.stack(pos_list, axis=0), axis=0).astype(np.float32)
        neg = np.nanmean(np.stack(neg_list, axis=0), axis=0).astype(np.float32)
    else:
        pos = np.full_like(dem, np.nan, dtype=np.float32)
        neg = np.full_like(dem, np.nan, dtype=np.float32)
    pos[~valid] = np.nan
    neg[~valid] = np.nan
    return pos, neg


# =============== メイン処理 ===============

def main():
    print("=== DEM → 地形特徴量 一括出力 v11（Wood 多項式フィッティング対応） ===")

    dem_path_str = ask_path("入力DEM GeoTIFF のパス", must_exist=True)
    dem_path = Path(dem_path_str)
    stem = dem_path.stem

    default_out = str(dem_path.parent / (stem + "_features"))
    out_dir_str = ask_path("出力フォルダ", must_exist=False, default=default_out)
    out_dir = Path(out_dir_str)

    if out_dir.suffix.lower() in (".tif", ".tiff", ".vrt") or (
        out_dir.exists() and out_dir.is_file()
    ):
        print(f"[WARN] 出力フォルダにファイルが指定されています → {default_out} に補正します。")
        out_dir = Path(default_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # DEM 読み込み
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999.0
        dem = np.where(np.isfinite(dem), dem, np.nan).astype(np.float32)
        transform: Affine = src.transform
        px = float(transform.a)
        py = float(-transform.e)
        meta = src.meta.copy()
        meta.update(
            dtype="float32", nodata=nodata, compress="lzw",
            tiled=True, blockxsize=512, blockysize=512, count=1,
        )

    h, w = dem.shape
    print(f"\n[OK] DEM サイズ: {w} × {h} px, ピクセルサイズ: {px:.3f} × {py:.3f} m")

    # ── スケール R 入力（全指標共通） ──
    print("\n=== スケール R[m] の指定（全指標共通） ===")
    print("  指定した R[m] は以下すべての指標に適用されます:")
    print("    比高 / 標準偏差 / TPI / TRI       ← k×k 窓 (k ≒ R/px, 奇数)")
    print("    勾配 / 曲率（Wood フィッティング）← 同一 k×k 窓")
    print("    地上 / 地下開度               ← レイ最大距離 = R[m]")
    print(f"  px={px:.3f}m, 例: R=30m → k=3 (10m DEM), k=7 (4m DEM)")

    r_list_raw = ask_float_list("R[m]（カンマ区切り可, 例: 30,50,90）", [30.0, 50.0, 90.0])
    r_list = _dedupe_preserve_order(
        [float(v) for v in r_list_raw if float(v) > 0],
        key=lambda v: round(v, 6),
    )

    k_list = _r_to_k_list(r_list, px)
    r_eff_list = [((k - 1) / 2.0) * px for k in k_list]

    print("\n  → R と窓サイズ k の対応:")
    for R, k, R_eff in zip(r_list, k_list, r_eff_list):
        print(f"    R={R:.3g}m → k={k} ({k}×{k} 窓), R_eff ≒ {R_eff:.3f}m")

    # 開度パラメータ
    n_dirs = ask_int("\n開度の方向数 n_dirs（8 or 16 推奨）", 8)
    step_stride = ask_int(
        "開度サンプリング間隔 [px]（1=全ステップ / 2=2px刻み / ...）", 1
    )
    print("\n  > 開度の評価設定")
    for R in r_list:
        n_px = max(1, int(round(R / px)))
        print(f"    R={R:.3g}m → 1〜{n_px}px 先まで（step_stride={step_stride}px 刻み）を評価")

    if h * w > 50_000_000:
        print(
            f"\n[WARN] DEM 総ピクセル数 {h*w/1e6:.1f}M。"
            "開度計算は時間がかかる場合があります。"
        )

    # 外縁マスク: 全指標で最大 k を統一
    mask_win_candidates = list(k_list)
    for R in r_list:
        mask_win_candidates.append(_winpix_from_meters(R, px, min_win=1, odd=True))
    global_mask_win = max(int(k) for k in mask_win_candidates) if mask_win_candidates else 3
    print(f"\n[OK] 外縁マスク共通窓: k={global_mask_win}")

    # 平面曲率・縦断曲率の出力オプション
    print("\n=== 曲率の出力オプション ===")
    print("  平均曲率（mean_curv）は常に出力します。")
    print("  1: 平均曲率のみ")
    print("  2: 平均曲率 + 平面曲率（plan_curv）+ 縦断曲率（prof_curv）")
    curv_mode = ask_int("番号を選んでください", 1)
    output_extra_curv = (curv_mode == 2)

    def write_feature(name: str, data: np.ndarray):
        arr_out = data.astype(np.float32)
        arr_out = np.where(np.isfinite(arr_out), arr_out, nodata).astype(np.float32)
        with rasterio.open(out_dir / name, "w", **meta) as dst:
            dst.write(arr_out, 1)

    # ── 特徴量計算（スケール別） ──
    for R, k, R_eff in zip(r_list, k_list, r_eff_list):
        tag = f"rin{int(round(R))}m_k{k}_reff{int(round(R_eff))}m"
        print(f"\n=== スケール R={R:.3g}m, k={k} ===")

        print("  > 比高 / 標準偏差 / TPI / TRI ...")
        write_feature(f"{stem}_relief_{tag}.tif",
                      _apply_valid_mask(local_relief(dem, k), global_mask_win))
        write_feature(f"{stem}_stddev_{tag}.tif",
                      _apply_valid_mask(local_stddev(dem, k), global_mask_win))
        write_feature(f"{stem}_tpi_{tag}.tif",
                      _apply_valid_mask(local_tpi(dem, k), global_mask_win))
        write_feature(f"{stem}_tri_{tag}.tif",
                      _apply_valid_mask(local_tri(dem, k), global_mask_win))

        print("  > Wood 多項式フィッティング（勾配・曲率）...")
        params = wood_terrain_params(dem, px, k, py)
        output_names = {
            "slope":     f"{stem}_slope_{tag}.tif",
            "aspect":    f"{stem}_aspect_{tag}.tif",
            "mean_curv": f"{stem}_mean_curv_{tag}.tif",
        }
        if output_extra_curv:
            output_names["plan_curv"] = f"{stem}_plan_curv_{tag}.tif"
            output_names["prof_curv"] = f"{stem}_prof_curv_{tag}.tif"
        for param_key, fname in output_names.items():
            write_feature(fname, _apply_valid_mask(params[param_key], global_mask_win))
        curv_msg = "slope / aspect / mean_curv" + (" / plan_curv / prof_curv" if output_extra_curv else "")
        print(f"    [OK] {curv_msg} → {tag}")

    # ── 開度 ──
    for R in r_list:
        rtag = int(round(R))
        max_step = max(1, int(R / px))
        print(f"\n  > 開度 R={R:.3g}m → 1〜{max_step}px 先（stride={step_stride}）")
        pos, neg = openness_pair(dem, px, R, n_dirs=n_dirs, step_stride=step_stride)
        tag_open = f"r{rtag}m_nd{n_dirs}dir"
        write_feature(
            f"{stem}_openness_pos_{tag_open}.tif",
            _apply_valid_mask(pos, global_mask_win),
        )
        write_feature(
            f"{stem}_openness_neg_{tag_open}.tif",
            _apply_valid_mask(neg, global_mask_win),
        )
        print(f"    [OK] openness_pos / openness_neg → {tag_open}")

    print("\n=== DONE: 全特徴量の出力が完了しました。===")
    print(f"  出力先: {out_dir}")


if __name__ == "__main__":
    main()
