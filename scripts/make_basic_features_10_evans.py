#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_basic_features_10_evans.py

単一DEMから「比高・標準偏差・傾斜・曲率・開度・TRI」など
地形分類用の基本特徴量を、マルチスケールで一括生成するスクリプト。

make_basic_features_10_largekern.py との違い
----------------------------------------------
全 10 特徴量を k×k 窓内の「全ピクセル」を使って計算する。

  比高・標準偏差・TPI
      → k×k 窓内の全ピクセルを集計（変更なし）

  TRI
      → 窓内全 k² ピクセルとの RMSD（二乗平均平方根偏差）
         TRI = sqrt( mean((z_neighbor - z_center)²) )
             = sqrt( stddev² + tpi² )

  傾斜・方位・ラプラシアン・平均曲率
      → Evans (1980) 最小二乗多項式フィット
         k×k 窓全ピクセルに z = Ax²+By²+Cxy+Dx+Ey+F を当てはめ
         係数から各微分量を算出

  開度（地上・地下）
      → R_open = R_eff = (k-1)/2 × px でレイキャスト（変更なし）

largekern との比較
  largekern : d ピクセル離れた端点のみを差分計算
  evans     : 窓内全ピクセルを最小二乗で使用（統計的に頑健）

"""

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import (
    correlate1d,
    uniform_filter,
    uniform_filter1d,
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
            print("  カンマ区切りの数値で入力してください。")


def ask_k_list(prompt, default_list, px: float):
    """奇数・3以上の窓サイズ k をカンマ区切りで受け取る。"""
    default_str = ",".join(str(v) for v in default_list)
    while True:
        s = input(f"{prompt} [{default_str}]: ").strip()
        if not s:
            return list(default_list)
        try:
            vals = [int(x.strip()) for x in s.split(",") if x.strip()]
            if not vals:
                raise ValueError
        except ValueError:
            print("  カンマ区切りの整数で入力してください（例: 3,5,7,9,15,31）。")
            continue

        errors = []
        for v in vals:
            if v < 3:
                errors.append(f"  [ERROR] k={v} は最小値 3 を下回っています（3×3 が最小窓）。")
            elif v % 2 == 0:
                errors.append(
                    f"  [ERROR] k={v} は偶数です。中心ピクセルが存在しないため使用できません"
                    f"（例: {v+1} を使ってください）。"
                )
        if errors:
            for e in errors:
                print(e)
            continue

        seen, unique = set(), []
        for v in vals:
            if v not in seen:
                seen.add(v)
                unique.append(v)
        return unique


def _winpix_from_meters(r_m, px, *, min_win=1, odd=True):
    if not np.isfinite(r_m) or r_m <= 0:
        k = min_win
    else:
        k = int(round(float(r_m) / float(px)))
        k = max(min_win, k)
    if odd and k % 2 == 0:
        k += 1
    return k


def _apply_valid_mask(arr, win_pix):
    """外縁 floor(k/2) ピクセルを NaN にする。"""
    h, w = arr.shape
    k = int(max(1, win_pix))
    r = k // 2
    mask = np.ones_like(arr, dtype=bool)
    mask[r:h-r, r:w-r] = False
    out = arr.copy()
    out[mask] = np.nan
    return out


# =============== 比高 / 標準偏差 / TPI ===============

def local_relief(arr, win_pix):
    valid = np.isfinite(arr)
    k = int(max(1, win_pix))
    loc_max = maximum_filter(np.where(valid, arr, -np.inf), size=k, mode="nearest")
    loc_min = minimum_filter(np.where(valid, arr, +np.inf), size=k, mode="nearest")
    out = (loc_max - loc_min).astype(np.float32)
    out[~valid] = np.nan
    return out


def local_stddev(arr, win_pix):
    k = int(max(1, win_pix))
    valid = np.isfinite(arr).astype(np.float32)
    a = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    n     = uniform_filter(valid,   size=k, mode="nearest")
    sum1  = uniform_filter(a,       size=k, mode="nearest") * n
    sum2  = uniform_filter(a * a,   size=k, mode="nearest") * n
    with np.errstate(invalid="ignore", divide="ignore"):
        mean    = sum1 / n
        mean_sq = sum2 / n
    var = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(var).astype(np.float32)
    std[n == 0] = np.nan
    return std


def local_mean_nan(arr, win_pix):
    k = int(max(1, win_pix))
    valid = np.isfinite(arr).astype(np.float32)
    a = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    n = uniform_filter(valid, size=k, mode="nearest")
    s = uniform_filter(a,     size=k, mode="nearest")
    out = np.full_like(a, np.nan, dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[n > 0] = s[n > 0] / n[n > 0]
    out[~np.isfinite(arr)] = np.nan
    return out


def local_tpi(arr, win_pix):
    m = local_mean_nan(arr, win_pix)
    out = (arr.astype(np.float32) - m).astype(np.float32)
    out[~np.isfinite(arr)] = np.nan
    return out


# =============== TRI（全 k×k ピクセル使用） ===============

def tri_window(arr, win_pix):
    """
    窓内全 k×k ピクセルとの RMSD。
    TRI = sqrt( mean((z_neighbor - z_center)²) )
        = sqrt( stddev² + tpi² )
    """
    std = local_stddev(arr, win_pix)
    tpi = local_tpi(arr, win_pix)
    out = np.sqrt(std.astype(np.float64)**2 + tpi.astype(np.float64)**2)
    out[~np.isfinite(arr)] = np.nan
    return out.astype(np.float32)


# =============== Evans (1980) LS 多項式フィット ===============

def _evans_precompute(k: int) -> dict:
    """
    k×k 等間隔格子に z = Ax²+By²+Cxy+Dx+Ey+F を LS フィットする際の
    定数・カーネルを事前計算する（次元は pixel 単位）。

    対称格子では正規方程式が分解し:
      D, E  は独立: D = Σ pj*z / (k * s2),  E = -Σ pi*z / (k * s2)
      C     は独立: C = Σ pi*pj*z / s2²
      A+B   : 2(A+B) = 2k(Px2z_part+Py2z_part - 2*s2*Pz_mean) / denom_lap
      A-B   : 2(A-B) = 2k(Px2z_part-Py2z_part) / denom_lap

    ここで
      Px2z_part = correlate1d(uniform_filter1d(z,k,axis=0), p², axis=1)
                = (1/k) * Σij pj² * z[i,j]
      Pz_mean   = uniform_filter(z, k)  (= k×k 窓平均)
      denom_lap = k*s4 - s2²
    """
    d = (k - 1) // 2
    p  = np.arange(-d, d + 1, dtype=np.float64)
    s2 = float(np.sum(p ** 2))
    s4 = float(np.sum(p ** 4))
    return {
        "k":          k,
        "p":          p,
        "p2":         p ** 2,
        "s2":         s2,
        "denom_lap":  k * s4 - s2 ** 2,
        "denom_cxy":  s2 ** 2,
    }


def _nan_fill(arr):
    """NaN を 0 で埋めた float64 配列を返す。"""
    return np.where(np.isfinite(arr), arr, 0.0).astype(np.float64)


def evans_gradients(arr: np.ndarray, px: float, k: int, ec: dict):
    """
    Evans LS 勾配 (gx, gy)。全 k×k ピクセルを使用。

    gx = Σij pj*z[i,j] / (k * s2 * h)
       = correlate1d(uniform_filter1d(z,k,axis=0), p, axis=1) / (s2 * h)

    gy は行方向（南向き増加）を北向き正に補正して返す。
    """
    s2, p = ec["s2"], ec["p"]
    mask = ~np.isfinite(arr)
    zf   = _nan_fill(arr)

    row_mean = uniform_filter1d(zf, k, axis=0, mode='nearest')
    gx = correlate1d(row_mean, p, axis=1, mode='nearest') / (s2 * px)

    col_mean = uniform_filter1d(zf, k, axis=1, mode='nearest')
    gy_raw = correlate1d(col_mean, p, axis=0, mode='nearest') / (s2 * px)
    gy = -gy_raw  # 行インデックス増加=南方向 → 反転して北向き正

    gx[mask] = np.nan
    gy[mask] = np.nan
    return gx.astype(np.float32), gy.astype(np.float32)


def evans_slope_deg(arr: np.ndarray, px: float, k: int, ec: dict) -> np.ndarray:
    """勾配角 [deg]（Evans LS, 全 k×k ピクセル使用）"""
    gx, gy = evans_gradients(arr, px, k, ec)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    slope[~np.isfinite(arr)] = np.nan
    return slope.astype(np.float32)


def evans_aspect_deg(arr: np.ndarray, px: float, k: int, ec: dict) -> np.ndarray:
    """斜面方位 [deg]（0=東, 90=北, Evans LS, 全 k×k ピクセル使用）"""
    gx, gy = evans_gradients(arr, px, k, ec)
    asp = np.degrees(np.arctan2(gy, gx))
    asp = np.mod(90.0 - asp, 360.0)
    asp[~np.isfinite(arr)] = np.nan
    return asp.astype(np.float32)


def _evans_second_derivs(arr: np.ndarray, px: float, k: int, ec: dict):
    """
    Evans LS 2次偏微分 (zxx, zyy, zxy)。全 k×k ピクセル使用。

    2(A+B) = 2k(Px2z_part + Py2z_part - 2*s2*Pz_mean) / (denom_lap * h²)
    2(A-B) = 2k(Px2z_part - Py2z_part) / (denom_lap * h²)
    zxx = 2A = (ApB + AmB) / 2,  zyy = 2B = (ApB - AmB) / 2
    zxy = C  = correlate_x(correlate_y(z, p)) / (denom_cxy * h²)
    """
    k_val, s2, p, p2 = ec["k"], ec["s2"], ec["p"], ec["p2"]
    denom_lap, denom_cxy = ec["denom_lap"], ec["denom_cxy"]
    mask = ~np.isfinite(arr)
    zf   = _nan_fill(arr)
    h2   = px * px

    # Px2z_part = (1/k) * Σij pj² * z[i,j]
    Px2z_part = correlate1d(
        uniform_filter1d(zf, k_val, axis=0, mode='nearest'), p2, axis=1, mode='nearest'
    )
    # Py2z_part = (1/k) * Σij pi² * z[i,j]
    Py2z_part = correlate1d(
        uniform_filter1d(zf, k_val, axis=1, mode='nearest'), p2, axis=0, mode='nearest'
    )
    # Pz_mean = k×k 窓平均
    Pz_mean = uniform_filter(zf, k_val, mode='nearest')

    ApB = 2.0 * k_val * (Px2z_part + Py2z_part - 2.0 * s2 * Pz_mean) / (denom_lap * h2)
    AmB = 2.0 * k_val * (Px2z_part - Py2z_part)                      / (denom_lap * h2)

    zxx = (ApB + AmB) / 2.0   # = 2A
    zyy = (ApB - AmB) / 2.0   # = 2B

    # C = Σij pi*pj*z / (s2² * h²)
    zxy = correlate1d(
        correlate1d(zf, p, axis=0, mode='nearest'), p, axis=1, mode='nearest'
    ) / (denom_cxy * h2)

    for arr2d in (zxx, zyy, zxy):
        arr2d[mask] = np.nan

    return zxx.astype(np.float32), zyy.astype(np.float32), zxy.astype(np.float32)


def evans_laplacian(arr: np.ndarray, px: float, k: int, ec: dict) -> np.ndarray:
    """ラプラシアン = zxx + zyy（Evans LS, 全 k×k ピクセル使用）"""
    zxx, zyy, _ = _evans_second_derivs(arr, px, k, ec)
    lap = zxx + zyy
    lap[~np.isfinite(arr)] = np.nan
    return lap.astype(np.float32)


def evans_mean_curvature(arr: np.ndarray, px: float, k: int, ec: dict) -> np.ndarray:
    """
    平均曲率 H（Evans LS, 全 k×k ピクセル使用）
    H = [(1+zy²)zxx - 2*zx*zy*zxy + (1+zx²)zyy] / [2*(1+zx²+zy²)^1.5]
    """
    gx, gy   = evans_gradients(arr, px, k, ec)
    zxx, zyy, zxy = _evans_second_derivs(arr, px, k, ec)

    gx, gy   = gx.astype(np.float64), gy.astype(np.float64)
    zxx_d    = zxx.astype(np.float64)
    zyy_d    = zyy.astype(np.float64)
    zxy_d    = zxy.astype(np.float64)

    zx2   = gx * gx
    zy2   = gy * gy
    denom = 2.0 * np.power(1.0 + zx2 + zy2, 1.5)
    num   = (1.0 + zy2) * zxx_d - 2.0 * gx * gy * zxy_d + (1.0 + zx2) * zyy_d

    with np.errstate(invalid="ignore", divide="ignore"):
        H = num / denom
    H[~np.isfinite(arr)] = np.nan
    return H.astype(np.float32)


# =============== 開度（Python 内蔵版） ===============

def _unit_dirs(n_dirs):
    thetas = np.linspace(0.0, math.pi, n_dirs, endpoint=False)
    return np.cos(thetas), np.sin(thetas)


def openness_pair(dem, px, r_open, n_dirs=8, step_stride=1):
    """正開度 / 負開度（レイキャスティング, R_open まで）"""
    h, w = dem.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    cos_t, sin_t = _unit_dirs(n_dirs)
    pos_list, neg_list = [], []
    max_step = max(1, int(r_open / px))
    valid = np.isfinite(dem)

    for ki in range(n_dirs):
        dx, dy = cos_t[ki], sin_t[ki]
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
            x0c = np.clip(x0, 0, w-1); x1c = np.clip(x1, 0, w-1)
            y0c = np.clip(y0, 0, h-1); y1c = np.clip(y1, 0, h-1)
            z_top = (dem[y0c,x0c]*(1-wx)*(1-wy) + dem[y0c,x1c]*wx*(1-wy) +
                     dem[y1c,x0c]*(1-wx)*wy      + dem[y1c,x1c]*wx*wy)
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
        with np.errstate(invalid="ignore"):
            pos_list.append(90.0 - np.nanmax(np.stack(alphas), axis=0))
            neg_list.append(90.0 - np.nanmax(np.stack(betas),  axis=0))

    if pos_list:
        pos = np.nanmean(np.stack(pos_list), axis=0).astype(np.float32)
        neg = np.nanmean(np.stack(neg_list), axis=0).astype(np.float32)
    else:
        pos = np.full_like(dem, np.nan, dtype=np.float32)
        neg = np.full_like(dem, np.nan, dtype=np.float32)
    pos[~valid] = np.nan
    neg[~valid] = np.nan
    return pos, neg


# =============== SAGA 開度 ===============

_GFLAGS_ERROR = "flag 'help' was defined more than once"


def confirm_saga_openness_tool(saga_cmd_path, tool_id=5):
    try:
        result = subprocess.run(
            [saga_cmd_path, "ta_lighting", "--help"],
            capture_output=True, text=True, timeout=15,
        )
        output = result.stdout + result.stderr
        if _GFLAGS_ERROR in output:
            print(f"  [WARN] SAGA の gflags 競合エラーを検出しました → Python 内蔵モードに切り替えます。")
            return False
        if str(tool_id) in output or "Topographic Openness" in output:
            return True
        result2 = subprocess.run(
            [saga_cmd_path, "ta_lighting"],
            capture_output=True, text=True, timeout=15,
        )
        out2 = result2.stdout + result2.stderr
        if _GFLAGS_ERROR in out2:
            print(f"  [WARN] SAGA の gflags 競合エラーを検出しました → Python 内蔵モードに切り替えます。")
            return False
        print(out2[:2000])
        return True
    except Exception as e:
        print(f"  SAGA 確認中にエラー: {e}")
        return False


def compute_openness_with_saga(
    saga_cmd_path, dem_path, out_dir, stem,
    k_r_pairs, n_dirs, nodata, meta, px, global_mask_win,
):
    """SAGA ta_lighting 5 で正開度 / 負開度を計算する。"""
    for k, r_open in k_r_pairs:
        R_eff = ((k - 1) / 2.0) * px
        tag   = f"k{k}_reff{int(round(R_eff))}m"
        pos_tif = out_dir / f"{stem}_openness_pos_{tag}.tif"
        neg_tif = out_dir / f"{stem}_openness_neg_{tag}.tif"
        tmp_pos = out_dir / f"_tmp_pos_k{k}.sdat"
        tmp_neg = out_dir / f"_tmp_neg_k{k}.sdat"
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
                        data = np.degrees(src.read(1).astype(np.float32))
                    data[~np.isfinite(data)] = np.nan
                    data = _apply_valid_mask(data, global_mask_win)
                    arr  = np.where(np.isfinite(data), data, nodata).astype(np.float32)
                    with rasterio.open(out_tif, "w", **meta) as dst:
                        dst.write(arr, 1)
                    print(f"    [{label}] → {out_tif.name}")
                    for ext in (".sdat", ".sgrd", ".prj", ".mgrd"):
                        p2 = tmp.with_suffix(ext)
                        if p2.exists():
                            p2.unlink()
                else:
                    print(f"  [WARN] SAGA 出力 {sdat} が見つかりません。")
        except Exception as e:
            print(f"  [ERROR] SAGA openness 失敗: {e}")


# =============== メイン処理 ===============

def main():
    print("=== DEM → 10特徴量 一括出力（Evans LS・全ピクセル使用） ===")

    dem_path_str = ask_path("入力DEM GeoTIFF のパス", must_exist=True)
    dem_path = Path(dem_path_str)
    stem = dem_path.stem

    default_out_dir = str(dem_path.with_suffix("").parent / (stem + "_features"))
    out_dir_str = ask_path(
        f"出力フォルダ [{default_out_dir}]", must_exist=False, default=default_out_dir
    )
    out_dir = Path(out_dir_str)
    if out_dir.suffix.lower() in (".tif", ".tiff", ".vrt"):
        print(f"[WARN] ファイルパスが指定されました → {default_out_dir} に補正します。")
        out_dir = Path(default_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dem_path) as src:
        dem    = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999.0
        dem    = np.where(np.isfinite(dem), dem, np.nan).astype(np.float32)
        transform: Affine = src.transform
        px = float(transform.a)
        meta = src.meta.copy()
        meta.update(
            dtype="float32", nodata=nodata,
            compress="lzw", tiled=True, blockxsize=512, blockysize=512, count=1,
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
            if not confirm_saga_openness_tool(saga_cmd_path):
                print("  → SAGA 開度は使用せず Python 内蔵に切り替えます。")
                openness_mode = 2
    else:
        print("  saga_cmd が見つかりませんでした。")
        print("  1: Python 内蔵の開度計算を使う")
        print("  2: 開度は計算しない")
        openness_mode = 2 if ask_int("番号を選んでください", 1) == 1 else 3

    # 窓サイズ指定
    print("\n[スケール指定（全10特徴量共通）]")
    print(f"  ピクセルサイズ: {px:.3f} m")
    print("  窓サイズ k（奇数・3以上）をカンマ区切りで指定してください。")
    print("  ※ 偶数は中心ピクセルが存在しないため使用できません。")
    print("  ※ 最小は k=3（3×3窓）です。")
    print(f"  例) k=3（{3*px:.0f}m相当）, k=5（{5*px:.0f}m相当）,"
          f" k=9（{9*px:.0f}m相当）, k=31（{31*px:.0f}m相当）\n")

    k_list = ask_k_list("窓サイズ k（例: 3,5,7,9,15,31）", [3, 5, 7, 9, 15, 31], px)

    print("\n  → 指定窓サイズと有効半径:")
    for k in k_list:
        R_eff = ((k - 1) / 2.0) * px
        print(f"    k={k}（{k}×{k}窓）  R_eff={R_eff:.1f}m  カバー範囲={k*px:.0f}m")

    # 開度 n_dirs・step_stride
    n_dirs, step_stride = 8, 1
    if openness_mode != 3:
        print(f"\n  → 開度の R_open は各 k の有効半径（R_eff）を自動使用します。")
        for k in k_list:
            R_eff = ((k - 1) / 2.0) * px
            print(f"     k={k} → R_open={R_eff:.1f}m")
        n_dirs = ask_int("開度の方向数 n_dirs（8 or 16 推奨）", 8)
        if openness_mode == 2:
            step_stride = ask_int(
                "開度のサンプリング間隔（1:毎px / 2:2px刻み / 3:3px刻み）", 1
            )
            if h * w > 50_000_000:
                print(
                    f"\n---注意--- DEM 総ピクセル数 {h*w/1e6:.1f} 百万。"
                    "\n       Python 内蔵開度は時間がかかる場合があります。"
                )

    # 外縁マスク窓
    mask_win_candidates = list(k_list)
    if openness_mode != 3:
        for k in k_list:
            R_eff = ((k - 1) / 2.0) * px
            mask_win_candidates.append(_winpix_from_meters(R_eff, px, min_win=1, odd=True))
    global_mask_win = max(int(v) for v in mask_win_candidates) if mask_win_candidates else 3
    print(f"\n---情報--- 外縁マスクは共通窓 k={global_mask_win} を使用します。")

    def write_feature(name, data):
        arr = np.where(np.isfinite(data), data, nodata).astype(np.float32)
        with rasterio.open(out_dir / name, "w", **meta) as dst:
            dst.write(arr, 1)

    # ─────────────────────────────────────────────────────────────
    # マルチスケールループ：全 10 特徴量を Evans LS で計算
    # ─────────────────────────────────────────────────────────────
    saga_k_r_pairs = []  # SAGA 開度用 (k, r_open) リスト

    for k in k_list:
        R_eff = ((k - 1) / 2.0) * px
        tag   = f"k{k}_reff{int(round(R_eff))}m"
        ec    = _evans_precompute(k)

        print(f"\n  > k={k}（{k}×{k}窓）  R_eff={R_eff:.1f}m ...")

        # 比高・標準偏差・TPI
        rel = local_relief(dem, k)
        write_feature(f"{stem}_relief_{tag}.tif",  _apply_valid_mask(rel, global_mask_win))

        std = local_stddev(dem, k)
        write_feature(f"{stem}_stddev_{tag}.tif",  _apply_valid_mask(std, global_mask_win))

        tpi = local_tpi(dem, k)
        write_feature(f"{stem}_tpi_{tag}.tif",     _apply_valid_mask(tpi, global_mask_win))

        # TRI（全 k×k ピクセル使用）
        tri = tri_window(dem, k)
        write_feature(f"{stem}_tri_{tag}.tif",     _apply_valid_mask(tri, global_mask_win))

        # 傾斜・方位・ラプラシアン・平均曲率（Evans LS）
        slope = evans_slope_deg(dem, px, k, ec)
        write_feature(f"{stem}_slope_deg_{tag}.tif",
                      _apply_valid_mask(slope, global_mask_win))

        aspect = evans_aspect_deg(dem, px, k, ec)
        write_feature(f"{stem}_aspect_deg_{tag}.tif",
                      _apply_valid_mask(aspect, global_mask_win))

        lap = evans_laplacian(dem, px, k, ec)
        write_feature(f"{stem}_laplacian_{tag}.tif",
                      _apply_valid_mask(lap, global_mask_win))

        mcurv = evans_mean_curvature(dem, px, k, ec)
        write_feature(f"{stem}_mean_curvature_{tag}.tif",
                      _apply_valid_mask(mcurv, global_mask_win))

        print(f"     比高・標準偏差・TPI・TRI・傾斜・方位・ラプラシアン・平均曲率 → 完了")

        # 開度（Python 内蔵）
        if openness_mode == 2:
            max_step = max(1, int(R_eff / px))
            print(f"  > 開度 (Python 内蔵) R_open={R_eff:.1f}m → {max_step}px先まで ...")
            pos, neg = openness_pair(dem, px, R_eff, n_dirs=n_dirs, step_stride=step_stride)
            write_feature(f"{stem}_openness_pos_{tag}.tif",
                          _apply_valid_mask(pos, global_mask_win))
            write_feature(f"{stem}_openness_neg_{tag}.tif",
                          _apply_valid_mask(neg, global_mask_win))
            print(f"     地上開度・地下開度 → 完了")

        elif openness_mode == 1:
            saga_k_r_pairs.append((k, R_eff))

    # 開度（SAGA：全スケールまとめて処理）
    if openness_mode == 1 and saga_k_r_pairs:
        print("\n  > 開度 (SAGA) を計算中 ...")
        compute_openness_with_saga(
            saga_cmd_path, dem_path, out_dir, stem,
            saga_k_r_pairs, n_dirs, nodata, meta, px, global_mask_win,
        )
    elif openness_mode == 3:
        print("\n  > 開度の計算はスキップされました。")

    print("\n=== DONE: 全 10 特徴量（Evans LS・全ピクセル使用）の出力が完了しました。 ===")


if __name__ == "__main__":
    main()
