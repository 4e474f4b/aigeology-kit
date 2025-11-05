#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DEMから地形特徴量を一括生成（GeoTIFF）し、任意のグリッドで学習データを出力（CSV/Parquet/GPKG）。

本版のポイント:
- 指定スケール r[m] ごとに 4通りの前処理で同じ8指標を算出
  A) **RSU**: r[m]に集約(average) → 元解像度へ復元（**bilinear/cubic選択**）
  B) **RSD**: r[m]に集約(average) → **粗解像のまま出力**
  C) **SM** : **box/gauss選択**（1mのまま）
  D) **RES**: **ハイパス**（DEM − LowPass_r）  ※LowPassはRSUまたはSM
- 各スケールで必要な組合せを出力（標高は出さない、MPIは出さない）
  8指標 = 比高(relief), 偏差(stddev), 傾斜(slope), 傾斜方位(aspect),
          ラプラシアン(laplacian), 平均曲率(mean_curvature),
          地上開度(openness_pos), 地下開度(openness_neg)

出力ファイル名の規約:
  {stem}_{metric}_r{m}m_{variant}.tif
    metric  : relief | stddev | slope_deg | aspect_deg | laplacian | meancurv
              | openness_pos | openness_neg
    variant : rsu (resampled-up) | rsd (resampled-down) | sm_* | residual

必要: numpy, pandas, rasterio, geopandas, shapely, scipy, pyarrow(任意), tqdm, numba(任意)
"""

import os, sys, glob
from pathlib import Path
import math

import numpy as np
import pandas as pd
from tqdm import tqdm

import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

import geopandas as gpd
from scipy.ndimage import gaussian_filter, uniform_filter, maximum_filter, minimum_filter
import gc

# ---- GDAL のうるさいログを切る / 文字化け回避 ----
os.environ.setdefault("CPL_LOG", "NUL")               # Windows: NUL, Linux/Mac: /dev/null
os.environ.setdefault("GDAL_FILENAME_IS_UTF8", "YES")
# （必要なら）端末に依存するログ自体を抑制
os.environ.setdefault("CPL_DEBUG", "OFF")

# 環境変数でON/OFF（既定ONを推奨）
USE_WINDOWED = os.environ.get("FEATURES_WINDOWED_WRITE", "1") == "1"
WINDOW_TILE = int(os.environ.get("FEATURES_WINDOW_TILE", "256"))

def _safe_write_windowed(dst, arr, tilesize=None):
    ts = tilesize or WINDOW_TILE
    H, W = arr.shape
    for y0 in range(0, H, ts):
        for x0 in range(0, W, ts):
            y1 = min(y0+ts, H); x1 = min(x0+ts, W)
            a = arr[y0:y1, x0:x1]
            a = np.where(np.isfinite(a), a.astype(np.float32), np.float32(-9999.0))
            dst.write(a, 1, window=((y0, y1), (x0, x1)))

OPENNESS_STEP_STRIDE = int(os.environ.get("OPENNESS_STEP_STRIDE", "2"))
# ← 未設定なら空文字にしておき、対話で決める
RES_BASE = os.environ.get("RES_BASE", "").lower()
RES_GAUSS_SIGMA_PX = os.environ.get("RES_GAUSS_SIGMA_PX", "")

# RES の出力枚数を制御（1=8指標すべて、0=dem/slope/relief の互換モード）
RES_ALL_METRICS = os.getenv("RES_ALL_METRICS", "1") == "1"

RSD_OUTPUT_GRID = (os.getenv("RSD_OUTPUT_GRID") or "1m").lower()  # "1m" or "coarse"
if RSD_OUTPUT_GRID not in {"1m", "coarse"}:
    RSD_OUTPUT_GRID = "1m"

# =============== ユーティリティ（対話） ===============

# === meter → pixel helper ===
def _pixsize_from_transform(tr) -> float:
    """
    ピクセルサイズ[m]。一般的な北向きラスタを想定（回転なし）。
    回転がある特殊ケースは abs(tr.a) で近似。
    """
    return abs(float(tr.a))

def _winpix_from_meters(r_m: float, px: float, *, min_win: int = 1, odd: bool = True) -> int:
    """
    r_m[m] の計算範囲をピクセルに換算。min_win 以上、必要なら奇数化。
    例）0.5m DEM で r_m=3m → round(3/0.5)=6 → 奇数化で 7
    """
    if not np.isfinite(r_m) or r_m <= 0:
        k = min_win
    else:
        k = int(round(float(r_m) / float(px)))
        k = max(min_win, k)
    if odd and (k % 2 == 0):
        k += 1
    return k

def _scale_from_meters(r_m: float, px: float) -> int:
    """
    RS（粗視化）用のブロックスケール（偶奇は気にしない）。最低 1。
    例）0.5m DEM で r_m=5m → round(5/0.5)=10
    """
    s = max(1, int(round(float(r_m) / float(px))))
    return s

def ask_path(prompt, must_exist=True, default=None, allow_empty=False):
    while True:
        s = input(f"{prompt}{' ['+default+']' if default else ''}: ").strip()
        if not s and default: s = default
        if not s:
            if allow_empty: return ""
            print("  入力してください。"); continue
        s = os.path.expanduser(os.path.expandvars(s.strip().strip('"').strip("'")))
        p = Path(s)
        try: p = p.resolve(strict=False)
        except Exception: pass
        if must_exist and not p.exists():
            print(f"  見つかりません: {p}"); continue
        return str(p)

def ask_float(prompt, default=None, minval=None, maxval=None):
    while True:
        s = input(f"{prompt}{' ['+str(default)+']' if default is not None else ''}: ").strip()
        if s=="" and default is not None: return float(default)
        try:
            v=float(s)
            if (minval is not None and v<minval) or (maxval is not None and v>maxval):
                print(f"  {minval}〜{maxval} の範囲で。"); continue
            return v
        except: print("  数値で入力してください。")

def ask_float_list(prompt, default=None, minval=None, maxval=None):
    while True:
        s = input(f"{prompt}{' ['+str(default)+']' if default is not None else ''}: ").strip()
        if not s:
            if default is None:
                print("  値を入力してください。"); continue
            s = str(default)
        try:
            s = s.replace("，", ",").replace("　", " ")
            vals = [float(v) for v in s.split(",") if v.strip()!=""]
            if minval is not None and any(v<minval for v in vals):
                print(f"  {minval}以上で指定してください。"); continue
            if maxval is not None and any(v>maxval for v in vals):
                print(f"  {maxval}以下で指定してください。"); continue
            return vals
        except ValueError:
            print("  数値（カンマ区切り可）で入力してください。")

def ask_int(prompt, default=None, minval=None, maxval=None):
    while True:
        s = input(f"{prompt}{' ['+str(default)+']' if default is not None else ''}: ").strip()
        if s=="" and default is not None: return int(default)
        try:
            v=int(s)
            if (minval is not None and v<minval) or (maxval is not None and v>maxval):
                print(f"  {minval}〜{maxval} の範囲で。"); continue
            return v
        except: print("  整数で入力してください。")

def ask_yesno(prompt, default="y"):
    s = input(f"{prompt} [y/N]" if default.lower()=="n" else f"{prompt} [Y/n]").strip().lower()
    if not s: return default.lower()=="y"
    return s in ("y","yes")

def ask_optional_float(prompt, default=None):
    s = input(f"{prompt}{' ['+str(default)+']' if default is not None else ''}: ").strip()
    if s == "": 
        return default
    try:
        return float(s)
    except:
        print("  数値で入力してください（空=既定）。")
        return default

def ask_bbox(prompt="BBOX (xmin,ymin,xmax,ymax)", default=None):
    while True:
        s = input(f"{prompt}{' ['+default+']' if default else ''}: ").strip()
        if not s and default: s = default
        try:
            xs = [float(v) for v in s.split(",")]
            if len(xs)!=4 or xs[0]>=xs[2] or xs[1]>=xs[3]: raise ValueError
            return tuple(xs)
        except: print("  形式は xmin,ymin,xmax,ymax です。")

# ==== 置き換え: 出力プロファイル ====
def write_like(src, out_path, dtype="float32"):
    meta = src.meta.copy()
    # GeoTIFF 出力を安定化
    meta.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=-9999.0,          # ← NaN ではなく実数に
        compress="DEFLATE",      # ← LZWより堅牢
        predictor=3,             # ← 浮動小数向け（DEFLATEでも有効）
        tiled=True,
        blockxsize=256,
        blockysize=256,
        bigtiff="YES",           # ← 大きくなる可能性に備える
        interleave="band",
    )
    if os.path.exists(out_path):
        os.remove(out_path)
    return rasterio.open(out_path, "w", **meta)

def write_custom(out_path, *, crs, transform, height, width, dtype="float32"):
    """RSDなど、解像度やtransformが変わる出力用のファクトリ。"""
    meta = dict(
        driver="GTiff", count=1, dtype=dtype,
        nodata=-9999.0, compress="DEFLATE", predictor=3,
        tiled=True, blockxsize=256, blockysize=256,
        bigtiff="YES", interleave="band",
        crs=crs, transform=transform,
        height=height, width=width,
    )
    if os.path.exists(out_path):
        os.remove(out_path)
    return rasterio.open(out_path, "w", **meta)

def _centered_geoms(tr: Affine, h: int, w: int, scale: int):
    """
    粗格子サイズと transform（端数は左右上下に均等配分してセンタリング）、
    1m格子（=元ラスタと同一格子）を返す。
      return ((wc, hc, tr_coarse), (w1, h1, tr_1m))
    """
    if scale <= 1:
        return (w, h, tr), (w, h, tr)
    wc = max(1, int(math.floor(w / scale)))
    hc = max(1, int(math.floor(h / scale)))
    rem_x = w - wc * scale
    rem_y = h - hc * scale
    off_x = (rem_x / 2.0) * tr.a
    off_y = (rem_y / 2.0) * tr.e  # 北上ラスタでは tr.e < 0
    tr_coarse = Affine(tr.a * scale, tr.b, tr.c + off_x,
                       tr.d, tr.e * scale, tr.f + off_y)
    return (wc, hc, tr_coarse), (w, h, tr)

# ==== 追加: NaN を NODATA に置換して安全に書く ====
def _safe_write(dst, arr):
    # NaN→-9999 に明示置換してから書く
    a = np.where(np.isfinite(arr), arr.astype(np.float32), np.float32(-9999.0))
    dst.write(a, 1)

# =============== 地形演算（配列） ===============

def local_relief_range(arr, win_pix: int):
    k = int(max(1, win_pix))
    valid = np.isfinite(arr)

    # 局所最大（NaNは -inf 扱い）
    a_maxin = np.where(valid, arr, -np.inf)
    loc_max = maximum_filter(a_maxin, size=k, mode="nearest")

    # 局所最小（NaNは +inf 扱い）
    a_minin = np.where(valid, arr, +np.inf)
    loc_min = minimum_filter(a_minin, size=k, mode="nearest")

    out = (loc_max - loc_min).astype(np.float32)
    out[~valid] = np.nan
    return out
def slope_deg(arr, px):
    gy, gx = np.gradient(arr, px, px)
    return np.degrees(np.arctan(np.hypot(gx, gy)))

def aspect_deg(arr, px):
    gy, gx = np.gradient(arr, px, px)
    return (np.degrees(np.arctan2(gy, gx)) + 360.0) % 360.0  # 0=東

def local_stddev(arr, win_pix):
    valid = np.isfinite(arr)
    a = np.where(valid, arr, 0.0)
    k = int(max(1, win_pix))
    mean  = uniform_filter(a, size=k, mode="nearest")
    mean2 = uniform_filter(a*a, size=k, mode="nearest")
    cnt   = uniform_filter(valid.astype(np.float64), size=k, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean  = np.where(cnt>0, mean/cnt,  np.nan)
        mean2 = np.where(cnt>0, mean2/cnt, np.nan)
        var = mean2 - mean*mean
    var[var<0] = 0.0
    return np.sqrt(var)

def laplacian(arr, px):
    """
    ラプラシアン ∇²z = z_xx + z_yy
    np.gradient ベースで二階差分を取り、RS/SM 間の離散化差を最小化。
    """
    # 1階微分
    zy, zx  = np.gradient(arr, px, px)   # 注意: 第1引数が y 方向, 第2が x 方向
    # 2階微分
    zyy, _  = np.gradient(zy,  px, px)
    _,  zxx = np.gradient(zx,  px, px)
    out = zxx + zyy
    out[~np.isfinite(arr)] = np.nan
    return out

def mean_curvature(arr, px):
    zy, zx  = np.gradient(arr, px, px)
    zyy, _  = np.gradient(zy,  px, px)
    zxy, _  = np.gradient(zx,  px, px)   # 混合2階（zx を y で微分）でOK
    _,  zxx = np.gradient(zx,  px, px)

    denom = 2.0 * np.power(1.0 + zx*zx + zy*zy, 1.5)
    # --- ここを追加 ---
    eps = np.finfo(np.float32).eps
    denom = np.where(denom < eps, eps, denom)
    # ------------------

    num = (1.0 + zy*zy)*zxx - 2.0*zx*zy*zxy + (1.0 + zx*zx)*zyy
    with np.errstate(invalid="ignore", divide="ignore"):
        H = num / denom
    H[~np.isfinite(arr)] = np.nan
    return H

def _apply_valid_mask(arr, win_pix: int):
    """局所窓の半径ぶんだけ四辺を NaN（= NODATA）にする。"""
    rpx = max(0, (int(win_pix) - 1) // 2)   # ← 修正：win_pix=1 → rpx=0
    out = arr.astype(np.float32, copy=True)
    if rpx > 0:
        out[:rpx, :] = np.nan
        out[-rpx:, :] = np.nan
        out[:, :rpx] = np.nan
        out[:, -rpx:] = np.nan
    return out

# =============== スケール別の2方式: resampled / smoothed ===============

def dem_resample_average(arr, transform, crs, scale: int):
    """
    面積平均で scale 倍だけ粗視化して配列と transform を返す。
    端数(W % scale, H % scale)は左右上下に半分ずつ配分（センタリング）。
    """
    if scale <= 1:
        return arr.astype(np.float32), transform

    h, w = arr.shape
    w_coarse = max(1, int(math.floor(w / scale)))
    h_coarse = max(1, int(math.floor(h / scale)))

    # 元ピクセルサイズ（北を上で tr.e は負）
    px_x = float(transform.a)
    px_y = float(transform.e)

    # 余り（右・下に出る端数）を左右上下で均等に割るためのオフセット（高解像px単位）
    rem_x = w - w_coarse * scale            # 例: 1000 - 20*49 = 20
    rem_y = h - h_coarse * scale            # 例: 1000 - 20*49 = 20
    off_x_hi = rem_x / 2.0                  # 例: 10 px → 10 m
    off_y_hi = rem_y / 2.0

    # 粗解像 transform：スケール＆原点をセンタリング分だけずらす
    # 注意: 北上ラスタでは px_y は負。tr.f に px_y*off_y_hi を足すと UL が下へ off_y_hi 分だけ移動。
    transform_coarse = Affine(
        px_x * scale, transform.b, transform.c + px_x * off_x_hi,
        transform.d, px_y * scale, transform.f + px_y * off_y_hi
    )

    coarse = np.full((h_coarse, w_coarse), np.nan, dtype=np.float32)
    reproject(
        source=arr,
        destination=coarse,
        src_transform=transform, src_crs=crs,
        dst_transform=transform_coarse, dst_crs=crs,
        src_nodata=np.nan, dst_nodata=np.nan,
        resampling=Resampling.average,
    )
    return coarse, transform_coarse

def dem_resampled_then_up(arr, transform, crs, r_m:float, px:float):
    """
    RSU の補助: 1m→(平均で)粗視化→元解像度へ補間で戻すための up領域を返す    
    arr: 元DEM配列
    transform: 元DEMのAffine
    crs: 元DEMのCRS (例: src.crs)
    r_m: スケール[m]
    px: 元のピクセルサイズ[m]
    """
    scale = _scale_from_meters(r_m, px)  # 例：0.5mDEMでr=5m→10倍ブロック
    if scale == 1:
        # up: 元サイズの空配列、coarse: 原配列（粗視化なし相当）、transform_coarse: transform のまま
        h, w = arr.shape
        up = np.full((h, w), np.nan, dtype=np.float32)
        return up, arr.astype(np.float32), transform    

    # 粗視化後のサイズ
    h, w = arr.shape
    w_coarse = max(1, int(math.floor(w / scale)))
    h_coarse = max(1, int(math.floor(h / scale)))

    # 粗視化後の transform（ピクセルサイズだけ scale 倍）
    transform_coarse = transform * Affine.scale(scale)

    coarse = np.full((h_coarse, w_coarse), np.nan, dtype=np.float32)

    # 1) 平均で粗視化（area average）
    reproject(
        source=arr,
        destination=coarse,
        src_transform=transform,
        src_crs=crs,
        dst_transform=transform_coarse,
        dst_crs=crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.average,
    )

    # 2) （後段で選択）元解像度にアップサンプル
    up = np.full((h, w), np.nan, dtype=np.float32)
    return up, coarse, transform_coarse  # ← アップは呼び出し側で resampling 指定

def dem_smoothed(arr, r_m: float, px: float):
    """r[m] 相当の移動平均（BOX）で平滑化（NaN対応・元解像度）"""
    k = _winpix_from_meters(r_m, px, min_win=1, odd=True)
    if k == 1:
        return arr.astype(np.float32)
    valid = np.isfinite(arr)
    a = np.where(valid, arr, 0.0).astype(np.float32)
    num = uniform_filter(a, size=k, mode='nearest')
    den = uniform_filter(valid.astype(np.float32), size=k, mode='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(den > 0, num / den, np.nan)
    return out.astype(np.float32)

def dem_smoothed_gauss(arr, sigma_px: float):
    """ガウシアン平滑（NaN対応・元解像度）"""
    sigma = max(0.3, float(sigma_px))
    valid = np.isfinite(arr).astype(np.float32)
    a0 = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    num = gaussian_filter(a0, sigma=sigma, mode="nearest")
    den = gaussian_filter(valid, sigma=sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den>0, num/den, np.nan)
    return out.astype(np.float32)

# =============== 開度（pos/neg）の中核（numba対応） ===============

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

if _HAS_NUMBA:
    @njit(cache=True)
    def _unit_dirs(n_dirs: int):
        dxu = np.empty(n_dirs, dtype=np.float64)
        dyu = np.empty(n_dirs, dtype=np.float64)
        step = math.pi / n_dirs
        for k in range(n_dirs):
            theta = k * step
            dxu[k] = math.cos(theta); dyu[k] = math.sin(theta)
        return dxu, dyu
else:
    def _unit_dirs(n_dirs: int):
        thetas = np.linspace(0.0, math.pi, n_dirs, endpoint=False)
        return np.cos(thetas), np.sin(thetas)

if _HAS_NUMBA:
    @njit(parallel=True, fastmath=True)
    def _openness_pair(dem: np.ndarray, res: float, radius_m: float, n_dirs: int, step_stride: int):
        h, w = dem.shape
        dxu, dyu = _unit_dirs(n_dirs)
        max_step = max(1, int(radius_m / res))
        if step_stride < 1: step_stride = 1
        pos = np.full((h, w), np.nan, dtype=np.float32)
        neg = np.full((h, w), np.nan, dtype=np.float32)
        for y in prange(h):
            for x in range(w):
                z0 = dem[y, x]
                if np.isnan(z0): continue
                acc_up = 0.0; acc_dn = 0.0; cnt = 0
                for k in range(n_dirs):
                    max_alpha_up = -1e9; max_alpha_dn = -1e9
                    dxk, dyk = dxu[k], dyu[k]
                    for s in range(step_stride, max_step + 1, step_stride):
                        ix = int(round(x + dxk * s))
                        iy = int(round(y + dyk * s))
                        if ix < 0 or iy < 0 or ix >= w or iy >= h: break
                        z = dem[iy, ix]
                        if np.isnan(z): continue
                        dist = s * res
                        au = math.atan((z - z0) / dist)
                        ad = math.atan((z0 - z) / dist)
                        if au > max_alpha_up: max_alpha_up = au
                        if ad > max_alpha_dn: max_alpha_dn = ad
                    if max_alpha_up > -1e8: acc_up += max_alpha_up
                    if max_alpha_dn > -1e8: acc_dn += max_alpha_dn
                    cnt += 1
                if cnt > 0:
                    mean_up = (acc_up / cnt) * 180.0 / math.pi
                    mean_dn = (acc_dn / cnt) * 180.0 / math.pi
                    pos[y, x] = 90.0 - mean_up
                    neg[y, x] = 90.0 - mean_dn
        return pos, neg
else:
    def _openness_pair(dem, res, radius_m, n_dirs, step_stride):
        h, w = dem.shape
        dxu, dyu = _unit_dirs(n_dirs)
        max_step = max(1, int(radius_m / res))
        if step_stride < 1:  # ← 追加
            step_stride = 1   # ← 追加
        pos = np.full((h, w), np.nan, dtype=np.float32)
        neg = np.full((h, w), np.nan, dtype=np.float32)
        for y in range(h):
            for x in range(w):
                z0 = dem[y, x]
                if np.isnan(z0): 
                    continue
                acc_up = 0.0; acc_dn = 0.0; cnt = 0
                for k in range(n_dirs):
                    max_alpha_up = -1e9; max_alpha_dn = -1e9
                    dxk, dyk = dxu[k], dyu[k]
                    hit = False  # ← 追加：その方向で1点でも評価できたか
                    for s in range(step_stride, max_step + 1, step_stride):  # ← 変更
                        ix = int(round(x + dxk * s))
                        iy = int(round(y + dyk * s))
                        if ix < 0 or iy < 0 or ix >= w or iy >= h:
                            break
                        z = dem[iy, ix]
                        if np.isnan(z): 
                            continue
                        dist = s * res
                        au = math.atan((z - z0) / dist)
                        ad = math.atan((z0 - z) / dist)
                        if au > max_alpha_up: max_alpha_up = au
                        if ad > max_alpha_dn: max_alpha_dn = ad
                        hit = True
                    if hit:  # ← 追加：ヒットした方向だけカウント
                        if max_alpha_up > -1e8: acc_up += max_alpha_up
                        if max_alpha_dn > -1e8: acc_dn += max_alpha_dn
                        cnt += 1
                if cnt > 0:
                    mean_up = (acc_up / cnt) * 180.0 / math.pi
                    mean_dn = (acc_dn / cnt) * 180.0 / math.pi
                    pos[y, x] = 90.0 - mean_up
                    neg[y, x] = 90.0 - mean_dn
        return pos, neg

# =============== スケール指標の一括出力（16ファイル/スケール） ===============

def run_scale_features_one(
    dem_path: Path, out_dir: Path, r: float, n_dirs: int = 16,
    do_rsu: bool = True, do_rsd: bool = False, do_sm: bool = True, do_res: bool = False,
    rsu_interp: str = "bilinear", sm_kernel: str = "gauss", sm_size_px: float = 1.0,
    valid_only: bool = False):

    dem_path = Path(dem_path)
    out_dir  = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    stem = dem_path.stem   # ← ここで先に決めておく

    def _write_one(src, out_dir, name, data):
        """書いて即解放。窓書きが定義済みなら自動使用。失敗時は無圧縮で再トライ。"""
        op = out_dir / name
        try:
            with write_like(src, str(op), dtype="float32") as dst:
                if 'USE_WINDOWED' in globals() and globals().get('USE_WINDOWED') and '_safe_write_windowed' in globals():
                    _safe_write_windowed(dst, data)
                else:
                    _safe_write(dst, data)
            print(f"[OK] {op}")
        except Exception as e:
            try:
                print(f"[RETRY: uncompressed] {op} :: {e}")
                # 無圧縮で作り直し（タイルは維持）
                with rasterio.open(str(op), "w",
                                   driver="GTiff",
                                   height=data.shape[0],
                                   width=data.shape[1],
                                   count=1,
                                   dtype="float32",
                                   crs=src.crs,
                                   transform=src.transform,
                                   nodata=-9999.0,
                                   tiled=True,
                                   blockxsize=256,
                                   blockysize=256,
                                   bigtiff="YES",
                                   interleave="band") as dst2:
                    if '_safe_write_windowed' in globals():
                        _safe_write_windowed(dst2, data)
                    else:
                        _safe_write(dst2, data)
                print(f"[OK: uncompressed] {op}")
            except Exception as e2:
                print(f"[FAIL] {op} :: {e2}")
        finally:
            del data
            gc.collect()

    with rasterio.open(dem_path) as src:
        arr = src.read(1).astype(np.float64)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)

        tr  = src.transform
        crs = src.crs
        px  = _pixsize_from_transform(tr)
        if r < px:
            print(f"[WARN] r({r}m) < pixel({px}m) → RS系は粗視化にならないため r=px に補正")
            r = px

        rtag    = int(round(float(r)))
        win_pix = _winpix_from_meters(r, px, min_win=1, odd=True)

        if win_pix == 1:
            print(f"[WARN] r={r}m → win_pix=1。relief/stddev は定義上ほぼ0になります。")

        maybe_mask = (lambda a: _apply_valid_mask(a, win_pix)) if valid_only else (lambda a: a)

        # ===== RSD（Down only＝粗解像で出力）/ RSU（Down→Up） =====
        scale = _scale_from_meters(r, px)
        h, w = arr.shape
        (wc, hc, tr_coarse), (w1, h1, tr_1m) = _centered_geoms(tr, h, w, scale)
        px_c = px * float(scale)
        # 粗格子アレイをセンタリングしたtransformで作る
        arr_coarse = np.full((hc, wc), np.nan, dtype=np.float32)
        reproject(
            source=arr, destination=arr_coarse,
            src_transform=tr, src_crs=crs,
            dst_transform=tr_coarse, dst_crs=crs,
            src_nodata=np.nan, dst_nodata=np.nan,
            resampling=Resampling.average,
        )
        # --- RSDだけ有効域が広く見える問題を修正：r/2 相当だけ内側にクロップ ---
        # 1mグリッドでの半径[px] → 粗格子セル数に換算
        rpx_hi = max(0, (int(win_pix) - 1) // 2)
        crop_c = int(math.ceil(rpx_hi / scale))
        if crop_c > 0 and arr_coarse.shape[0] > 2*crop_c and arr_coarse.shape[1] > 2*crop_c:
            y0, y1 = crop_c, arr_coarse.shape[0] - crop_c
            x0, x1 = crop_c, arr_coarse.shape[1] - crop_c
            arr_coarse = arr_coarse[y0:y1, x0:x1]
            # transform をクロップ分だけ前進（粗格子セル単位でOK）
            tr_coarse = tr_coarse * Affine.translation(crop_c, crop_c)
            # 後続の write_custom 用にサイズも更新
            hc, wc = arr_coarse.shape

        # ---- RSD: 粗解像で直接書く（transform/shapeが異なる）
        if do_rsd:
            # relief/stddev は「元解像度で算出→面積平均で粗解像へ縮約」
            # RSD は粗解像側では縁マスクをしない（値の意味は粗格子セル中心）
            maybe_mask_c = (lambda a: a)
            if win_pix > 1:
                # 1) 元解像度で r 窓の指標を計算
                relief_hi = local_relief_range(arr, win_pix).astype(np.float32)
                std_hi    = local_stddev(arr,       win_pix).astype(np.float32)
                # ★ 追加：r に起因する縁はここ（高解像度側）で落とす
                relief_hi = maybe_mask(relief_hi)
                std_hi    = maybe_mask(std_hi)

                # 2) 面積平均で粗解像グリッドへ再投影（値の要約）
                relief_rsd = np.full(arr_coarse.shape, np.nan, dtype=np.float32)
                std_rsd    = np.full(arr_coarse.shape, np.nan, dtype=np.float32)
                reproject(
                    relief_hi, relief_rsd,
                    src_transform=tr, src_crs=crs,
                    dst_transform=tr_coarse, dst_crs=crs,
                    src_nodata=np.nan, dst_nodata=np.nan,
                    resampling=Resampling.average,
                )
                reproject(
                    std_hi, std_rsd,
                    src_transform=tr, src_crs=crs,
                    dst_transform=tr_coarse, dst_crs=crs,
                    src_nodata=np.nan, dst_nodata=np.nan,
                    resampling=Resampling.average,
                )
                if RSD_OUTPUT_GRID == "1m":
                    # 粗 → 1m に載せ替えてから出力（SM/RES/RSU と同じ格子に統一）
                    relief_rsd_1m = np.full((h1, w1), np.nan, dtype=np.float32)
                    std_rsd_1m    = np.full((h1, w1), np.nan, dtype=np.float32)
                    reproject(relief_rsd, relief_rsd_1m,
                              src_transform=tr_coarse, src_crs=crs,
                              dst_transform=tr_1m,    dst_crs=crs,
                              src_nodata=np.nan, dst_nodata=np.nan,
                              resampling=Resampling.bilinear)
                    reproject(std_rsd, std_rsd_1m,
                              src_transform=tr_coarse, src_crs=crs,
                              dst_transform=tr_1m,    dst_crs=crs,
                              src_nodata=np.nan, dst_nodata=np.nan,
                              resampling=Resampling.bilinear)
                    with write_custom(str(out_dir/f"{stem}_relief_r{rtag}m_rsd.tif"),
                                      crs=crs, transform=tr_1m, height=h1, width=w1) as dst:
                        _safe_write_windowed(dst, maybe_mask(relief_rsd_1m))  # 1m側の縁規則
                    with write_custom(str(out_dir/f"{stem}_stddev_r{rtag}m_rsd.tif"),
                                      crs=crs, transform=tr_1m, height=h1, width=w1) as dst:
                        _safe_write_windowed(dst, maybe_mask(std_rsd_1m))
                else:
                    # 従来通り：粗グリッドに直接書く
                    with write_custom(str(out_dir/f"{stem}_relief_r{rtag}m_rsd.tif"),
                                      crs=crs, transform=tr_coarse,
                                      height=arr_coarse.shape[0], width=arr_coarse.shape[1]) as dst:
                        _safe_write_windowed(dst, maybe_mask_c(relief_rsd))
                    with write_custom(str(out_dir/f"{stem}_stddev_r{rtag}m_rsd.tif"),
                                      crs=crs, transform=tr_coarse,
                                      height=arr_coarse.shape[0], width=arr_coarse.shape[1]) as dst:
                        _safe_write_windowed(dst, maybe_mask_c(std_rsd))
                # 一時配列を解放
                try:
                    del relief_hi, std_hi, relief_rsd, std_rsd
                except:
                    pass

            # 1次・2次微分系（粗解像“r世界”での値）
            # --- slope ---
            slope_rsd = slope_deg(arr_coarse, px_c).astype(np.float32)
            if RSD_OUTPUT_GRID == "1m":
                slope_rsd_1m = np.full((h1, w1), np.nan, dtype=np.float32)
                reproject(slope_rsd, slope_rsd_1m,
                    src_transform=tr_coarse, src_crs=crs,
                    dst_transform=tr_1m,    dst_crs=crs,
                    src_nodata=np.nan, dst_nodata=np.nan,
                    resampling=Resampling.bilinear)
                with write_custom(str(out_dir/f"{stem}_slope_deg_r{rtag}m_rsd.tif"),
                        crs=crs, transform=tr_1m, height=h1, width=w1) as dst:
                    _safe_write_windowed(dst, maybe_mask(slope_rsd_1m))
            else:
                with write_custom(str(out_dir/f"{stem}_slope_deg_r{rtag}m_rsd.tif"),
                                    crs=crs, transform=tr_coarse,
                                    height=arr_coarse.shape[0], width=arr_coarse.shape[1]) as dst:
                    _safe_write_windowed(dst, maybe_mask_c(slope_rsd))

            # --- aspect ---
            aspect_rsd = aspect_deg(arr_coarse, px_c).astype(np.float32)
            if RSD_OUTPUT_GRID == "1m":
                aspect_rsd_1m = np.full((h1, w1), np.nan, dtype=np.float32)
                reproject(aspect_rsd, aspect_rsd_1m,
                          src_transform=tr_coarse, src_crs=crs,
                          dst_transform=tr_1m,    dst_crs=crs,
                          src_nodata=np.nan, dst_nodata=np.nan,
                          resampling=Resampling.bilinear)
                with write_custom(str(out_dir/f"{stem}_aspect_deg_r{rtag}m_rsd.tif"),
                                  crs=crs, transform=tr_1m, height=h1, width=w1) as dst:
                    _safe_write_windowed(dst, maybe_mask(aspect_rsd_1m))
            else:
                with write_custom(str(out_dir/f"{stem}_aspect_deg_r{rtag}m_rsd.tif"),
                                  crs=crs, transform=tr_coarse,
                                  height=arr_coarse.shape[0], width=arr_coarse.shape[1]) as dst:
                    _safe_write_windowed(dst, maybe_mask_c(aspect_rsd))

            # --- laplacian / meancurv ---
            lap_rsd   = laplacian(arr_coarse, px_c).astype(np.float32)
            meanc_rsd = mean_curvature(arr_coarse, px_c).astype(np.float32)
            if RSD_OUTPUT_GRID == "1m":
                for nm, arr_out in [("laplacian", lap_rsd), ("meancurv", meanc_rsd)]:
                    arr_1m = np.full((h1, w1), np.nan, dtype=np.float32)
                    reproject(arr_out, arr_1m,
                              src_transform=tr_coarse, src_crs=crs,
                              dst_transform=tr_1m,    dst_crs=crs,
                              src_nodata=np.nan, dst_nodata=np.nan,
                              resampling=Resampling.bilinear)
                    with write_custom(str(out_dir/f"{stem}_{nm}_r{rtag}m_rsd.tif"),
                                      crs=crs, transform=tr_1m, height=h1, width=w1) as dst:
                        _safe_write_windowed(dst, maybe_mask(arr_1m))
            else:
                for nm, arr_out in [("laplacian", lap_rsd), ("meancurv", meanc_rsd)]:
                    with write_custom(str(out_dir/f"{stem}_{nm}_r{rtag}m_rsd.tif"),
                                      crs=crs, transform=tr_coarse,
                                      height=arr_coarse.shape[0], width=arr_coarse.shape[1]) as dst:
                        _safe_write_windowed(dst, maybe_mask_c(arr_out))

            # 開度：粗解像DEM上で半径rを使って計算
            dem_in_c = np.ascontiguousarray(arr_coarse, dtype=np.float32)
            pos_c, neg_c = _openness_pair(dem_in_c, float(px_c), float(r), int(n_dirs), int(OPENNESS_STEP_STRIDE))
            if RSD_OUTPUT_GRID == "1m":
                for nm, arr_out in [("openness_pos", pos_c), ("openness_neg", neg_c)]:
                    arr_1m = np.full((h1, w1), np.nan, dtype=np.float32)
                    reproject(arr_out, arr_1m,
                              src_transform=tr_coarse, src_crs=crs,
                              dst_transform=tr_1m,    dst_crs=crs,
                              src_nodata=np.nan, dst_nodata=np.nan,
                              resampling=Resampling.bilinear)
                    with write_custom(str(out_dir/f"{stem}_{nm}_r{rtag}m_rsd.tif"),
                                      crs=crs, transform=tr_1m, height=h1, width=w1) as dst:
                        _safe_write_windowed(dst, maybe_mask(arr_1m))
            else:
                for nm, arr_out in [("openness_pos", pos_c), ("openness_neg", neg_c)]:
                    with write_custom(str(out_dir/f"{stem}_{nm}_r{rtag}m_rsd.tif"),
                                      crs=crs, transform=tr_coarse,
                                      height=arr_coarse.shape[0], width=arr_coarse.shape[1]) as dst:
                        _safe_write_windowed(dst, maybe_mask_c(arr_out))

            # --- 任意：RSDで使った開度の中間結果を解放 ---
            for a in ("pos_c","neg_c"):
                if a in locals():
                    try:
                        del locals()[a]
                    except:
                        pass
            gc.collect()

        # ---- RSU: 1mへ復元（bilinear/cubic 選択）
        if do_rsu:
            # センタリングした1mグリッドへアップサンプル
            up = np.full((h1, w1), np.nan, dtype=np.float32)
            resmap = {"bilinear": Resampling.bilinear, "cubic": Resampling.cubic}
            reproject(source=arr_coarse, destination=up,
                      src_transform=tr_coarse, src_crs=crs,
                      dst_transform=tr_1m,  dst_crs=crs,
                      src_nodata=np.nan, dst_nodata=np.nan,
                      resampling=resmap.get(rsu_interp, Resampling.bilinear))
            arr_rsu = up  # 1m上の低周波DEM
            # 任意: 参照を減らす（arr_rsuは残る）
            try: del up
            except: pass

            # relief/stddev（win_pixは元px基準）
            if win_pix > 1:
                relief_rsu = local_relief_range(arr_rsu, win_pix)
                with write_custom(str(out_dir/f"{stem}_relief_r{rtag}m_rsu_{rsu_interp}.tif"),
                                  crs=crs, transform=tr_1m, height=h1, width=w1) as dst:
                    _safe_write_windowed(dst, maybe_mask(relief_rsu))
                std_rsu = local_stddev(arr_rsu, win_pix).astype(np.float32)
                with write_custom(str(out_dir/f"{stem}_stddev_r{rtag}m_rsu_{rsu_interp}.tif"),
                                  crs=crs, transform=tr_1m, height=h1, width=w1) as dst:
                    _safe_write_windowed(dst, maybe_mask(std_rsu))
            # slope/aspect は1m上で
            slope_rsu = slope_deg(arr_rsu, px).astype(np.float32)
            with write_custom(str(out_dir/f"{stem}_slope_deg_r{rtag}m_rsu_{rsu_interp}.tif"),
                              crs=crs, transform=tr_1m, height=h1, width=w1) as dst:
                _safe_write_windowed(dst, maybe_mask(slope_rsu))
            aspect_rsu = aspect_deg(arr_rsu, px).astype(np.float32)
            with write_custom(str(out_dir/f"{stem}_aspect_deg_r{rtag}m_rsu_{rsu_interp}.tif"),
                              crs=crs, transform=tr_1m, height=h1, width=w1) as dst:
                _safe_write_windowed(dst, maybe_mask(aspect_rsu))
            # 2階微分は粗面→1mへ再投影済（上で済ませてもOK）
            lap_c = laplacian(arr_coarse, px_c).astype(np.float32)
            meanc_c = mean_curvature(arr_coarse, px_c).astype(np.float32)
            lap_rsu   = np.full((h1, w1), np.nan, dtype=np.float32)
            meanc_rsu = np.full((h1, w1), np.nan, dtype=np.float32)
            reproject(lap_c, lap_rsu, src_transform=tr_coarse, src_crs=crs,
                      dst_transform=tr_1m, dst_crs=crs, src_nodata=np.nan, dst_nodata=np.nan,
                      resampling=resmap.get(rsu_interp, Resampling.bilinear))
            reproject(meanc_c, meanc_rsu, src_transform=tr_coarse, src_crs=crs,
                      dst_transform=tr_1m, dst_crs=crs, src_nodata=np.nan, dst_nodata=np.nan,
                      resampling=resmap.get(rsu_interp, Resampling.bilinear))
            try:
                del lap_c, meanc_c
            except:
                pass
            with write_custom(str(out_dir/f"{stem}_laplacian_r{rtag}m_rsu_{rsu_interp}.tif"),
                              crs=crs, transform=tr_1m, height=h1, width=w1) as dst:
                _safe_write_windowed(dst, maybe_mask(lap_rsu))
            with write_custom(str(out_dir/f"{stem}_meancurv_r{rtag}m_rsu_{rsu_interp}.tif"),
                              crs=crs, transform=tr_1m, height=h1, width=w1) as dst:
                _safe_write_windowed(dst, maybe_mask(meanc_rsu))
            # 開度：1m上のRSUで OK
            dem_in = np.ascontiguousarray(arr_rsu, dtype=np.float32)
            pos_rsu, neg_rsu = _openness_pair(dem_in, float(px), float(r), int(n_dirs), int(OPENNESS_STEP_STRIDE))
            with write_custom(str(out_dir/f"{stem}_openness_pos_r{rtag}m_rsu_{rsu_interp}.tif"),
                              crs=crs, transform=tr_1m, height=h1, width=w1) as dst:
                _safe_write_windowed(dst, maybe_mask(pos_rsu))
            with write_custom(str(out_dir/f"{stem}_openness_neg_r{rtag}m_rsu_{rsu_interp}.tif"),
                              crs=crs, transform=tr_1m, height=h1, width=w1) as dst:
                _safe_write_windowed(dst, maybe_mask(neg_rsu))
            # --- 任意：RSU関連の中間結果（RESを使わないなら解放） ---
            if not do_res:
                for a in ("arr_rsu","lap_rsu","meanc_rsu","pos_rsu","neg_rsu"):
                    if a in locals():
                        try:
                            del locals()[a]
                        except:
                            pass
                gc.collect()

        # ===== SM（box/gauss 切替） =====
        if do_sm:
            if sm_kernel == "box":
                arr_sm = dem_smoothed(arr, r_m=r, px=px)  # win=~r/px
                sm_tag = f"sm_box_w{_winpix_from_meters(r, px)}px"
            else:
                arr_sm = dem_smoothed_gauss(arr, sigma_px=sm_size_px)
                sm_tag = f"sm_gauss_sigma{sm_size_px:.2f}px"

            if win_pix > 1:
                relief_sm = local_relief_range(arr_sm, win_pix)
                _write_one(src, out_dir, f"{stem}_relief_r{rtag}m_{sm_tag}.tif", maybe_mask(relief_sm))
                std_sm = local_stddev(arr_sm, win_pix).astype(np.float32)
                _write_one(src, out_dir, f"{stem}_stddev_r{rtag}m_{sm_tag}.tif", maybe_mask(std_sm))
            else:
                print("[SKIP] SM: win_pix=1 のため relief/stddev をスキップ")

            slope_sm  = slope_deg(arr_sm, px).astype(np.float32)
            def _to_center_grid(a: np.ndarray, resampling=Resampling.bilinear):
                out = np.full((h1, w1), np.nan, dtype=np.float32)
                reproject(a, out,
                          src_transform=tr, src_crs=crs,
                          dst_transform=tr_1m, dst_crs=crs,
                          src_nodata=np.nan, dst_nodata=np.nan,
                          resampling=resampling)
                return out
            slope_sm_c = _to_center_grid(slope_sm)
            with write_custom(str(out_dir/f"{stem}_slope_deg_r{rtag}m_{sm_tag}.tif"),
                              crs=crs, transform=tr_1m, height=h1, width=w1) as dst:
                _safe_write_windowed(dst, maybe_mask(slope_sm_c))
            aspect_sm = aspect_deg(arr_sm, px).astype(np.float32)
            aspect_sm_c = _to_center_grid(aspect_sm)
            with write_custom(str(out_dir/f"{stem}_aspect_deg_r{rtag}m_{sm_tag}.tif"),
                              crs=crs, transform=tr_1m, height=h1, width=w1) as dst:
                _safe_write_windowed(dst, maybe_mask(aspect_sm_c))
            lap_sm    = laplacian(arr_sm, px).astype(np.float32)
            lap_sm_c = _to_center_grid(lap_sm)
            with write_custom(str(out_dir/f"{stem}_laplacian_r{rtag}m_{sm_tag}.tif"),
                              crs=crs, transform=tr_1m, height=h1, width=w1) as dst:
                _safe_write_windowed(dst, maybe_mask(lap_sm_c))
            meanc_sm  = mean_curvature(arr_sm, px).astype(np.float32)
            meanc_sm_c = _to_center_grid(meanc_sm)
            with write_custom(str(out_dir/f"{stem}_meancurv_r{rtag}m_{sm_tag}.tif"),
                              crs=crs, transform=tr_1m, height=h1, width=w1) as dst:
                _safe_write_windowed(dst, maybe_mask(meanc_sm_c))
            dem_in = np.ascontiguousarray(arr_sm, dtype=np.float32)
            pos_sm, neg_sm = _openness_pair(dem_in, float(px), float(r), int(n_dirs), int(OPENNESS_STEP_STRIDE))
            pos_sm_c = _to_center_grid(pos_sm)
            neg_sm_c = _to_center_grid(neg_sm)
            with write_custom(str(out_dir/f"{stem}_openness_pos_r{rtag}m_{sm_tag}.tif"),
                              crs=crs, transform=tr_1m, height=h1, width=w1) as dst:
                _safe_write_windowed(dst, maybe_mask(pos_sm_c))
            with write_custom(str(out_dir/f"{stem}_openness_neg_r{rtag}m_{sm_tag}.tif"),
                              crs=crs, transform=tr_1m, height=h1, width=w1) as dst:
                _safe_write_windowed(dst, maybe_mask(neg_sm_c))
            del arr_sm

            # --- 任意：SMの開度結果は再利用しないなら解放 ---
            for a in ("pos_sm","neg_sm"):
                if a in locals():
                    try:
                        del locals()[a]
                    except:
                        pass
            gc.collect()

        # ===== RES（High-pass：DEM − LowPass_r） =====
        if do_res:
            # 基準面の選択：RES_BASE が空→既定 rsu。sm_gauss を選ぶと格子縞を抑制。
            base_mode = (RES_BASE or "rsu").lower()
            if base_mode == "sm_gauss":
                if RES_GAUSS_SIGMA_PX.strip():
                    sigma_px = float(RES_GAUSS_SIGMA_PX)
                else:
                    # 自動：r/3 px（r[m]→px換算してから）
                    sigma_px = max(0.3, float(r) / float(px) / 3.0)
                base = dem_smoothed_gauss(arr, sigma_px=sigma_px).astype(np.float32)
            else:
                # 従来通り RSU を基準（粗視化→1mへ復元）
                base = locals().get("arr_rsu", None)
                if base is None:
                    up_tmp = np.full(arr.shape, np.nan, dtype=np.float32)
                    reproject(arr_coarse, up_tmp, src_transform=tr_coarse, src_crs=crs,
                              dst_transform=tr_1m, dst_crs=crs,
                              src_nodata=np.nan, dst_nodata=np.nan,
                              resampling=Resampling.bilinear)
                    base = up_tmp

            def _angle_residual_deg(a_raw, a_base):
                d = (a_raw - a_base + 180.0) % 360.0 - 180.0
                d[~np.isfinite(a_raw) | ~np.isfinite(a_base)] = np.nan
                return d.astype(np.float32)

            if not RES_ALL_METRICS:
                # 互換モード：従来の3枚（dem/slope/relief）
                slope_raw  = slope_deg(arr,  px).astype(np.float32)
                slope_base = slope_deg(base, px).astype(np.float32)
                res_slope  = (slope_raw - slope_base).astype(np.float32)
                _write_one(src, out_dir, f"{stem}_slope_deg_r{rtag}m_residual.tif",   maybe_mask(res_slope))
                if win_pix > 1:
                    relief_raw  = local_relief_range(arr,  win_pix).astype(np.float32)
                    relief_base = local_relief_range(base, win_pix).astype(np.float32)
                    res_relief  = (relief_raw - relief_base).astype(np.float32)
                    _write_one(src, out_dir, f"{stem}_relief_r{rtag}m_residual.tif", maybe_mask(res_relief))
            else:
                # 32枚体制：RESも8指標すべて
                if win_pix > 1:
                    relief_raw  = local_relief_range(arr,  win_pix).astype(np.float32)
                    relief_base = local_relief_range(base, win_pix).astype(np.float32)
                    _write_one(src, out_dir, f"{stem}_relief_r{rtag}m_residual.tif",
                               maybe_mask((relief_raw - relief_base).astype(np.float32)))
                    std_raw  = local_stddev(arr,  win_pix).astype(np.float32)
                    std_base = local_stddev(base, win_pix).astype(np.float32)
                    _write_one(src, out_dir, f"{stem}_stddev_r{rtag}m_residual.tif",
                               maybe_mask((std_raw - std_base).astype(np.float32)))

                slope_raw  = slope_deg(arr,  px).astype(np.float32)
                slope_base = slope_deg(base, px).astype(np.float32)
                _write_one(src, out_dir, f"{stem}_slope_deg_r{rtag}m_residual.tif",
                           maybe_mask((slope_raw - slope_base).astype(np.float32)))

                aspect_raw  = aspect_deg(arr,  px).astype(np.float32)
                aspect_base = aspect_deg(base, px).astype(np.float32)
                _write_one(src, out_dir, f"{stem}_aspect_deg_r{rtag}m_residual.tif",
                           maybe_mask(_angle_residual_deg(aspect_raw, aspect_base)))

                lap_raw  = laplacian(arr,  px).astype(np.float32)
                lap_base = laplacian(base, px).astype(np.float32)
                _write_one(src, out_dir, f"{stem}_laplacian_r{rtag}m_residual.tif",
                           maybe_mask((lap_raw - lap_base).astype(np.float32)))

                meanc_raw  = mean_curvature(arr,  px).astype(np.float32)
                meanc_base = mean_curvature(base, px).astype(np.float32)
                _write_one(src, out_dir, f"{stem}_meancurv_r{rtag}m_residual.tif",
                           maybe_mask((meanc_raw - meanc_base).astype(np.float32)))

                # 開度（重い）—必要なら OPENNESS_STEP_STRIDE=4 等で軽量化
                dem_in_raw  = np.ascontiguousarray(arr,  dtype=np.float32)
                dem_in_base = np.ascontiguousarray(base, dtype=np.float32)
                pos_raw, neg_raw   = _openness_pair(dem_in_raw,  float(px), float(r), int(n_dirs), int(OPENNESS_STEP_STRIDE))
                pos_base, neg_base = _openness_pair(dem_in_base, float(px), float(r), int(n_dirs), int(OPENNESS_STEP_STRIDE))
                _write_one(src, out_dir, f"{stem}_openness_pos_r{rtag}m_residual.tif",
                           maybe_mask((pos_raw - pos_base).astype(np.float32)))
                _write_one(src, out_dir, f"{stem}_openness_neg_r{rtag}m_residual.tif",
                           maybe_mask((neg_raw - neg_base).astype(np.float32)))

            # 後片付け
            for a in ("base","up_tmp","slope_raw","slope_base",
                      "relief_raw","relief_base","std_raw","std_base",
                      "aspect_raw","aspect_base","lap_raw","lap_base",
                      "meanc_raw","meanc_base","pos_raw","neg_raw","pos_base","neg_base"):
                if a in locals():
                    try: del locals()[a]
                    except: pass
            gc.collect()

            # --- メモリ掃除：大きめ配列を明示破棄 ---
            for a in ("arr_rsu","lap_rsu","meanc_rsu","pos_rsu","neg_rsu","pos_c","neg_c"):
                if a in locals():
                    try:
                        del locals()[a]
                    except:
                        pass
            gc.collect()

            # 任意: RES用の一時も解放
            for a in ("base","up_tmp"):
                if a in locals():
                    try: del locals()[a]
                    except: pass
            gc.collect()

        # 粗面も解放（RES後は不要）
        try:
            del arr_coarse
        except NameError:
            pass
        gc.collect()

# =============== フロント：特徴量生成（単発 or フォルダ一括） ===============

def run_make_features():
    global RES_BASE, RES_GAUSS_SIGMA_PX
    print("[INFO] スケールごとに rsu/rsd/sm/res を出せます（標高/MPIは出力しません）。")
    dem_or_dir = ask_path("DEM(GeoTIFF) のパス（またはフォルダ）")

    produced = []

    if Path(dem_or_dir).is_dir():
        root = Path(dem_or_dir)
        dem_list = sorted(glob.glob(str(root / "**" / "*.tif"), recursive=True))
        if not dem_list:
            print("[ERROR] 指定フォルダに .tif が見つかりません。"); return []
        print(f"[INFO] 検出された GeoTIFF: {len(dem_list)} 件")

        out_root = ask_path("出力フォルダ（空= 各DEMの隣に feat_out）", must_exist=False, default="")
        if out_root: Path(out_root).mkdir(parents=True, exist_ok=True)

        scales = ask_float_list("スケール r[m]（カンマ区切り。例: 10,25,50）", default="25", minval=1.0)
        n_dirs = ask_int("開度の方向数 n_dirs", default=16, minval=4, maxval=180)

        use_rsu = ask_yesno("RSU（粗視化→1mへ復元）を出力しますか？", default="y")
        use_rsd = ask_yesno("RSD（粗視化のみ＝粗解像）を出力しますか？", default="y")
        use_sm  = ask_yesno("SM（1mのまま平滑）を出力しますか？", default="y")
        use_res = ask_yesno("RES（ハイパス：DEM−LowPass_r）を出力しますか？", default="y")
        rsu_interp = (input("RSUの補間（bilinear/cubic）[bilinear]: ").strip().lower() or "bilinear")
        if rsu_interp not in ("bilinear","cubic"): rsu_interp="bilinear"
        sm_kernel  = (input("SMカーネル（box/gauss）[gauss]: ").strip().lower() or "gauss")
        if sm_kernel not in ("box","gauss"): sm_kernel="gauss"
        # SMサイズ：px指定（推奨）。m指定にしたい場合は追って拡張可
        sm_size_px = float(input("SMサイズ（box=窓幅px / gauss=σpx）[1.0]: ").strip() or "1.0")
        valid_only = ask_yesno("端を欠けさせて有効領域のみを書き出しますか？", default="n")

        # --- RES の基準面を対話で選ぶ（環境変数未指定のときだけ） ---
        if use_res and not RES_BASE:
            if ask_yesno("RESのブロック縞を抑える（基準=ガウシアンSM）にしますか？", default="y"):
                RES_BASE = "sm_gauss"
                # σ[px] は空=自動（r/px/3 を後で算出）
                sig = ask_optional_float("  ガウシアンσ[px]（空=自動 r/3）", default=None)
                RES_GAUSS_SIGMA_PX = "" if sig is None else str(sig)
            else:
                RES_BASE = "rsu"  # 従来通り

        for i, dem in enumerate(dem_list, 1):
            dem = Path(dem)
            out_dir = (Path(out_root)/dem.stem) if out_root else (dem.parent/"feat_out")
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n=== [{i}/{len(dem_list)}] {dem.name} → {out_dir} ===")
            for r in scales:
                run_scale_features_one(
                    dem, out_dir, float(r), n_dirs=n_dirs,
                    do_rsu=use_rsu, do_rsd=use_rsd, do_sm=use_sm, do_res=use_res,
                    rsu_interp=rsu_interp, sm_kernel=sm_kernel, sm_size_px=sm_size_px,
                    valid_only=valid_only
                )
                produced.extend(str(p) for p in Path(out_dir).glob(f"*r{int(round(r))}m_*.tif"))

        print("\n[OK] 全DEM・全スケールの処理が完了しました。")
        return [(Path(p).stem, p) for p in sorted(set(produced))]

    else:
        dem = Path(dem_or_dir)
        out_dir = ask_path("出力フォルダ", must_exist=False, default=str(dem.parent/"feat_out"))
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        scales = ask_float_list("スケール r[m]（カンマ区切り。例: 10,25,50）", default="25", minval=1.0)
        n_dirs = ask_int("開度の方向数 n_dirs", default=16, minval=4, maxval=180)

        use_rsu = ask_yesno("RSU（粗視化→1mへ復元）を出力しますか？", default="y")
        use_rsd = ask_yesno("RSD（粗視化のみ＝粗解像）を出力しますか？", default="y")
        use_sm  = ask_yesno("SM（1mのまま平滑）を出力しますか？", default="y")
        use_res = ask_yesno("RES（ハイパス：DEM−LowPass_r）を出力しますか？", default="y")
        rsu_interp = (input("RSUの補間（bilinear/cubic）[bilinear]: ").strip().lower() or "bilinear")
        if rsu_interp not in ("bilinear","cubic"): rsu_interp="bilinear"
        sm_kernel  = (input("SMカーネル（box/gauss）[gauss]: ").strip().lower() or "gauss")
        if sm_kernel not in ("box","gauss"): sm_kernel="gauss"
        sm_size_px = float(input("SMサイズ（box=窓幅px / gauss=σpx）[1.0]: ").strip() or "1.0")

        valid_only = ask_yesno("端を欠けさせて有効領域のみを書き出しますか？", default="n")


        # --- RES の基準面を対話で選ぶ（環境変数未指定のときだけ） ---
        if use_res and not RES_BASE:
            if ask_yesno("RESのブロック縞を抑える（基準=ガウシアンSM）にしますか？", default="y"):
                RES_BASE = "sm_gauss"
                sig = ask_optional_float("  ガウシアンσ[px]（空=自動 r/3）", default=None)
                RES_GAUSS_SIGMA_PX = "" if sig is None else str(sig)
            else:
                RES_BASE = "rsu"  # 従来通り

        for r in scales:
            run_scale_features_one(
                dem, out_dir, float(r), n_dirs=n_dirs,
                do_rsu=use_rsu, do_rsd=use_rsd, do_sm=use_sm, do_res=use_res,
                rsu_interp=rsu_interp, sm_kernel=sm_kernel, sm_size_px=sm_size_px,
                valid_only=valid_only
            )

        produced = sorted(str(p) for p in Path(out_dir).glob("*.tif"))
        return [(Path(p).stem, p) for p in produced]

# =============== 以下は学習データ作成（あなたの既存ロジックを踏襲） ===============

def grid_centers(bbox, res):
    xmin,ymin,xmax,ymax = bbox
    xs = np.arange(xmin + res/2, xmax, res, dtype=float)
    ys = np.arange(ymin + res/2, ymax, res, dtype=float)
    if xs.size==0 or ys.size==0: return np.empty((0,2))
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])

def intersect_bounds(paths):
    first=True
    for p in paths:
        if not p: continue
        with rasterio.open(p) as src:
            b = src.bounds
            if first:
                xmin,ymin,xmax,ymax = b.left,b.bottom,b.right,b.top
                crs0 = src.crs; first=False
            else:
                if src.crs != crs0:
                    raise ValueError("共通BBOXは同一CRSのみ（ズレはVRTで吸収推奨）")
                xmin=max(xmin,b.left); ymin=max(ymin,b.bottom)
                xmax=min(xmax,b.right); ymax=min(ymax,b.top)
    if first or (xmax<=xmin) or (ymax<=ymin):
        raise ValueError("共通交差範囲なし")
    return (xmin,ymin,xmax,ymax)

def sample_rasters_chunked(pts_xy, rasters, target_epsg, batch=500_000):
    from rasterio.vrt import WarpedVRT
    out = {name: np.empty(len(pts_xy), dtype=float) for name,_ in rasters}
    x = pts_xy[:,0]; y = pts_xy[:,1]
    for name, path in rasters:
        with rasterio.open(path) as src:
            with WarpedVRT(src, crs=f"EPSG:{target_epsg}",
                           resampling=Resampling.nearest,
                           src_nodata=src.nodata, dst_nodata=np.nan) as vrt:
                for i in tqdm(range(0, len(x), batch), desc=f"  サンプル: {name}", unit="pt"):
                    j = min(i+batch, len(x))
                    vals = np.fromiter((v[0] for v in vrt.sample(zip(x[i:j], y[i:j]))),
                                       dtype=float, count=(j-i))
                    nd = vrt.nodata
                    if nd is not None:
                        if isinstance(nd, float) and np.isnan(nd):
                            vals = np.where(np.isnan(vals), np.nan, vals)
                        else:
                            vals = np.where(vals==nd, np.nan, vals)
                    out[name][i:j] = vals
    return out

def _parse_col_selection(sel: str, cols_all):
    if not sel.strip(): return list(cols_all)
    out = []
    tokens = [t.strip() for t in sel.split(",") if t.strip()]
    for t in tokens:
        if "-" in t and t.replace("-", "").replace(" ", "").isdigit():
            a, b = [int(x) for x in t.split("-")]
            a, b = min(a,b), max(a,b)
            for i in range(a,b+1):
                if 0<=i<len(cols_all): out.append(cols_all[i])
        elif t.isdigit():
            i=int(t); 
            if 0<=i<len(cols_all): out.append(cols_all[i])
        else:
            cand=[c for c in cols_all if c.lower()==t.lower()]
            if cand: out.append(cand[0])
    seen=set(); uniq=[]
    for c in out:
        if c not in seen: uniq.append(c); seen.add(c)
    return uniq

def attach_polygon_attrs(df, epsg, poly_path, layer=None, attr_cols=None,
                         predicate="within", out_prefix=""):
    gdf_pts = gpd.GeoDataFrame(df.copy(),
        geometry=gpd.points_from_xy(df["x"], df["y"]),
        crs=f"EPSG:{epsg}")
    poly = gpd.read_file(poly_path, layer=layer) if layer else gpd.read_file(poly_path)
    if poly.crs is None:
        raise ValueError("ポリゴンのCRSが未設定です。")
    if str(poly.crs) != str(gdf_pts.crs):
        poly = poly.to_crs(gdf_pts.crs)
    keep_cols = [c for c in poly.columns if c != "geometry"]
    if attr_cols:
        want={c.lower() for c in attr_cols}
        keep_cols=[c for c in keep_cols if c.lower() in want]
        if not keep_cols:
            print("[WARN] 指定属性が見つかりません。")
    try:
        sj = gpd.sjoin(gdf_pts, poly[keep_cols+["geometry"]], how="left", predicate=predicate)
    except TypeError:
        sj = gpd.sjoin(gdf_pts, poly[keep_cols+["geometry"]], how="left", op=predicate)
    sj_first = sj.groupby(level=0).first()
    for c in keep_cols:
        new_name = f"{out_prefix}{c}" if out_prefix else c
        df[new_name] = sj_first[c].reindex(df.index)
    return df

def run_export_points(feature_paths=None):
    rasters = []
    preset_poly_paths = []  # 追加：ラスタ入力と同タイミングで集めるポリゴン候補
    if feature_paths:
        if isinstance(feature_paths, dict):
            for name, p in feature_paths.items(): rasters.append((name, str(p)))
        elif isinstance(feature_paths, (list, tuple)):
            for item in feature_paths:
                if isinstance(item, (list, tuple)) and len(item)==2:
                    rasters.append((str(item[0]), str(item[1])))
                else:
                    p=str(item); rasters.append((Path(p).stem, p))
        else:
            p=str(feature_paths); rasters.append((Path(p).stem, p))

    print("\n連続値ラスタを追加（空で終了）:")
    while True:
        raw = input('  ラスターのパス: ')
        p = raw.strip().strip('"')
        norm = os.path.normpath(p)
        if not p: break
        if os.path.isdir(norm):
            found = sorted(glob.glob(os.path.join(norm,'**','*.tif'), recursive=True) +
                           glob.glob(os.path.join(norm,'**','*.tiff'), recursive=True))
            if not found:
                print('   [WARN] フォルダ内に .tif/.tiff が見つかりません:', norm); continue
            print(f'   [INFO] {len(found)} 個の GeoTIFF を検出')
            seen = {}
            for tif in found:
                base = Path(tif).stem
                if base in seen:
                    seen[base]+=1; name=f"{base}_{seen[base]}"
                else:
                    seen[base]=0; name=base
                rasters.append((name, tif))
            continue
        if os.path.isfile(norm) and norm.lower().endswith(('.tif','.tiff')):
            name = input('  列名（例: slope_deg_r25m_sm）: ').strip() or Path(norm).stem
            rasters.append((name, norm)); continue
        print('   [WARN] tif/tiff でもフォルダでもありません。')

    if not rasters:
        print('[ERROR] ラスタが1つも指定されていません。'); return

    # 追加：ラスタと同じタイミングでポリゴン候補も登録
    print("\nポリゴンラベルを追加（空で終了）:")
    print("  ※ フォルダを指定すると .gpkg/.geojson/.json/.shp を再帰で収集します")
    while True:
        rawp = input('  ポリゴンのパス: ').strip().strip('"')
        if not rawp:
            break
        pnorm = os.path.normpath(rawp)
        if os.path.isdir(pnorm):
            foundp = sorted(
                glob.glob(os.path.join(pnorm, "**", "*.gpkg"), recursive=True) +
                glob.glob(os.path.join(pnorm, "**", "*.geojson"), recursive=True) +
                glob.glob(os.path.join(pnorm, "**", "*.json"), recursive=True) +
                glob.glob(os.path.join(pnorm, "**", "*.shp"), recursive=True)
            )
            if not foundp:
                print('   [WARN] フォルダ内に .gpkg/.geojson/.json/.shp が見つかりません:', pnorm); 
                continue
            print(f'   [INFO] {len(foundp)} 個のポリゴンソースを検出')
            preset_poly_paths.extend(foundp)
            continue
        if os.path.isfile(pnorm) and pnorm.lower().endswith(('.gpkg','.geojson','.json','.shp')):
            preset_poly_paths.append(pnorm)
            continue
        print('   [WARN] gpkg/geojson/json/shp でもフォルダでもありません。')

    epsg = ask_int("EPSGコード（例 6674）", default=6674)

    # BBOX（共通交差を既定）
    try:
        raster_paths = [p for _, p in rasters]
        def _intersect(paths):
            first=True
            for p in paths:
                with rasterio.open(p) as src:
                    b=src.bounds
                    if first:
                        xmin,ymin,xmax,ymax=b.left,b.bottom,b.right,b.top
                        crs0=src.crs; first=False
                    else:
                        if src.crs != crs0: raise ValueError("CRSが異なります。VRT等で揃えてください。")
                        xmin=max(xmin,b.left); ymin=max(ymin,b.bottom)
                        xmax=min(xmax,b.right); ymax=min(ymax,b.top)
            if first or xmax<=xmin or ymax<=ymin: raise ValueError("共通交差範囲なし")
            return (xmin,ymin,xmax,ymax)
        bbox_def = _intersect(raster_paths)
        bbox_in = input(f"BBOX (xmin,ymin,xmax,ymax) [Enter=共通交差 {bbox_def}]: ").strip()
        bbox = bbox_def if bbox_in=="" else tuple(float(v) for v in bbox_in.split(","))
    except Exception:
        bbox = ask_bbox("BBOX (xmin,ymin,xmax,ymax)")

    res    = ask_float("サンプリング点グリッド解像度 [m]", default=1.0, minval=0.01)
    stride = ask_int("間引き間隔（1=全点, 2=1/4, 3=1/9 ...）", default=1, minval=1)
    rnd = input("ランダム間引き (0<f<=1, 空=無効) 例 0.2: ").strip()
    random_fraction = float(rnd) if rnd else None
    if random_fraction is not None and not (0 < random_fraction <= 1):
        print("[ERROR] 0<f<=1 にしてください。"); return
    seed = ask_int("乱数シード", default=42)
    outtab = ask_path("出力テーブル（.parquet / .csv / .gpkg）", must_exist=False,
                      default=str(Path.cwd() / "train_points.parquet"))

    # 点生成
    xmin,ymin,xmax,ymax = bbox
    xs = np.arange(xmin + res/2, xmax, res, dtype=float)
    ys = np.arange(ymin + res/2, ymax, res, dtype=float)
    if xs.size==0 or ys.size==0:
        print("点が生成できません。BBOX/解像度を確認してください。"); return
    xx,yy = np.meshgrid(xs, ys); pts = np.column_stack([xx.ravel(), yy.ravel()])
    if stride>1:
        nx = len(xs)
        iy, ix = np.divmod(np.arange(pts.shape[0]), nx)
        keep = (ix % stride == 0) & (iy % stride == 0)
        pts = pts[keep]
    if random_fraction is not None:
        rng = np.random.default_rng(seed)
        pts = pts[rng.random(pts.shape[0]) < random_fraction]
    print(f"[INFO] 点数: {len(pts):,}")

    # サンプリング
    from rasterio.vrt import WarpedVRT
    outcols = {name: np.empty(len(pts), dtype=float) for name,_ in rasters}
    failed = []  # 追加：失敗ログ

    for name, path in rasters:
        try:
            with rasterio.open(path) as src:
                with WarpedVRT(src, crs=f"EPSG:{epsg}",
                               resampling=Resampling.nearest,
                               src_nodata=src.nodata, dst_nodata=np.nan) as vrt:
                    vals = np.fromiter((v[0] for v in vrt.sample(pts)),
                                       dtype=float, count=len(pts))
                    nd = vrt.nodata
                    if nd is not None:
                        if isinstance(nd, float) and np.isnan(nd):
                            vals = np.where(np.isnan(vals), np.nan, vals)
                        else:
                            vals = np.where(vals==nd, np.nan, vals)
                    outcols[name] = vals
            print(f"[OK] {name}")
        except Exception as e:
            print(f"[FAIL] {name} :: {e}")
            failed.append((name, path, str(e)))
            # 失敗列は全部 NaN で埋めておく（列は残す）
            outcols[name] = np.full(len(pts), np.nan, dtype=float)

    df = pd.DataFrame({"x": pts[:,0], "y": pts[:,1]})
    for k,v in outcols.items(): df[k] = v
    df.attrs["epsg"] = str(epsg)


    # --- メモリ/I-O安定化：特徴量だけ float32 に（x,y は精度保持のため float64 のまま）---
    for c in df.columns:
        if c not in ("x", "y"):
            df[c] = df[c].astype("float32")

    # --- ポリゴン属性の一括付与（任意・複数可） ----------------------
    # 先に登録があれば自動で有効化。なければ従来どおり確認。
    use_poly = bool(preset_poly_paths) or ask_yesno("GPKG/GeoJSON/Shapefile等のポリゴン属性を点に付けますか？", default="n")
    while use_poly:
        # 先行登録があればそれを使う。なければ都度入力。
        if preset_poly_paths:
            seeds = list(preset_poly_paths)
            preset_poly_paths = []  # 使い切り
        else:
            raw = ask_path("ポリゴンファイル or ディレクトリのパス（空でスキップ可）", must_exist=True, allow_empty=True)
            if not raw:
                break
            seeds = [raw]
        poly_jobs = []


        # 収集: ファイル/フォルダ（再帰）
        def _push_file(fp):
            low = fp.lower()
            if low.endswith(".gpkg"):
                try:
                    import fiona
                    layers = fiona.listlayers(fp)
                    if len(layers) <= 1:
                        poly_jobs.append((fp, layers[0] if layers else None))
                    else:
                        print(f"\nレイヤー一覧: {fp}")
                        for i, nm in enumerate(layers):
                            print(f"  [{i:02d}] {nm}")
                        sel = input("読み込むレイヤ番号（カンマ/範囲可。空=すべて）: ").strip()
                        if sel:
                            # 例: "0,2-4"
                            picks = _parse_col_selection(sel, layers)
                            for nm in picks:
                                poly_jobs.append((fp, nm))
                        else:
                            for nm in layers:
                                poly_jobs.append((fp, nm))
                except Exception:
                    poly_jobs.append((fp, None))
            elif low.endswith((".geojson", ".json", ".shp")):
                poly_jobs.append((fp, None))

        # 先行シードを展開
        for seed in seeds:
            pth = os.path.normpath(str(seed))
            if os.path.isdir(pth):
                found = sorted(
                    glob.glob(os.path.join(pth, "**", "*.gpkg"), recursive=True)
                    + glob.glob(os.path.join(pth, "**", "*.geojson"), recursive=True)
                    + glob.glob(os.path.join(pth, "**", "*.json"), recursive=True)
                    + glob.glob(os.path.join(pth, "**", "*.shp"), recursive=True)
                )
                if not found:
                    print("[WARN] フォルダ内に .gpkg/.geojson/.json/.shp が見つかりません:", pth)
                for fp in found:
                    _push_file(fp)
            elif os.path.isfile(pth):
                _push_file(pth)
            else:
                print("[WARN] ファイル/フォルダではありません:", pth)
        if poly_jobs:
            print(f"[INFO] ポリゴン {len(poly_jobs)} 件を検出")


        if not poly_jobs:
            use_poly = ask_yesno("別のポリゴン指定を試しますか？", default="n"); 
            continue

        # 付与列の選択（既定=全部）、結合条件、接頭辞、ポリゴン外の点を落とすか
        predicate = input("結合条件（within/contains/intersects。空=within）: ").strip() or "within"
        prefix_all = input("付与列の接頭辞（ファイルごとに自動付与するなら空）: ").strip()
        drop_out = ask_yesno("ポリゴン外（付与列が全NaN）の点を除外しますか？", default="n")
        # ファイル間のヒット集約ロジック（AND=全て命中 / OR=どれか命中）
        hit_logic = "AND"
        if drop_out and len(poly_jobs) > 1:
            s = (input("複数ファイルのヒット条件（AND/OR）[AND]: ").strip().upper() or "AND")
            hit_logic = "OR" if s == "OR" else "AND"

        kept_mask_global = None
        for (poly_path, layer) in poly_jobs:
            # 列一覧取得
            poly_gdf = gpd.read_file(poly_path, layer=layer) if layer else gpd.read_file(poly_path)
            cand_cols = [c for c in poly_gdf.columns if c != "geometry"]
            print(f"\n付与候補の列（{os.path.basename(poly_path)}"
                  + (f"::{layer}" if layer else "") + "）:")
            for i, c in enumerate(cand_cols):
                print(f"[{i:02d}] {c}")
            sel_cols = input("付与する属性（番号/名前のカンマ区切り。範囲可。空=全部）: ").strip()
            keep_cols = _parse_col_selection(sel_cols, cand_cols) if sel_cols else cand_cols

            # プレフィクス決定
            if prefix_all:
                prefix = prefix_all
            else:
                base = Path(poly_path).stem
                prefix = f"{base}_"  # ファイル名で衝突回避

            before_cols = set(df.columns)
            df = attach_polygon_attrs(
                df, epsg, poly_path, layer=layer,
                attr_cols=keep_cols, predicate=predicate, out_prefix=prefix
            )
            added = [c for c in df.columns if c not in before_cols]

            if drop_out and added:
                keep_mask = np.zeros(len(df), dtype=bool)
                for c in added:
                    keep_mask |= df[c].notna().values
                if kept_mask_global is None:
                    kept_mask_global = keep_mask
                else:
                    kept_mask_global = (kept_mask_global & keep_mask) if hit_logic=="AND" else (kept_mask_global | keep_mask)

        if drop_out and kept_mask_global is not None:
            df = df.loc[kept_mask_global].reset_index(drop=True)
            print(f"[INFO] ポリゴン内のみ: {len(df):,} 点")

        use_poly = ask_yesno("さらに別のポリゴンセットを追加しますか？", default="n")
    # --- 追加ここまで -------------------------------------------------------

    # 保存
    out_path = str(outtab)
    low = out_path.lower()
    if low.endswith(".parquet"):
        try:
            import pyarrow  # noqa
            df.to_parquet(
                out_path,
                index=False,
                compression="snappy",
                row_group_size=200_000
            )
        except Exception as e:
            print(f"  Parquet失敗→CSVへ: {e}")
            from pathlib import Path as _P
            df.to_csv(str(_P(out_path).with_suffix(".csv")), index=False)
    elif low.endswith(".csv"):
        df.to_csv(out_path, index=False)
    elif low.endswith(".gpkg"):
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["x"], df["y"]),
            crs=f"EPSG:{df.attrs.get('epsg','4326')}"
        )
        layer = Path(out_path).stem
        gdf.to_file(out_path, layer=layer, driver="GPKG")
    else:
        try:
            import pyarrow  # noqa
            df.to_parquet(
                out_path + ".parquet",
                index=False,
                compression="snappy",
                row_group_size=200_000
            )
        except Exception:
            df.to_csv(out_path + ".csv", index=False)
    print(f"[OK] 出力: {out_path}")
    # 失敗サマリ（あればCSVへ）
    if failed:
        print("\n[SUMMARY] 失敗ラスタ:")
        for name, path, msg in failed:
            print(f"  - {name} :: {path} :: {msg}")      
        try:
            import pandas as _pd
            base = str(Path(out_path).with_suffix(''))
            _pd.DataFrame(failed, columns=["name", "path", "error"]).to_csv(
                base + "_failed_rasters.csv", index=False, encoding="utf-8"
            )
            print("[INFO] 失敗一覧CSVを書き出しました。")
        except Exception as _e:
            print(f"[WARN] 失敗一覧CSVの書き出しに失敗: {_e}")
  
# =============== メイン ===============

def main():
    print("\n=== DEM → 特徴量作成（RSU/RSD/SM/RES 対応）＋ 学習データ出力 ===")
    print(" 1) 特徴量TIFを作る（選択した前処理×スケール）")
    print(" 2) 既存TIFから学習データを作る（サンプリングのみ）")
    print(" 3) まとめて：特徴量TIF → サンプリング")
    print(" 0) 終了")
    # 実行時設定の確認
    print(f"[CFG] RES_ALL_METRICS={int(RES_ALL_METRICS)}  OPENNESS_STEP_STRIDE={OPENNESS_STEP_STRIDE}  "
          f"WINDOW_TILE={WINDOW_TILE}  WINDOWED_WRITE={'ON' if USE_WINDOWED else 'OFF'}  "
          f"RES_BASE={RES_BASE or '(ask)'}  RES_GAUSS_SIGMA_PX={RES_GAUSS_SIGMA_PX or '(auto r/3)'}  "
          f"RSD_OUTPUT_GRID={RSD_OUTPUT_GRID}")
    ch = input("番号を選んでください [0-3]: ").strip() or "1"

    if ch == "1":
        _ = run_make_features()
    elif ch == "2":
        run_export_points(feature_paths=None)
    elif ch == "3":
        feats = run_make_features()  # [(name, path), ...]
        run_export_points(feature_paths=feats)
    else:
        print("終了します。")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断しました。")
