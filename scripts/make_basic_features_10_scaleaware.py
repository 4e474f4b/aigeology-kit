#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_basic_features_10_scaleaware.py

単一DEMから「比高・標準偏差・傾斜・曲率・開度・TRI」など
地形分類用の基本特徴量を、マルチスケールで一括生成するスクリプト。

使い方・パラメータ仕様の詳細は 下に示すmd を参照してください。
  - docs/make_basic_features_8_scaleaware.md

DEMの作り方など他スクリプトとの関係は、下に示す README を参照してください。
  - scripts/README.md

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
    convolve,
    gaussian_filter,
)

def _r_to_k_list(relief_r_list, px):
    """
    R[m] -> k（窓サイズ, 奇数）に変換する。
    例: px=10m のとき R=30,50,90 -> k=3,5,9
    """
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
    """
    R[m] -> k（窓サイズ, 奇数）に変換する。
    例: px=10m のとき R=30,50,90 -> k=3,5,9
    """
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


def _dedupe_preserve_order(items, key=None):
    """順序を維持して重複を除去する（表示用・上書き防止用）。"""
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
    """
    カンマ区切りで複数半径を受け取る。
    例: "2,5,10"
    """
    default_str = ",".join(str(v) for v in default_list)
    while True:
        s = input(f"{prompt} [{default_str}]: ").strip()
        if not s:
            return list(default_list)
        try:
            # 例: "10.0,20,30]" のような入力を許容（コピペ対策）
            s2 = s.strip().strip("[]")
            vals = [float(x.strip()) for x in s2.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except ValueError:
            print("  カンマ区切りの数値で入力してください（例: 10,20,30）。")


def ask_int_list(prompt, default_list):
    """
    カンマ区切りで複数の整数を受け取る。
    例: "3,7,15"
    """
    default_str = ",".join(str(v) for v in default_list)
    while True:
        s = input(f"{prompt} [{default_str}]: ").strip()
        if not s:
            return list(default_list)
        try:
            vals = [int(x) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except ValueError:
            print("  カンマ区切りの整数で入力してください（例: 3,5,11）。")


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


def _apply_valid_mask(arr: np.ndarray, win_pix: int):
    """
    外縁マスク用ヘルパ:
      - win_pix が k×k 窓サイズだとすると、
        半径 r = floor(k/2) ピクセル分の外縁を「近傍が足りない」とみなし NaN にする。
      - その後、書き出し時に NaN は nodata_value に変換される。
    """
    h, w = arr.shape
    k = int(max(1, win_pix))
    r = k // 2

    mask = np.ones_like(arr, dtype=bool)
    mask[r:h-r, r:w-r] = False
    out = arr.copy()
    out[mask] = np.nan
    return out


# =============== NaN 対応スムージング ===============

def smooth_box_nan(arr: np.ndarray, win_pix: int) -> np.ndarray:
    """
    NaN（NoData）に対応した移動平均（ボックスフィルタ）平滑化。
    - arr: DEM（NaN = NoData）
    - win_pix: 窓サイズ（ピクセル数, k×k）
    """
    k = int(max(1, win_pix))
    valid = np.isfinite(arr).astype(np.float32)
    filled = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)

    num = uniform_filter(filled, size=k, mode="nearest")
    den = uniform_filter(valid, size=k, mode="nearest")

    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out.astype(np.float32)


def smooth_gauss_nan(arr: np.ndarray, sigma_pix: float) -> np.ndarray:
    """
    NaN（NoData）に対応したガウシアン平滑化。
    - sigma_pix: ガウシアンの σ [pixel]
    """
    if sigma_pix <= 0:
        return arr.astype(np.float32)

    valid = np.isfinite(arr).astype(np.float32)
    filled = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)

    num = gaussian_filter(filled, sigma=sigma_pix, mode="nearest")
    den = gaussian_filter(valid, sigma=sigma_pix, mode="nearest")

    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out.astype(np.float32)


# =============== 比高 / 標準偏差 ===============

def local_relief(arr, win_pix: int):
    """局所比高：窓内の max - min"""
    valid = np.isfinite(arr)

    # 局所最大（NaNは -inf 扱い）
    a_maxin = np.where(valid, arr, -np.inf)
    k = int(max(1, win_pix))
    loc_max = maximum_filter(a_maxin, size=k, mode="nearest")

    # 局所最小（NaNは +inf 扱い）
    a_minin = np.where(valid, arr, +np.inf)
    loc_min = minimum_filter(a_minin, size=k, mode="nearest")

    out = (loc_max - loc_min).astype(np.float32)
    out[~valid] = np.nan
    return out


def local_stddev(arr, win_pix: int):
    """局所標準偏差（NaN 対応：分子/分母を別平滑し正規化）"""
    k = int(max(1, win_pix))
    valid = np.isfinite(arr).astype(np.float32)
    a = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    n = uniform_filter(valid, size=k, mode="nearest")              # 有効ピクセル数（平滑）
    sum1 = uniform_filter(a, size=k, mode="nearest") * n           # μ * n
    sum2 = uniform_filter(a * a, size=k, mode="nearest") * n       # E[z^2] * n
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sum1 / n
        mean_sq = sum2 / n
    var = mean_sq - mean * mean
    var[var < 0] = 0.0
    std = np.sqrt(var).astype(np.float32)
    std[n == 0] = np.nan
    return std


def local_mean_nan(arr, win_pix: int):
    """局所平均（NaN対応）"""
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
    """TPI = z - mean(neighborhood)（NaN対応）"""
    m = local_mean_nan(arr, win_pix)
    out = (arr.astype(np.float32) - m).astype(np.float32)
    out[~np.isfinite(arr)] = np.nan
    return out

def _shift_edge(arr, dy: int, dx: int):
    """端は edge で埋めるシフト（wrap禁止）"""
    h, w = arr.shape
    py, px = abs(dy), abs(dx)
    a = np.pad(arr, ((py, py), (px, px)), mode="edge")
    y0 = py - dy
    x0 = px - dx
    return a[y0:y0+h, x0:x0+w]

def local_tri(arr, win_pix: int):
    """
    TRI（8近傍差分型, スケールは win_pix から d=(k-1)//2 を採用）
    TRI = mean(|z - z_neighbor|) over 8 directions, NaN対応
    """
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


# =============== 勾配 / 方位 / ラプラシアン / 平均曲率 ===============

def _horn_gradients(arr, px: float):
    """
    Horn (1981) の 3×3 オペレータで勾配ベクトル (gx, gy) を計算
    """
    mask = ~np.isfinite(arr)

    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32) / (8.0 * px)

    ky = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]], dtype=np.float32) / (8.0 * px)

    gx = convolve(arr, kx, mode="nearest")
    gy = convolve(arr, ky, mode="nearest")

    gx[mask] = np.nan
    gy[mask] = np.nan
    return gx, gy


def slope_deg(arr, px: float):
    """勾配角 [deg]（8近傍）"""
    gx, gy = _horn_gradients(arr, px)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    slope[~np.isfinite(arr)] = np.nan
    return slope.astype(np.float32)


def aspect_deg(arr, px: float):
    """斜面方位 [deg]（0=東, 90=北, 8近傍）"""
    gx, gy = _horn_gradients(arr, px)
    asp = np.degrees(np.arctan2(gy, gx))
    asp = np.mod(90.0 - asp, 360.0)
    asp[~np.isfinite(arr)] = np.nan
    return asp.astype(np.float32)


def laplacian(arr, px: float):
    """
    8近傍ラプラシアン
    """
    mask = ~np.isfinite(arr)

    k = np.array([[1,  1, 1],
                  [1, -8, 1],
                  [1,  1, 1]], dtype=np.float32) / (px * px)

    out = convolve(arr, k, mode="nearest").astype(np.float32)
    out[mask] = np.nan
    return out


def mean_curvature(arr, px: float):
    """
    平均曲率（簡易実装）
    """
    gy, gx = np.gradient(arr, px, px)
    zyy, zyx = np.gradient(gy, px, px)
    zxy, zxx = np.gradient(gx, px, px)

    zx2 = gx * gx
    zy2 = gy * gy
    denom = 2.0 * np.power(1.0 + zx2 + zy2, 1.5)

    num = (1.0 + zy2) * zxx - 2.0 * gx * gy * zxy + (1.0 + zx2) * zyy
    with np.errstate(invalid="ignore", divide="ignore"):
        H = num / denom
    H[~np.isfinite(arr)] = np.nan
    return H.astype(np.float32)


# =============== 開度（Python 内蔵版） ===============

def _unit_dirs(n_dirs: int):
    """0〜πの半円方向に n_dirs 本の単位ベクトル"""
    thetas = np.linspace(0.0, math.pi, n_dirs, endpoint=False)
    return np.cos(thetas), np.sin(thetas)


def openness_pair(dem, px: float, r_open: float, n_dirs: int = 8, step_stride: int = 1):
    """
    正開度 / 負開度 を Python のみで計算する簡易実装。
    - r_open[m] まで、各方向にレイを飛ばし最大仰角 / 最大俯角を評価。
    """
    h, w = dem.shape
    cx = np.arange(w)
    cy = np.arange(h)
    X, Y = np.meshgrid(cx, cy)

    # 単位方向ベクトル
    cos_t, sin_t = _unit_dirs(n_dirs)

    # 出力配列
    pos_list = []
    neg_list = []

    max_step = max(1, int(r_open / px))

    valid = np.isfinite(dem)

    for k in range(n_dirs):
        dx = cos_t[k]
        dy = sin_t[k]

        # 各ステップでの仰角 / 俯角を評価
        alphas = []
        betas = []

        for step in range(1, max_step + 1, step_stride):
            # 実数座標
            x_f = X + dx * step
            y_f = Y - dy * step

            x0 = np.floor(x_f).astype(int)
            y0 = np.floor(y_f).astype(int)
            x1 = x0 + 1
            y1 = y0 + 1

            # 範囲チェック
            inside = (
                (x0 >= 0) & (x1 < w) &
                (y0 >= 0) & (y1 < h)
            )

            if not inside.any():
                continue

            # bilinear
            wx = x_f - x0
            wy = y_f - y0

            # inside=False の要素にも範囲外インデックスが含まれるため、
            # 参照前にクリップして IndexError を防ぐ
            x0c = np.clip(x0, 0, w - 1)
            x1c = np.clip(x1, 0, w - 1)
            y0c = np.clip(y0, 0, h - 1)
            y1c = np.clip(y1, 0, h - 1)

            z00 = dem[y0c, x0c]
            z10 = dem[y0c, x1c]
            z01 = dem[y1c, x0c]
            z11 = dem[y1c, x1c]

            z_top = (
                z00 * (1 - wx) * (1 - wy) +
                z10 * wx * (1 - wy) +
                z01 * (1 - wx) * wy +
                z11 * wx * wy
            )

            d = step * px  # 水平距離

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
            max_alpha = np.nanmax(alpha_stack, axis=0)
            max_beta = np.nanmax(beta_stack, axis=0)

        # 天頂角ベース
        pos_dir = 90.0 - max_alpha
        neg_dir = 90.0 - max_beta

        pos_list.append(pos_dir)
        neg_list.append(neg_dir)

    if pos_list:
        pos = np.nanmean(np.stack(pos_list, axis=0), axis=0).astype(np.float32)
        neg = np.nanmean(np.stack(neg_list, axis=0), axis=0).astype(np.float32)
    else:
        pos = np.full_like(dem, np.nan, dtype=np.float32)
        neg = np.full_like(dem, np.nan, dtype=np.float32)
    pos[~valid] = np.nan
    neg[~valid] = np.nan
    return pos, neg


# =============== SAGA を用いた開度 ===============

def confirm_saga_openness_tool(saga_cmd_path: str, tool_id: int = 5) -> bool:
    """
    saga_cmd ta_lighting -h で 5: Topographic Openness の存在を確認。
    SAGA 側の「[Error] select a tool」は単なるヘルプ終了メッセージなので、
    コンソールには出さないようにフィルタする。
    """
    print("\n=== SAGA ta_lighting モジュールの一覧を確認します... ===")
    try:
        proc = subprocess.run(
            [saga_cmd_path, "ta_lighting", "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        out = proc.stdout or ""

        # 紛らわしいだけのヘルプ終端メッセージを除去
        filtered_lines = []
        for line in out.splitlines():
            if "[Error]" in line and "select a tool" in line:
                continue
            filtered_lines.append(line)

        print("\n".join(filtered_lines))
    except Exception as e:
        print(f"[WARN] saga_cmd の実行に失敗しました: {e}")
        return False

    return f"[{tool_id}]" in out


def compute_openness_with_saga(
    saga_cmd_path: str,
    dem_path: Path,
    out_dir: Path,
    stem: str,
    openness_r_list,
    n_dirs: int,
    nodata_value: float,
    meta,
    px: float,
    global_mask_win: int,
):
    """
    SAGA ta_lighting:5 (Topographic Openness) を利用して
    正開度 / 負開度を計算する。
    """
    for r_open in openness_r_list:
        rtag = int(round(r_open))

        # r_open[m] から「窓サイズ」と同等のピクセル幅を計算
        win_pix_open = _winpix_from_meters(r_open, px, min_win=1, odd=True)

        tmp_dir = out_dir / f"_tmp_saga_r{rtag}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        tmp_pos = tmp_dir / "pos.sdat"
        tmp_neg = tmp_dir / "neg.sdat"

        print(f"\n  [SAGA] R_open={r_open}m, n_dirs={n_dirs} で開度を計算中...")

        cmd = [
            saga_cmd_path,
            "ta_lighting",
            "5",  # Topographic Openness
            "-DEM", str(dem_path),
            "-POS", str(tmp_pos),
            "-NEG", str(tmp_neg),
            "-RADIUS", str(r_open),
            "-NDIRS", str(n_dirs),
            "-DIRECTIONS", "1",   # 1: all directions
            "-METHOD", "1",       # 1: line tracing
            "-UNIT", "1",         # 1: Degree
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] SAGA ta_lighting でエラーが発生しました: {e}")
            continue

        def _sdat_to_tif(sdat_path: Path, out_tif: Path):
            """
            SAGA の .sdat を GeoTIFF に変換する簡易ラッパ。
            """
            with rasterio.open(sdat_path) as ssrc:
                data = ssrc.read(1).astype(np.float32)
                saga_nodata = ssrc.nodata

            mask = ~np.isfinite(data)
            if saga_nodata is not None:
                mask |= (data == saga_nodata)
            data[mask] = np.nan

            # 外縁マスクもグローバルな窓サイズで共通化
            data = _apply_valid_mask(data, global_mask_win)

            data = np.where(np.isfinite(data), data, nodata_value).astype(np.float32)

            meta_out = meta.copy()
            meta_out.update(
                dtype="float32",
                nodata=nodata_value,
                count=1,
            )
            with rasterio.open(out_tif, "w", **meta_out) as dst:
                dst.write(data, 1)

        pos_tif = out_dir / f"{stem}_openness_pos_r{rtag}m_nd{n_dirs}dir.tif"
        neg_tif = out_dir / f"{stem}_openness_neg_r{rtag}m_nd{n_dirs}dir.tif"

        _sdat_to_tif(tmp_pos, pos_tif)
        _sdat_to_tif(tmp_neg, neg_tif)

        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass


# =============== メイン処理 ===============

def main():
    print("=== DEM → 10特徴量 一括出力（マルチスケール＋SAGA開度対応＋外縁マスク統一） ===")

    dem_path_str = ask_path("入力DEM GeoTIFF のパス", must_exist=True)
    dem_path = Path(dem_path_str)
    stem = dem_path.stem

    default_out_dir = str(dem_path.with_suffix("").parent / (stem + "_features"))
    out_dir_str = ask_path(f"出力フォルダ [{default_out_dir}]", must_exist=False, default=default_out_dir)
    out_dir = Path(out_dir_str)

    # 出力フォルダにファイルパス（例: *.tif）を入れても落ちないように補正
    try:
        is_file = out_dir.exists() and out_dir.is_file()
    except Exception:
        is_file = False
    if is_file or out_dir.suffix.lower() in (".tif", ".tiff", ".vrt"):
        print(f"[WARN] 出力フォルダにファイルが指定されています: {out_dir}")
        print(f"       → {default_out_dir} に補正します。")
        out_dir = Path(default_out_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)

    # DEM 読み込み
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is None:
            nodata = -9999.0
        dem = np.where(np.isfinite(dem), dem, np.nan).astype(np.float32)

        transform: Affine = src.transform
        px = transform.a  # ピクセルサイズ（x方向）
        py = -transform.e

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

    # SAGA が使えるか判定
    saga_cmd_path = shutil.which("saga_cmd")
    has_saga = saga_cmd_path is not None

    # ------------------------------
    # 開度の計算方法を先に決める
    # ------------------------------
    print("\n=== 開度の計算方法 ===")
    if has_saga:
        print(f"  saga_cmd が見つかりました: {saga_cmd_path}")
        print("  ※ このスクリプトを実行している環境で `saga_cmd` に PATH が通っていれば、そのまま利用できます。")
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
        print("  saga_cmd が見つかりませんでした（PATH 未設定 or SAGA 未インストール）。")
        print("  ※ SAGA の開度を使いたい場合は、OS に SAGA をインストールし、`saga_cmd` に PATH を通してください。")
        print("  1: Python 内蔵の開度計算を使う（小〜中規模DEM向け）")
        print("  2: 開度は計算しない")
        openness_mode_raw = ask_int("番号を選んでください", 1)
        if openness_mode_raw == 1:
            openness_mode = 2  # Python 内蔵
        else:
            openness_mode = 3  # スキップ

    # 比高・偏差：スケール指定（m or px）
    print("\n[比高・標準偏差のスケール指定]\n")
    print("比高・標準偏差の空間スケール R[m] を指定してください（カンマ区切り可）。")
    print("  例) px=10m のとき R=30 → 3×3窓, R=50 → 5×5窓, R=90 → 9×9窓")
    print("  ※ R から窓サイズ k（奇数）は自動計算します。\n")

    relief_r_list = ask_float_list(
        "R[m]（例: 30,50,90）",
        [30.0, 50.0, 90.0],
    )

    # 入力Rは保持（ファイル名の重複回避・「入力8本=8スケール」担保のため）
    relief_r_in_list = list(relief_r_list)
    k_list = _r_to_k_list(relief_r_in_list, px)
    adopted_r_list = [float(k) * float(px) for k in k_list]
    
    print("\n  → 入力Rに対して採用される実効R（=k×px）:")
    for Rin, k, Radopt in zip(relief_r_in_list, k_list, adopted_r_list):
        print(f"    R_in={Rin:.3f} m → k={k}（{k}×{k}窓） → R_adopt={Radopt:.3f} m")

    # 計算は k で行う（Rを上書きしない）
    # relief_r_list = adopted_r_list  # ←削除

    # R_open_match の求め方（外接円 / 面積等価円 / 内接円）を選択
    print("  1: 外接円（正方形窓の外接円） R_open = √2 × R_eff")
    print("  2: 面積等価円（k×k窓と同面積） R_open = (2/√π) × R_eff")
    print("  3: 内接円（正方形窓の内接円） R_open = R_eff")
    # デフォルトは docs と揃えて「面積等価円」
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
    print(f"     R_open = {coeff:.5f} × R_eff")

    print("\n---情報--- 指定した R[m] から窓サイズ k と R_open_match を計算します。")
    win_pix_list = []
    r_eff_list = []
    r_open_match_list = []
    # N_R（=比高/標準偏差のスケール数）
    # ※この後の平滑化 r_px_smooth 自動提案に使う
    for Rin, k in zip(relief_r_in_list, k_list):
        win_pix_list.append(k)
        R_eff = ((k - 1) / 2.0) * px
        r_eff_list.append(R_eff)
        R_open_match = coeff * R_eff
        r_open_match_list.append(R_open_match)
        print(
            f"  R_in={Rin:.3f}m → k={k} (窓サイズ {k}×{k}), "
            f"R_eff≒{R_eff:.3f}m, R_open_match≒{R_open_match:.3f}m"
        )

    print(
        "\n---注意--- R が DEM ピクセルサイズに対して大きくなるほど、"
        "比高・標準偏差はより広域の地形（谷/尾根/盆地など）を表すようになります。"
    )

    # 外縁マスク候補（窓サイズ）の初期値: 比高・標準偏差に必要な窓サイズ
    mask_win_candidates = list(win_pix_list)

    # 開度パラメータ（必要なときだけ）
    openness_r_list = []
    n_dirs = None
    step_stride = 1

    if openness_mode != 3:
        default_open_list = [float(f"{v:.3f}") for v in r_open_match_list]

        print("\n---注意--- 開度 R_open の解釈について"
              "\n  ・R_open は中心セルから各方向に伸びるレイの「最大距離[m]」です。"
              "\n  ・R_open が大きいほど、より広域の地形骨格（大きな谷・尾根・盆地）を反映します。"
              "\n  ・比高/標準偏差の窓と空間スケールを揃えたい場合は、"
              "\n    上で表示した R_open_match を目安に指定してください。")

        info_pairs = ", ".join(
            f"R={R:.3g}m→R_open_match≒{Ropen:.3g}m"
            for R, Ropen in zip(relief_r_list, r_open_match_list)
        )
        prompt_open = (
            "開度の最大距離 R_open[m]（カンマ区切り可, Enterで「比高と同スケール」の推奨値）"
            f"\n  スケール対応の目安: {info_pairs}"
        )
        openness_r_list = ask_float_list(
            prompt_open,
            default_open_list,
        )

        # 0 以下を除外（R_eff=0 → R_open=0 の無駄計算を防ぐ）
        openness_r_list = [float(x) for x in openness_r_list if float(x) > 0.0]
        # 重複除去（表示順は維持）
        _seen = set()
        _tmp = []
        for x in openness_r_list:
            key = round(x, 6)
            if key in _seen:
                continue
            _seen.add(key)
            _tmp.append(x)
        openness_r_list = _tmp

        n_dirs = ask_int(
            "開度の方向数 n_dirs（8 or 16 推奨）", 8
        )

        if openness_mode == 2:
            step_stride = ask_int(
                "開度のサンプリング間隔（ピクセル, Python 内蔵のみ）"
                "\n  1: 1,2,3,...ピクセルごと / 2: 2,4,6,... / 3: 3,6,9,...",
                1,
            )

            print("\n  > 開度 (Python 内蔵) の評価設定")
            for r_open in openness_r_list:
                n_px = max(1, int(round(float(r_open) / float(px))))
                print(
                    f"    - R_open={r_open}m → 1〜{n_px} ピクセル先まで"
                    f"（step_stride={step_stride} ピクセル刻み）を評価"
                )
        elif openness_mode == 1:
            print("\n  > 開度 (SAGA ta_lighting 5) の評価設定")
            for r_open in openness_r_list:
                n_px = max(1, int(round(float(r_open) / float(px))))
                print(
                    f"    - R_open={r_open}m → 1〜{n_px} ピクセル先まで"
                    f"（n_dirs={n_dirs} 方向）の最大天頂角を平均"
                )

        # 開度に必要なマスク窓サイズも候補に追加
        for r_open in openness_r_list:
            win_pix_open = _winpix_from_meters(r_open, px, min_win=1, odd=True)
            mask_win_candidates.append(win_pix_open)


    if openness_mode == 2:
        total_px = h * w
        if total_px > 50_000_000:
            print(
                f"\n---注意--- DEM の総ピクセル数は約 {total_px/1e6:.1f} 百万ピクセルです。"
                "\n       Python 内蔵の開度計算は非常に時間がかかる可能性があります。"
                "\n       必要であれば SAGA の利用や解像度ダウンサンプリングも検討してください。"
            )

    # ------------------------------
    # 勾配・曲率系のスムージング設定
    # ------------------------------
    print("\n=== 勾配・曲率系のスムージング設定 ===\n")
    # --- ガイド: 比高スケールに基づく推奨スムージング ---
    def _round_m(v, base=1.0):
        # m 単位を見やすく丸め（1m 単位。必要なら 5m/10m に変更）
        return round(v / base) * base

    # 提案セット（各 R_eff から [R_eff, R_eff/2, R_eff/4]）
    guide_sets = []
    for R_eff in r_eff_list:
        cand = [_round_m(R_eff), _round_m(R_eff/2.0), _round_m(R_eff/4.0)]
        # 非負・重複排除
        cand = [c for c in cand if c > 0]
        uniq = []
        for c in cand:
            if not any(abs(c - u) < 1e-6 for u in uniq):
                uniq.append(c)
        guide_sets.append(uniq)

    print("  ─ スムージング半径の目安（比高 R に整合） ─")
    for i, (R, R_eff, ks, gs) in enumerate(zip(relief_r_list, r_eff_list, win_pix_list, r_open_match_list)):
        # 表示用：Box窓・Gaussσ の換算例は「R_smooth = R_eff」を代表として出す
        Rsm = max(1e-6, R_eff)
        k_box = _winpix_from_meters(2.0 * Rsm, px, min_win=3, odd=True)  # 幅≒2*Rsm
        sigma_px = max(0.5, float(Rsm) / (3.0 * float(px)))
        width_m = k_box * px  # Box窓の物理幅 [m]（概ね「見ている範囲」）

        print(f"    ・比高スケール R={R:.3g}m（窓サイズ ≒{k_box}ピクセル ≒{width_m:.3g}m）")
        print(f"       → 平滑化の有効半径の代表値 R_eff≒{R_eff:.3g}m")
        print(f"       推奨候補（R_smooth[m]）: {', '.join(f'{v:.0f}' for v in guide_sets[i])}")
        print(f"       例）R_smooth={R_eff:.0f}m の場合: Box窓≈{k_box}ピクセル（幅≒{width_m:.3g}m）, Gauss σ≈{sigma_px:.2f}px")
    print("  （※ここでの R_smooth は『平滑化の有効半径[m]』です。"
          "Box 窓の物理幅≒2×R_smooth≒比高 R となるように設計しています）\n")
    print("  1: 生DEMのみで計算（従来通り）")
    print("  2: 生DEM + スムージング版も出力")
    print("  3: スムージング版のみ出力")
    slope_mode = ask_int("番号を選んでください", 1)

    smooth_filter_type = None
    smooth_r_list = []

    if slope_mode in (2, 3):
        print("\n  [平滑化フィルタの種類]")
        print("    1: 移動平均（ボックスフィルタ）")
        print("    2: ガウシアンフィルタ")
        fmode = ask_int("番号を選んでください", 1)
        smooth_filter_type = "box" if fmode == 1 else "gauss"

        print("\n  [平滑化スケールの指定方法]")
        print("    1: 距離 [m] で指定（例: 10,30,50）")
        print("    2: ピクセル数 [px] で指定（例: 3,5,11）")
        sm_mode = ask_int("番号を選んでください", 1)

        if sm_mode == 1:
            # --- 平滑化スケール R_smooth[m] のデフォルトの考え方 ---
            # guide_sets は「各 比高 R に対応する推奨平滑化半径の候補リスト」の集合。
            #   例）比高 R が1つ（R=6m）のとき: guide_sets = [[2.0, 1.0, 0.5]]
            #       比高 R が2つ（R=6m, 15m）のとき: guide_sets = [[2.0, 1.0, 0.5], [5.0, 3.0, 1.0]]
            #
            # ポリシー:
            #   ・比高 R が 1 つだけ → その R に対応する「先頭の値」だけをデフォルトにする（例: [2.0]）
            #   ・比高 R が複数      → 各 R に対応する「先頭の値」だけを並べたリストをデフォルトにする（例: [2.0, 5.0]）
            #   ・guide_sets が空    → フォールバックとして [10.0] を採用
            if not guide_sets:
                default_ms = [10.0]
            else:
                n_r = len(guide_sets)
                if n_r == 1:
                    default_ms = [guide_sets[0][0]]
                else:
                    default_ms = [g[0] for g in guide_sets]

            # 表示用: 比高R と整合する推奨 R_smooth を 1 行で提示
            _hint_ms = "  スケール対応の目安: " + ", ".join(
                f"R={float(R):.3g}m→R_smooth≒{float(ms):.3g}m"
                for R, ms in zip(relief_r_list, default_ms)
            )
            smooth_r_list = ask_float_list(
                "平滑化スケール R_smooth[m]（カンマ区切り可, 例: 10,30,50）\n"
                 f"{_hint_ms}\n"
                "    ※ここで指定する R_smooth は「平滑化の有効半径[m]」です。\n"
                "      上で表示された『推奨候補（m）』のうち先頭の値を選ぶと、\n"
                "      比高 R と勾配・曲率の平滑化が、ほぼ同じ空間スケール\n"
                "      （窓幅 ≒ 比高の R）を見るように調整されています。\n"
                "      （例：2mDEM で R=6m のとき、R_smooth=2m → 3×3窓（幅≒6m））",
                default_ms,
            )
        else:
            # デフォルト: 比高と同じスケール数 N_R=len(win_pix_list) を基準に、
            #            r_px_smooth（=R_smooth/px）を自動提案（R_smooth≒R_eff）
            N_R = len(win_pix_list)
            if N_R <= 0:
                default_px_list_sm = [1]
            else:
                default_px_list_sm = [max(1, int(round(R_eff / px))) for R_eff in r_eff_list]
                if len(default_px_list_sm) < N_R:
                    default_px_list_sm += [default_px_list_sm[-1]] * (N_R - len(default_px_list_sm))
                elif len(default_px_list_sm) > N_R:
                    default_px_list_sm = default_px_list_sm[:N_R]

            # 表示用: 比高R と整合する推奨 r_px_smooth（および換算 R_smooth）を 1 行で提示
            _hint_px = "  スケール対応の目安: " + ", ".join(
                f"R={float(R):.3g}m→r_px_smooth≒{int(p)}px(R_smooth≒{float(p)*float(px):.3g}m)"
                for R, p in zip(relief_r_list, default_px_list_sm)
            )
            px_list_sm = ask_int_list(
                "平滑化スケール（ピクセル数）r_px_smooth（カンマ区切り可, 例: 5,11,21）\n"
                f"{_hint_px}\n"
                "    ※ここで指定する r_px_smooth は「有効半径 R_smooth をピクセル単位にした値」です。\n"
                "      Box の場合: R_smooth≒r_px_smooth×px となり、窓サイズは概ね k≒2*r_px_smooth+1。\n"
                "      Gauss の場合: R_smooth≒r_px_smooth×px とみなし、3σ≒R_smooth → σ_px≒r_px_smooth/3。",
                default_px_list_sm,
            )
            if len(px_list_sm) == 1 and N_R > 1:
                px_list_sm = px_list_sm * N_R
                print(f"[INFO] r_px_smooth は 1 個入力のため N_R={N_R} に展開します: {px_list_sm}")
            elif len(px_list_sm) < N_R:
                px_list_sm += [px_list_sm[-1]] * (N_R - len(px_list_sm))
                print(f"[INFO] r_px_smooth が不足のため末尾値で補完します: {px_list_sm}")
            elif len(px_list_sm) > N_R:
                px_list_sm = px_list_sm[:N_R]
                print(f"[INFO] r_px_smooth が過剰のため先頭 N_R={N_R} 個に切り詰めます: {px_list_sm}")
            smooth_r_list = [p * px for p in px_list_sm]
            print("\n  → ピクセル数から換算した R_smooth [m]:")
            for p, R in zip(px_list_sm, smooth_r_list):
                print(f"    {p} px → R_smooth ≒ {R:.3f} m")

        # 重複除去（順序維持）：同一R_smoothは同一出力名になり上書きされるため除去
        if smooth_r_list:
            _orig = list(smooth_r_list)
            smooth_r_list = _dedupe_preserve_order(smooth_r_list, key=lambda v: round(float(v), 6))
            if len(smooth_r_list) != len(_orig):
                print(f"[INFO] R_smooth の重複を除去しました: {smooth_r_list}")

        print("\n---情報--- スムージング版の勾配/曲率は、指定した R_smooth を")
        if smooth_filter_type == "box":
            print("       「中心から R_smooth[m] 程度の範囲を移動平均した DEM」から計算します。")
        else:
            print("       「有効半径 R_smooth[m]≒3σ となるガウシアンで平滑化した DEM」から計算します。")

    use_raw_deriv = slope_mode in (1, 2)
    use_smooth_deriv = slope_mode in (2, 3)

    # ここで「外縁マスク用 窓サイズ」を全指標で共通化
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

    # --- 比高 / 標準偏差（マルチスケール） ---
    for Rin, win_pix in zip(relief_r_in_list, k_list):
        R_eff = ((win_pix - 1) / 2.0) * px

        print(
            f"\n  > 比高 / 標準偏差: 入力R={Rin:.3f}m, "
            f"k={win_pix}, 有効半径R_eff≒{R_eff:.3f}m ..."
        )

        rel = local_relief(dem, win_pix)
        # 外縁マスクは global_mask_win で他指標と共通化
        rel_m = _apply_valid_mask(rel, global_mask_win)
        tag = f"rin{int(round(Rin))}m_k{win_pix}_reff{int(round(R_eff))}m"
        write_feature(f"{stem}_relief_{tag}.tif", rel_m)

        std = local_stddev(dem, win_pix)
        std_m = _apply_valid_mask(std, global_mask_win)
        write_feature(f"{stem}_stddev_{tag}.tif", std_m)


        # --- TPI / TRI（比高・標準偏差と同一スケール・同一命名規則） ---
        tpi = local_tpi(dem, win_pix)
        tpi_m = _apply_valid_mask(tpi, global_mask_win)
        write_feature(f"{stem}_tpi_{tag}.tif", tpi_m)

        tri = local_tri(dem, win_pix)
        tri_m = _apply_valid_mask(tri, global_mask_win)
        write_feature(f"{stem}_tri_{tag}.tif", tri_m)


    # --- 勾配 / 方位 / ラプラシアン / 平均曲率（生 DEM 版） ---
    if use_raw_deriv:
        print("\n  > 勾配角 (slope_deg) ...")
        slope = slope_deg(dem, px)
        slope = _apply_valid_mask(slope, global_mask_win)
        write_feature(f"{stem}_slope_deg.tif", slope)

        print("  > 斜面方位 (aspect_deg) ...")
        aspect = aspect_deg(dem, px)
        aspect = _apply_valid_mask(aspect, global_mask_win)
        write_feature(f"{stem}_aspect_deg.tif", aspect)

        print("  > ラプラシアン (laplacian) ...")
        lap = laplacian(dem, px)
        lap = _apply_valid_mask(lap, global_mask_win)
        write_feature(f"{stem}_laplacian.tif", lap)

        print("  > 平均曲率 (mean_curvature) ...")
        meancurv = mean_curvature(dem, px)
        meancurv = _apply_valid_mask(meancurv, global_mask_win)
        write_feature(f"{stem}_mean_curvature.tif", meancurv)

    # --- 勾配 / 方位 / ラプラシアン / 平均曲率（スムージング版） ---
    if use_smooth_deriv and smooth_r_list:
        for R_s in smooth_r_list:
            rtag_s = int(round(R_s))

            if smooth_filter_type == "box":
                win_s = _winpix_from_meters(R_s, px, min_win=3, odd=True)
                print(f"\n  > 平滑化DEM（移動平均） R_smooth={R_s}m (k={win_s}) から勾配/曲率を計算...")
                dem_s = smooth_box_nan(dem, win_s)
                smooth_tag = f"r{rtag_s}m_box"
            else:
                # 3σ ≈ R_smooth → σ_px ≈ R_smooth / (3 * px)
                sigma_pix = max(0.5, float(R_s) / (3.0 * float(px)))
                print(f"\n  > 平滑化DEM（ガウシアン） R_smooth={R_s}m (σ≈{sigma_pix:.2f}px) から勾配/曲率を計算...")
                dem_s = smooth_gauss_nan(dem, sigma_pix)
                smooth_tag = f"r{rtag_s}m_gauss"

            # このスムージング半径でも外縁マスク候補を更新しておく
            # （すでに global_mask_win は決まっているが、将来拡張用に残してもよい）
            # mask_win_candidates.append(win_for_mask)  # ※ main() ではすでに決定済み

            # スムージングDEMに対して、通常と同じ 3×3 オペレータを当てる
            slope_s = slope_deg(dem_s, px)
            slope_s = _apply_valid_mask(slope_s, global_mask_win)
            write_feature(f"{stem}_slope_deg_{smooth_tag}.tif", slope_s)

            aspect_s = aspect_deg(dem_s, px)
            aspect_s = _apply_valid_mask(aspect_s, global_mask_win)
            write_feature(f"{stem}_aspect_deg_{smooth_tag}.tif", aspect_s)

            lap_s = laplacian(dem_s, px)
            lap_s = _apply_valid_mask(lap_s, global_mask_win)
            write_feature(f"{stem}_laplacian_{smooth_tag}.tif", lap_s)

            meancurv_s = mean_curvature(dem_s, px)
            meancurv_s = _apply_valid_mask(meancurv_s, global_mask_win)            
            write_feature(f"{stem}_mean_curvature_{smooth_tag}.tif", meancurv_s)

    # --- 開度（マルチスケール） ---
    if openness_mode == 3:
        print("\n  > 開度の計算はスキップされました。")
    elif openness_mode == 1:
        compute_openness_with_saga(
            saga_cmd_path,
            dem_path,
            out_dir,
            stem,
            openness_r_list,
            n_dirs,
            nodata,
            meta,
            px,
            global_mask_win,
        )
    elif openness_mode == 2:
        for r_open in openness_r_list:
            max_step = max(1, int(r_open / px))
            print(f"\n  > 開度 (Python 内蔵) R_open={r_open}m → 1〜{max_step} ピクセル先まで（stride={step_stride}) を評価")
            pos, neg = openness_pair(dem, px, r_open, n_dirs=n_dirs, step_stride=step_stride)
            # マスクも共通窓でそろえる
            pos_m = _apply_valid_mask(pos, global_mask_win)
            neg_m = _apply_valid_mask(neg, global_mask_win)

            rtag_open = int(round(r_open))
            tag_open = f"r{rtag_open}m_nd{n_dirs}dir"

            print("    - 地上開度 (openness_pos) ...")
            write_feature(f"{stem}_openness_pos_{tag_open}.tif", pos_m)
            print("    - 地下開度 (openness_neg) ...")
            write_feature(f"{stem}_openness_neg_{tag_open}.tif", neg_m)

    print("\n=== DONE: 全 10 指標（＋必要に応じてスムージング版勾配/曲率）の出力が完了しました。 ===")

if __name__ == "__main__":
    main()
