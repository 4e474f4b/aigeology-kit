#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
単一DEMから 8 つの地形特徴量を一括生成するスクリプト
（マルチスケール＋SAGA開度対応＋外縁マスク統一）

入力:
  - DEM (GeoTIFF, 北が上・等方ピクセル前提)

出力:
  - out_dir/
      {stem}_relief_r{R}m.tif
      {stem}_stddev_r{R}m.tif
      {stem}_slope_deg.tif
      {stem}_aspect_deg.tif
      {stem}_laplacian.tif
      {stem}_mean_curvature.tif
      {stem}_openness_pos_r{Ropen}m_nd{N}dir.tif
      {stem}_openness_neg_r{Ropen}m_nd{N}dir.tif

R, Ropen は複数指定可（例: 2,5,10）。

方針:
  - 近傍が完全に取れない外縁は、すべての特徴量で NODATA に統一。
  - 開度は SAGA (saga_cmd) / Python 内蔵 / スキップ を選択。
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
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter


# =============== ユーティリティ ===============

def ask_path(prompt, must_exist=True, default=None):
    while True:
        s = input(f"{prompt}{' [' + default + ']' if default else ''}: ").strip()
        if not s and default:
            s = default
        if not s:
            print("  入力してください。")
            continue
        s = os.path.expanduser(os.path.expandvars(s.strip().strip('"').strip("'")))
        p = Path(s)
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
            vals = [float(v) for v in s.split(",") if v.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except ValueError:
            print("  カンマ区切りの数値で入力してください（例: 2,5,10）。")


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


def _apply_valid_mask(arr, win_pix: int):
    """局所窓の半径ぶんだけ四辺を NaN（= NODATA）にする。"""
    rpx = max(0, (int(win_pix) - 1) // 2)
    out = arr.astype(np.float32, copy=True)
    if rpx > 0:
        out[:rpx, :] = np.nan
        out[-rpx:, :] = np.nan
        out[:, :rpx] = np.nan
        out[:, -rpx:] = np.nan
    return out


# =============== 地形演算（配列） ===============

def local_relief_range(arr, win_pix: int):
    """局所最大値 − 局所最小値"""
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


def local_stddev(arr, win_pix: int):
    """局所標準偏差"""
    valid = np.isfinite(arr)
    a = np.where(valid, arr, 0.0)
    k = int(max(1, win_pix))
    mean = uniform_filter(a, size=k, mode="nearest")
    mean2 = uniform_filter(a * a, size=k, mode="nearest")
    cnt = uniform_filter(valid.astype(np.float64), size=k, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, mean / cnt, np.nan)
        mean2 = np.where(cnt > 0, mean2 / cnt, np.nan)
        var = mean2 - mean * mean
    var[var < 0] = 0.0
    out = np.sqrt(var).astype(np.float32)
    out[~valid] = np.nan
    return out


def slope_deg(arr, px: float):
    """勾配角 [deg]"""
    gy, gx = np.gradient(arr, px, px)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    slope[~np.isfinite(arr)] = np.nan
    return slope.astype(np.float32)


def aspect_deg(arr, px: float):
    """斜面方位 [deg]（0=東, 90=北）"""
    gy, gx = np.gradient(arr, px, px)
    aspect = (np.degrees(np.arctan2(gy, gx)) + 360.0) % 360.0
    aspect[~np.isfinite(arr)] = np.nan
    return aspect.astype(np.float32)


def laplacian(arr, px: float):
    """ラプラシアン ∇²z = z_xx + z_yy"""
    zy, zx = np.gradient(arr, px, px)      # 1階微分
    zyy, _ = np.gradient(zy, px, px)
    _, zxx = np.gradient(zx, px, px)
    out = (zxx + zyy).astype(np.float32)
    out[~np.isfinite(arr)] = np.nan
    return out


def mean_curvature(arr, px: float):
    """平均曲率 H"""
    zy, zx = np.gradient(arr, px, px)
    zyy, _ = np.gradient(zy, px, px)
    zxy, _ = np.gradient(zx, px, px)
    _, zxx = np.gradient(zx, px, px)

    denom = 2.0 * np.power(1.0 + zx * zx + zy * zy, 1.5)
    eps = np.finfo(np.float32).eps
    denom = np.where(denom < eps, eps, denom)

    num = (1.0 + zy * zy) * zxx - 2.0 * zx * zy * zxy + (1.0 + zx * zx) * zyy
    with np.errstate(invalid="ignore", divide="ignore"):
        H = num / denom
    H[~np.isfinite(arr)] = np.nan
    return H.astype(np.float32)


def _unit_dirs(n_dirs: int):
    """0〜πの半円方向に n_dirs 本の単位ベクトル"""
    thetas = np.linspace(0.0, math.pi, n_dirs, endpoint=False)
    return np.cos(thetas), np.sin(thetas)


def openness_pair(dem: np.ndarray, res: float, radius_m: float,
                  n_dirs: int = 16, step_stride: int = 1):
    """
    地上開度/地下開度を同時に計算（視線角ベースのシンプル版）。
    大きなDEMに対しては非常に重いので注意。
    """
    h, w = dem.shape
    dxu, dyu = _unit_dirs(n_dirs)
    max_step = max(1, int(radius_m / res))
    if step_stride < 1:
        step_stride = 1

    pos = np.full((h, w), np.nan, dtype=np.float32)
    neg = np.full((h, w), np.nan, dtype=np.float32)

    for y in range(h):
        for x in range(w):
            z0 = dem[y, x]
            if np.isnan(z0):
                continue
            acc_up = 0.0
            acc_dn = 0.0
            cnt = 0
            for k in range(n_dirs):
                max_alpha_up = -1e9
                max_alpha_dn = -1e9
                dxk, dyk = dxu[k], dyu[k]
                hit = False
                for s in range(step_stride, max_step + 1, step_stride):
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
                    if au > max_alpha_up:
                        max_alpha_up = au
                    if ad > max_alpha_dn:
                        max_alpha_dn = ad
                    hit = True
                if hit:
                    if max_alpha_up > -1e8:
                        acc_up += max_alpha_up
                    if max_alpha_dn > -1e8:
                        acc_dn += max_alpha_dn
                    cnt += 1
            if cnt > 0:
                mean_up = (acc_up / cnt) * 180.0 / math.pi
                mean_dn = (acc_dn / cnt) * 180.0 / math.pi
                pos[y, x] = 90.0 - mean_up
                neg[y, x] = 90.0 - mean_dn

    return pos, neg


# =============== SAGA で開度 ===============

def compute_openness_with_saga(
    saga_cmd_path: str,
    dem_path: Path,
    out_dir: Path,
    stem: str,
    openness_r_list,
    n_dirs: int,
    nodata: float,
    meta,
):
    """
    SAGA の Tool 'Topographic Openness' (ta_lighting 5) を使って
    正/負開度を計算し、GeoTIFF に書き出す。
    """
    for r_open in openness_r_list:
        rtag_open = int(round(r_open))
        tag_open = f"r{rtag_open}m_nd{n_dirs}dir"

        # SAGA の中間ファイル（SAGA Grid）
        pos_grid = out_dir / f"{stem}_openness_pos_{tag_open}.sdat"
        neg_grid = out_dir / f"{stem}_openness_neg_{tag_open}.sdat"

        print(f"\n  > SAGA で開度 r_open={r_open}m を計算中...")
        cmd = [
            saga_cmd_path,
            "ta_lighting",
            "5",  # Topographic Openness
            "-DEM", str(dem_path),
            "-POS", str(pos_grid),
            "-NEG", str(neg_grid),
            "-RADIUS", str(float(r_open)),   # map units = m 前提
            "-DIRECTIONS", "1",              # all directions
            "-NDIRS", str(int(n_dirs)),
            "-METHOD", "1",                  # line tracing
            "-UNIT", "1",                    # degree
            "-NADIR", "1",                   # difference from nadir
        ]

        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except FileNotFoundError:
            print("  [ERROR] saga_cmd が実行できませんでした。パス/環境変数を確認してください。")
            return

        if res.returncode != 0:
            print("  [ERROR] saga_cmd 実行中にエラーが発生しました。")
            print("  ---- stdout ----")
            print(res.stdout)
            print("  ---- stderr ----")
            print(res.stderr)
            return

        # SAGA Grid (.sdat) を読み込んで GeoTIFF に書き出し
        def convert_saga_grid_to_gtiff(grid_path: Path, out_tif_name: str):
            out_tif_path = out_dir / out_tif_name
            with rasterio.open(grid_path) as src_saga:
                data = src_saga.read(1).astype(np.float32)
                data = np.where(np.isfinite(data), data, nodata).astype(np.float32)
                meta2 = meta.copy()
                meta2.update(
                    width=src_saga.width,
                    height=src_saga.height,
                    transform=src_saga.transform,
                    crs=src_saga.crs,
                )
                with rasterio.open(out_tif_path, "w", **meta2) as dst:
                    dst.write(data, 1)
            print(f"    [OK] {out_tif_path}")

        convert_saga_grid_to_gtiff(
            pos_grid,
            f"{stem}_openness_pos_{tag_open}.tif",
        )
        convert_saga_grid_to_gtiff(
            neg_grid,
            f"{stem}_openness_neg_{tag_open}.tif",
        )


# =============== メイン処理 ===============

def main():
    print("\n=== DEM → 8特徴量 一括出力（マルチスケール＋SAGA開度対応＋外縁マスク統一） ===")

    dem_path_str = ask_path("入力DEM GeoTIFF のパス")
    dem_path = Path(dem_path_str)
    stem = dem_path.stem

    default_out = str(dem_path.with_name(stem + "_features"))
    out_dir_str = ask_path("出力フォルダ", must_exist=False, default=default_out)
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    # DEM 読み込み
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        transform: Affine = src.transform
        crs = src.crs
        px_x = transform.a
        px_y = -transform.e

        if not np.isfinite(px_x) or not np.isfinite(px_y) or px_x <= 0 or px_y <= 0:
            print("ERROR: ピクセルサイズが不正です。")
            sys.exit(1)

        if abs(px_x - px_y) > 1e-6:
            print(f"[WARN] x/y ピクセルサイズが異なります: {px_x} vs {px_y}")
        px = float((px_x + px_y) / 2.0)

        nodata = src.nodata
        if nodata is None:
            nodata = -9999.0

        dem = np.where(dem == nodata, np.nan, dem)
        dem = np.where(np.isfinite(dem), dem, np.nan)

        meta = src.meta.copy()
        meta.update(
            dtype="float32",
            nodata=nodata,
            compress="DEFLATE",
            predictor=3,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            count=1,
        )

    h, w = dem.shape
    print(f"\n[INFO] DEM サイズ: {w} × {h} ピクセル, ピクセルサイズ: {px:.3f} m")

    # SAGA が使えるか判定
    saga_cmd_path = shutil.which("saga_cmd")
    has_saga = saga_cmd_path is not None

    # 比高・偏差：複数半径
    relief_r_list = ask_float_list(
        "比高・偏差の計算範囲 r[m]（カンマ区切り可, 例: 2,5,10）"
        "\n  例) 1mDEMで r=2 → 3×3窓, r=5 → 11×11窓",
        [5.0],
    )

    # 開度：複数半径（デフォルトは比高と同じ）
    openness_r_list = ask_float_list(
        "開度の最大距離 r_open[m]（カンマ区切り可, Enterで比高と同じ）"
        "\n  例) 1mDEMで r=2 → 1〜2ピクセル先まで, r=10 → 1〜10ピクセル先まで",
        relief_r_list,
    )

    n_dirs = ask_int(
        "開度の方向数 n_dirs（8 or 16 推奨）", 8
    )

    step_stride = ask_int(
        "開度のサンプリング間隔（ピクセル）"
        "\n  1: 1,2,3,...ピクセルごと / 2: 2,4,6,... / 3: 3,6,9,...",
        1,
    )

    # 開度計算モード選択
    print("\n[開度の計算方法]")
    if has_saga:
        print(f"  saga_cmd が見つかりました: {saga_cmd_path}")
        print("  1: SAGA (ta_lighting 5) で開度を計算する（大規模DEM向け推奨）")
        print("  2: Python 内蔵の開度計算を使う（小〜中規模DEM向け）")
        print("  3: 開度は計算しない")
        openness_mode = ask_int("番号を選んでください", 1)
    else:
        print("  saga_cmd が見つかりませんでした（PATH 未設定 or SAGA 未インストール）。")
        print("  1: Python 内蔵の開度計算を使う（小〜中規模DEM向け）")
        print("  2: 開度は計算しない")
        openness_mode_raw = ask_int("番号を選んでください", 1)
        if openness_mode_raw == 1:
            openness_mode = 2  # Python 内蔵
        else:
            openness_mode = 3  # スキップ

    # Python 内蔵開度でのサイズ警告
    if openness_mode == 2:
        total_px = h * w
        if total_px > 50_000_000:  # 適当な閾値
            print(
                f"\n[WARN] DEM の総ピクセル数は約 {total_px/1e6:.1f} 百万ピクセルです。"
                "\n       Python 内蔵の開度計算は非常に時間がかかる可能性があります。"
                "\n       必要であれば SAGA の利用や解像度ダウンサンプリングも検討してください。"
            )

    def write_feature(name: str, data: np.ndarray):
        arr = data.astype(np.float32)
        arr = np.where(np.isfinite(arr), arr, nodata).astype(np.float32)
        out_path = out_dir / name
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(arr, 1)
        print(f"[OK] {out_path}")

    # --- 比高・偏差（マルチスケール） ---
    for r in relief_r_list:
        win_pix_rel = _winpix_from_meters(r, px, min_win=1, odd=True)
        print(f"\n  > 比高・偏差 r={r}m → 窓サイズ {win_pix_rel}×{win_pix_rel} ピクセル")
        if win_pix_rel == 1:
            print("    [WARN] win_pix=1 のため relief/stddev はほぼ0になります。")

        print("    - 比高 (relief) ...")
        relief = local_relief_range(dem, win_pix_rel)
        relief = _apply_valid_mask(relief, win_pix_rel)
        rtag = int(round(r))
        write_feature(f"{stem}_relief_r{rtag}m.tif", relief)

        print("    - 偏差 (stddev) ...")
        stddev = local_stddev(dem, win_pix_rel)
        stddev = _apply_valid_mask(stddev, win_pix_rel)
        write_feature(f"{stem}_stddev_r{rtag}m.tif", stddev)

    # --- 傾斜・方位・ラプラシアン・平均曲率（単スケール） ---
    # 勾配・曲率は実質 3×3 近傍で決まるので、外周1ピクセルはNODATAに統一する

    print("\n  > 傾斜 (slope_deg) ...")
    slope = slope_deg(dem, px)
    slope = _apply_valid_mask(slope, 3)  # 3×3窓想定 → 外周1ピクセル NaN → NODATA
    write_feature(f"{stem}_slope_deg.tif", slope)

    print("  > 斜面方位 (aspect_deg) ...")
    aspect = aspect_deg(dem, px)
    aspect = _apply_valid_mask(aspect, 3)
    write_feature(f"{stem}_aspect_deg.tif", aspect)

    print("  > ラプラシアン (laplacian) ...")
    lap = laplacian(dem, px)
    lap = _apply_valid_mask(lap, 3)
    write_feature(f"{stem}_laplacian.tif", lap)

    print("  > 平均曲率 (mean_curvature) ...")
    meancurv = mean_curvature(dem, px)
    meancurv = _apply_valid_mask(meancurv, 3)
    write_feature(f"{stem}_mean_curvature.tif", meancurv)

    # --- 開度（マルチスケール） ---
    if openness_mode == 3:
        print("\n  > 開度の計算はスキップされました。")
    elif openness_mode == 1:
        # SAGA 版
        compute_openness_with_saga(
            saga_cmd_path,
            dem_path,
            out_dir,
            stem,
            openness_r_list,
            n_dirs,
            nodata,
            meta,
        )
    elif openness_mode == 2:
        # Python 版
        for r_open in openness_r_list:
            max_step = max(1, int(r_open / px))
            print(f"\n  > 開度 (Python 内蔵) r_open={r_open}m → 1〜{max_step} ピクセル先まで（stride={step_stride}) を評価")
            pos, neg = openness_pair(dem, px, r_open, n_dirs=n_dirs, step_stride=step_stride)
            win_pix_open = _winpix_from_meters(r_open, px, min_win=1, odd=True)
            pos_m = _apply_valid_mask(pos, win_pix_open)
            neg_m = _apply_valid_mask(neg, win_pix_open)

            rtag_open = int(round(r_open))
            tag_open = f"r{rtag_open}m_nd{n_dirs}dir"

            print("    - 地上開度 (openness_pos) ...")
            write_feature(f"{stem}_openness_pos_{tag_open}.tif", pos_m)
            print("    - 地下開度 (openness_neg) ...")
            write_feature(f"{stem}_openness_neg_{tag_open}.tif", neg_m)

    print("\n[DONE] 全 8 指標（マルチスケール）の出力が完了しました。")


if __name__ == "__main__":
    main()
