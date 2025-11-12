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
            vals = [float(x) for x in s.split(",") if x.strip() != ""]
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
    """方位角 [deg, 0〜360)"""
    gy, gx = np.gradient(arr, px, px)
    aspect = np.degrees(np.arctan2(-gx, gy))  # 北=0, 東=90...
    aspect = np.mod(aspect, 360.0)
    aspect[~np.isfinite(arr)] = np.nan
    return aspect.astype(np.float32)


def laplacian(arr, px: float):
    """ラプラシアン（2次微分の和）"""
    gy, gx = np.gradient(arr, px, px)
    gyy, gyx = np.gradient(gy, px, px)
    gxy, gxx = np.gradient(gx, px, px)
    lap = gxx + gyy
    lap[~np.isfinite(arr)] = np.nan
    return lap.astype(np.float32)


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


def _unit_dirs(n_dirs: int):
    """0〜πの半円方向に n_dirs 本の単位ベクトル"""
    thetas = np.linspace(0.0, math.pi, n_dirs, endpoint=False)
    return np.cos(thetas), np.sin(thetas)


def openness_pair(dem: np.ndarray, res: float, radius_m: float,
                  n_dirs: int = 8, step_stride: int = 1):
    """
    地上開度/地下開度を同時に計算（視線角ベースのシンプル版）。

    定義は Yokoyama et al. (2002) の正/負開度と同じで、

      1. 各方向ごとに、中心点から半径 L までの「最大仰角」を求める
      2. その仰角を 90° から引いた「天頂角」を求める
      3. 天頂角を方向平均したものを開度（正/負）とする

    ここでは仰角を直接足し合わせた平均から 90° を引いているだけで、
    数学的には同等。
    """
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")

    h, w = dem.shape
    nodata_mask = ~np.isfinite(dem)

    max_step = max(1, int(radius_m / res))
    # step_stride = 1 → 1,2,3,...step_max
    # step_stride = 2 → 2,4,6,...
    # step_stride = 3 → 3,6,9,...
    step_indices = np.arange(1, max_step + 1, step_stride, dtype=int)
    if step_indices.size == 0:
        step_indices = np.array([1], dtype=int)

    dx_dirs, dy_dirs = _unit_dirs(n_dirs)

    pos = np.zeros_like(dem, dtype=np.float32)
    neg = np.zeros_like(dem, dtype=np.float32)

    for y0 in range(h):
        for x0 in range(w):
            z0 = dem[y0, x0]
            if not np.isfinite(z0):
                pos[y0, x0] = np.nan
                neg[y0, x0] = np.nan
                continue

            max_up = np.zeros(n_dirs, dtype=np.float32) - np.inf
            max_dn = np.zeros(n_dirs, dtype=np.float32) - np.inf

            for k_dir in range(n_dirs):
                dx = dx_dirs[k_dir]
                dy = dy_dirs[k_dir]

                for s in step_indices:
                    xx = x0 + dx * s
                    yy = y0 + dy * s
                    ix = int(round(xx))
                    iy = int(round(yy))
                    if ix < 0 or ix >= w or iy < 0 or iy >= h:
                        continue
                    z = dem[iy, ix]
                    if not np.isfinite(z):
                        continue

                    dist = math.hypot((ix - x0) * res, (iy - y0) * res)
                    if dist <= 0:
                        continue

                    au = math.atan((z - z0) / dist)      # 上向き（仰角）
                    ad = math.atan((z0 - z) / dist)      # 下向き（俯角）

                    if au > max_up[k_dir]:
                        max_up[k_dir] = au
                    if ad > max_dn[k_dir]:
                        max_dn[k_dir] = ad

            finite_up = np.isfinite(max_up)
            finite_dn = np.isfinite(max_dn)

            if not finite_up.any():
                pos[y0, x0] = np.nan
            else:
                mean_up = float(max_up[finite_up].mean()) * 180.0 / math.pi
                pos[y0, x0] = 90.0 - mean_up

            if not finite_dn.any():
                neg[y0, x0] = np.nan
            else:
                mean_dn = float(max_dn[finite_dn].mean()) * 180.0 / math.pi
                neg[y0, x0] = 90.0 - mean_dn

    pos[nodata_mask] = np.nan
    neg[nodata_mask] = np.nan
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

    - この関数は外部コマンド `saga_cmd` を subprocess で実行します。
    - このスクリプトを実行するシェル環境で `saga_cmd` に PATH が通っていることが前提です。
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
    print("=== DEM → 8特徴量 一括出力（マルチスケール＋SAGA開度対応＋外縁マスク統一） ===")

    dem_path_str = ask_path("入力DEM GeoTIFF のパス", must_exist=True)
    dem_path = Path(dem_path_str)

    default_out_dir = str(dem_path.with_suffix("").parent / (dem_path.stem + "_features"))
    out_dir_str = ask_path(f"出力フォルダ [{default_out_dir}]", must_exist=False, default=default_out_dir)
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    # DEM 読み込み
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is None:
            nodata = np.nan
        transform = src.transform
        if not isinstance(transform, Affine):
            transform = Affine(*transform)
        px = transform.a  # assume square pixels
        # 高さ方向の変換は無視（Z単位は m 前提）

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
    print(f"\n[INFO] DEM サイズ: {w} × {h} ピクセル, ピクセルサイズ: {px:.3f} m")

    # SAGA が使えるか判定
    saga_cmd_path = shutil.which("saga_cmd")
    has_saga = saga_cmd_path is not None

    # 比高・偏差：複数半径
    relief_r_list = ask_float_list(
        "比高・偏差の計算範囲 r[m]（カンマ区切り可, 例: 2,5,10)"
        "\n  例) 1mDEMで r=2 → 3×3窓, r=5 → 11×11窓",
        [5.0],
    )

    # ------------------------------
    # 開度の計算方法を先に決める
    # ------------------------------
    print("\n[開度の計算方法]")
    if has_saga:
        print(f"  saga_cmd が見つかりました: {saga_cmd_path}")
        print("  ※ このスクリプトを実行している環境で `saga_cmd` に PATH が通っていれば、そのまま利用できます。")
        print("  1: SAGA (ta_lighting 5) で開度を計算する（大規模DEM向け推奨）")
        print("  2: Python 内蔵の開度計算を使う（小〜中規模DEM向け）")
        print("  3: 開度は計算しない")
        openness_mode = ask_int("番号を選んでください", 1)
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

    # 開度パラメータ（必要なときだけ）
    openness_r_list = []
    n_dirs = None
    step_stride = 1

    if openness_mode != 3:
        # 評価範囲 L (= r_open) は SAGA / Python 共通
        openness_r_list = ask_float_list(
            "開度の最大距離 r_open[m]（カンマ区切り可, Enterで比高と同じ）"
            "\n  例) 1mDEMで r=2 → 1〜2ピクセル先まで, r=10 → 1〜10ピクセル先まで",
            relief_r_list,
        )

        n_dirs = ask_int(
            "開度の方向数 n_dirs（8 or 16 推奨）", 8
        )

        # サンプリング間隔は Python 内蔵開度のときだけ聞く
        if openness_mode == 2:
            step_stride = ask_int(
                "開度のサンプリング間隔（ピクセル, Python 内蔵のみ）"
                "\n  1: 1,2,3,...ピクセルごと / 2: 2,4,6,... / 3: 3,6,9,...",
                1,
            )

            # どう解釈されるかを明示
            print("\n  > 開度 (Python 内蔵) の評価設定")
            for r_open in openness_r_list:
                n_px = max(1, int(round(float(r_open) / float(px))))
                print(
                    f"    - r_open={r_open}m → 1〜{n_px} ピクセル先まで"
                    f"（stride={step_stride} ピクセル刻み）を評価"
                )
        elif openness_mode == 1:
            # SAGA モードの評価設定も軽く表示
            print("\n  > 開度 (SAGA ta_lighting 5) の評価設定")
            for r_open in openness_r_list:
                n_px = max(1, int(round(float(r_open) / float(px))))
                print(
                    f"    - r_open={r_open}m → 1〜{n_px} ピクセル先まで"
                    f"（n_dirs={n_dirs} 方向）の最大天頂角を平均"
                )

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
        write_feature(f"{dem_path.stem}_relief_r{rtag}m.tif", relief)

        print("    - 標準偏差 (stddev) ...")
        stddev = local_stddev(dem, win_pix_rel)
        stddev = _apply_valid_mask(stddev, win_pix_rel)
        write_feature(f"{dem_path.stem}_stddev_r{rtag}m.tif", stddev)

    # --- 勾配・方位 ---
    print("\n  > 勾配 (slope_deg) ...")
    slope = slope_deg(dem, px)
    slope = _apply_valid_mask(slope, 3)
    write_feature(f"{dem_path.stem}_slope_deg.tif", slope)

    print("  > 方位 (aspect_deg) ...")
    aspect = aspect_deg(dem, px)
    aspect = _apply_valid_mask(aspect, 3)
    write_feature(f"{dem_path.stem}_aspect_deg.tif", aspect)

    # --- ラプラシアン・平均曲率 ---
    print("  > ラプラシアン (laplacian) ...")
    lap = laplacian(dem, px)
    lap = _apply_valid_mask(lap, 3)
    write_feature(f"{dem_path.stem}_laplacian.tif", lap)

    print("  > 平均曲率 (mean_curvature) ...")
    meancurv = mean_curvature(dem, px)
    meancurv = _apply_valid_mask(meancurv, 3)
    write_feature(f"{dem_path.stem}_mean_curvature.tif", meancurv)

    # --- 開度（マルチスケール） ---
    if openness_mode == 3:
        print("\n  > 開度の計算はスキップされました。")
    elif openness_mode == 1:
        # SAGA 版
        compute_openness_with_saga(
            saga_cmd_path,
            dem_path,
            out_dir,
            dem_path.stem,
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
            write_feature(f"{dem_path.stem}_openness_pos_{tag_open}.tif", pos_m)
            print("    - 地下開度 (openness_neg) ...")
            write_feature(f"{dem_path.stem}_openness_neg_{tag_open}.tif", neg_m)

    print("\n[DONE] 全 8 指標（マルチスケール）の出力が完了しました。」")


if __name__ == "__main__":
    main()
