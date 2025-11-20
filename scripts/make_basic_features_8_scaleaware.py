"""
単一 DEM から 8 つの地形特徴量を一括生成するスクリプト
（マルチスケール＋SAGA 開度対応＋外縁マスク統一＋勾配/曲率スムージング＋R↔R_open スケール整合）

ファイル名:
  make_basic_features_8_scaleaware.py

想定用途:
  - 1 枚の DEM から、地形分類用の「基本 8 特徴量」を一括生成する。
  - 比高・標準偏差・開度については、ユーザが指定したスケール R[m]／R_open[m] に応じて
    「どの空間スケールの地形（谷・尾根・盆地など）を見たいか」を制御できる。
  - 勾配・斜面方位・ラプラシアン・平均曲率は、Horn(1981) 型の 3×3 オペレータによる
    ピクセルレベルの指標としつつ、任意スケールの平滑化 DEM から計算した派生版も出力できる。
  - R と R_open のスケール整合（面積等価 / 内接円 / 外接円）を選択可能。

入力:
  - DEM (GeoTIFF, 北が上・等方ピクセル前提, 解像度 px [m])

    * スクリプト内では、水平解像度を px[m] として扱う。
    * 対話モードで r（比高/偏差の半径）や R_open（開度の最大距離）を
      「m 指定」または「ピクセル数指定」で入力できるが、
      内部ではすべて物理スケール [m] に変換して扱う。

出力:
  - out_dir/
  - out_dir/
      {stem}_relief_r{R_eff}m.tif
      {stem}_stddev_r{R_eff}m.tif
      {stem}_slope_deg.tif
      {stem}_aspect_deg.tif
      {stem}_laplacian.tif
      {stem}_mean_curvature.tif
      {stem}_openness_pos_r{R_open}m_nd{n_dirs}dir.tif
      {stem}_openness_neg_r{R_open}m_nd{n_dirs}dir.tif

    ＋（オプション: 勾配・曲率のスムージング版）
      {stem}_slope_deg_r{R_s}m_box.tif
      {stem}_aspect_deg_r{R_s}m_box.tif
      {stem}_laplacian_r{R_s}m_box.tif
      {stem}_mean_curvature_r{R_s}m_box.tif

      {stem}_slope_deg_r{R_s}m_gauss.tif
      {stem}_aspect_deg_r{R_s}m_gauss.tif
      {stem}_laplacian_r{R_s}m_gauss.tif
      {stem}_mean_curvature_r{R_s}m_gauss.tif

  - R, R_open, R_s は「物理スケール [m]」としてファイル名に反映される。
    ただし比高 / 標準偏差については、ファイル名の rXXm は
      「ユーザ指定 R[m] から決まる窓の有効半径 R_eff[m]」
    を用いる（R_eff≒((k-1)/2)*px）。
----------------------------------------------------------------------
■ このスクリプトで出力される 8 指標
----------------------------------------------------------------------

  1) 局所比高 (relief_r{R_eff}m)
  2) 局所標準偏差 (stddev_r{R_eff}m)
  3) 勾配角 (slope_deg)
  4) 斜面方位 (aspect_deg)
  5) ラプラシアン (laplacian)
  6) 平均曲率 (mean_curvature)
  7) 正開度 (openness_pos_r{R_open}m_nd{n_dirs}dir)
  8) 負開度 (openness_neg_r{R_open}m_nd{n_dirs}dir)

  - 1), 2), 7), 8) はユーザ指定の空間スケール R[m], R_open[m] に依存。
  - 3)〜6) は Horn(1981) 型の 3×3 オペレータ（ピクセルレベル）で計算する
    「基本版」に加え、任意スケールで平滑化した DEM から計算した派生版
    （R_smooth = R_s[m]）もオプションで出力可能。

----------------------------------------------------------------------
■ ピクセル解像度と窓サイズ k の関係
----------------------------------------------------------------------

DEM の水平解像度を px[m] とし、比高・標準偏差の半径 R[m] を指定したとき、
ローカル統計用の窓サイズ k[ピクセル] を

    k = round(R / px)
    k が偶数なら k ← k + 1   （中心セルを持つよう必ず奇数にする）

とする。これにより、中心セル (i, j) を中心とする k×k の正方形窓 W_k(i, j) を用いる。
物理的な「有効半径」は

    R_eff ≈ ((k - 1) / 2) * px   [m]

となる。

※ 比高 / 標準偏差は、この k×k 窓内の「全ての有効セル」を使う。
   リング状に内側を飛ばす「ドーナツ窓」は用いない。

----------------------------------------------------------------------
■ 局所比高 / 局所標準偏差 の定義
----------------------------------------------------------------------

1) 局所比高 (relief_r{R_eff}m)
   - イメージ: k×k 窓に含まれる標高値の「最大値 − 最小値」
   - 近傍: k×k 窓（8近傍を含む窓内全ピクセル）
   - 定義:
       R(i, j; k) = max_{(u, v) ∈ W_k(i, j)} z(u, v)
                  - min_{(u, v) ∈ W_k(i, j)} z(u, v)

2) 局所標準偏差 (stddev_r{R_eff}m)
   - イメージ: k×k 窓に含まれる標高値の「ばらつき」
   - 近傍: k×k 窓（8近傍を含む窓内全ピクセル）
   - 定義:
       μ(i, j) = (1 / N) Σ_{(u, v) ∈ W_k(i, j)} z(u, v)
       σ(i, j) = sqrt( (1 / N) Σ_{(u, v) ∈ W_k(i, j)} z(u, v)^2 - μ(i, j)^2 )

     （N は窓内の有効セル数）

----------------------------------------------------------------------
■ 勾配角 / 斜面方位 / ラプラシアン / 平均曲率 の定義
  （R や R_open に依存しない「ピクセルレベル」の指標）
----------------------------------------------------------------------

3) 勾配角 (slope_deg)
   - イメージ: 3×3 近傍全体を 1 枚の斜平面とみなしたときの「傾きの大きさ」
   - 近傍: 3×3 の 8近傍
   - Horn (1981) の 3×3 オペレータで gx, gy を求める:
       gx ≈ (1 / (8 px)) * [[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]] * z
       gy ≈ (1 / (8 px)) * [[ 1,  2,  1],
                            [ 0,  0,  0],
                            [-1, -2, -1]] * z
   - 勾配角:
       slope(i, j) = atan( sqrt(gx^2 + gy^2) )  [rad] を [deg] に変換

4) 斜面方位 (aspect_deg)
   - イメージ: 3×3 近傍全体を 1 枚の斜平面とみなしたときの「傾きの向き」
   - 近傍: 勾配と同じ 3×3 の 8近傍
   - 定義（0°=東, 90°=北, 反時計回り）:
       aspect(i, j) = atan2(gy, gx)  [rad] を 0〜360° に正規化

5) ラプラシアン (laplacian)
   - イメージ: 中心セルが、周囲セル（3×3 窓, 中心セル除く）の平均から
                どれだけ「浮き・沈み」しているか
   - 近傍: 3×3 の 8近傍
   - 8近傍ラプラシアンカーネル:
       ∇²z(i, j) ≈ (1 / px^2) * (
           z_{i-1,j-1} + z_{i,  j-1} + z_{i+1,j-1} +
           z_{i-1,j  } - 8 z_{i,  j  } + z_{i+1,j  } +
           z_{i-1,j+1} + z_{i,  j+1} + z_{i+1,j+1}
       )

6) 平均曲率 (mean_curvature)
   - イメージ: 中心セルまわりの地表がどれだけ丸く「凸/凹」しているか
                （谷底では +、尾根では −、一直線の斜面では 0 に近づく）
   - 近傍: 主に 3×3 の 8近傍と、その延長方向の有限差分
   - R や R_open には依存せず、DEM の解像度 px[m] のみでスケールが決まる。
   - 実装では np.gradient を用いて、連続場 z(x, y) の平均曲率 H を
     有限差分で近似している：

       zx  = ∂z/∂x,  zy  = ∂z/∂y
       zxx = ∂²z/∂x², zyy = ∂²z/∂y², zxy = ∂²z/∂x∂y

       H = [ (1 + zy²) zxx - 2 zx zy zxy + (1 + zx²) zyy ]
           / [ 2 (1 + zx² + zy²)^(3/2) ]

     ここで zx, zy, zxx, zyy, zxy は、上下左右およびその周辺セルの
     差分から求めており、「指定半径 R 内のリング」だけを使うような
     ドーナツ型の窓は用いていない。

  - スムージング版の勾配・曲率は、
      ・DEM を移動平均（ボックスフィルタ） / ガウシアンフィルタで平滑化
      → その平滑化 DEM に対して上記 3)〜6) と同じ式を適用
    して求める。
    （平滑化スケール R_s[m] も対話的に指定可能）

----------------------------------------------------------------------
■ 開度（正開度 / 負開度）と R_open の定義
  （R_open[m] に依存する「方向レイ型」の指標）
----------------------------------------------------------------------

開度（Topographic Openness）は、中心セル (i, j) に対して

   「中心から各方向へ伸ばしたレイ上の標高を、最大距離 R_open[m] の範囲で
     見上げ角／見下ろし角として評価し、それを全方向で平均した角度指標」

である。

DEM の水平分解能を px[m] とし、開度の最大探索半径を R_open[m] とすると、
レイの最大ステップ数 r_max は

    r_max = floor(R_open / px)

となる。

----------------------------------------------------------------------
● 方向分割とレイ（近傍の取り方）
----------------------------------------------------------------------

全周を n_dirs 本の方向に等分し、方向 d（d = 1…n_dirs）の方位角を θ_d とする。
各方向 d について、中心セル (i, j) から最大 r_max ピクセル先まで

    (x_d(r), y_d(r)) = (i + r cosθ_d,  j + r sinθ_d)   （r = 1…r_max）

に従ってレイを伸ばす。水平距離は

    L(r) = r * px   [m]

標高値 z_d(r) は（SAGA ta_lighting の場合は線分追跡、
Python 内蔵実装ではバイリニア補間）で取得する。

----------------------------------------------------------------------
● 仰角／俯角の定義
----------------------------------------------------------------------

中心セルの標高を z0 = z(i, j) とすると、方向 d, ステップ r における

    仰角（上向き）:  α_d(r) = arctan( (z_d(r) - z0) / L(r) )   [deg]
    俯角（下向き）:  β_d(r) = arctan( (z0 - z_d(r)) / L(r) )   [deg]

とする。方向 d における最大仰角／最大俯角を

    α_d^max = max_r α_d(r)
    β_d^max = max_r β_d(r)

と定義する。

----------------------------------------------------------------------
● 方向別の正開度／負開度
----------------------------------------------------------------------

方向 d における正開度・負開度は

    POS_d = 90° - α_d^max
    NEG_d = 90° - β_d^max

と定義する。

  - 正開度 POS_d が大きい   → 上方向に空が大きく見える（尾根・台地側）
  - 正開度 POS_d が小さい   → 周囲を崖で囲まれた谷底・凹地

  - 負開度 NEG_d が大きい   → 周囲に向かって落ち込む凸地形（尾根・崖頭）
  - 負開度 NEG_d が小さい   → 周囲から見下ろされる凹地形（谷・盆地）

----------------------------------------------------------------------
● 全方向平均としての開度
----------------------------------------------------------------------

中心セル (i, j) の正開度／負開度は

    openness_pos(i, j)
        = (1 / n_dirs) Σ_d POS_d
        = (1 / n_dirs) Σ_d (90° - α_d^max)

    openness_neg(i, j)
        = (1 / n_dirs) Σ_d NEG_d
        = (1 / n_dirs) Σ_d (90° - β_d^max)

とする。本スクリプトでは、この openness_pos / openness_neg を
GeoTIFF として出力する。

----------------------------------------------------------------------
■ R_open[m] の役割（R と同様「空間スケール」の指定）
----------------------------------------------------------------------

R_open はレイの最大探索距離であり、比高・標準偏差の R[m] と同じく

    「どの空間スケールの谷／尾根／盆地を見たいか」

を決めるパラメータである。

  - R_open が大きい → 広域の谷・尾根・盆地の“骨格”が強調
  - R_open が小さい → 小崖・小凹凸・微地形が強調

本スクリプトでは

    R（比高・偏差の半径）, R_open（開度の最大距離）, R_s（平滑化スケール）

のすべてを「物理スケール [m]」として扱う。

----------------------------------------------------------------------
■ R と R_open の“空間スケール整合”の幾何学的対応
----------------------------------------------------------------------

比高・標準偏差は k×k の正方形窓 W_k(i, j) を用い、その物理的な有効半径を

    R_eff = ((k - 1) / 2) * px   [m]

とする。

開度（円形レイ）でこの R_eff と同等のスケールを扱いたい場合、
本スクリプトでは以下 3 方式から R_open の決め方を選択できる。

  1) 面積等価円（デフォルト）
       - 正方形窓（辺長 ≒ 2 R_eff）の面積 A_sq と、同じ面積をもつ円の半径 R_eq を
         等しくする。
           A_sq ≒ (2 R_eff)^2
           π R_eq^2 = A_sq
           → R_eq = R_eff * sqrt(π/2) ≒ 1.2533 R_eff

       - スクリプト上では、
           R_open ≒ R_eff * sqrt(π/2)
         として自動設定する。

  2) 内接円半径（最小スケール）
       - 正方形窓の内接円半径をとる。
           R_open = R_eff

       - 比高・標準偏差の「実効半径 R_eff」と真っ直ぐ対応するため
         最も保守的な（やや小さめの）スケール整合となる。

  3) 外接円半径（最大スケール）
       - 正方形窓の外接円半径をとる。
           R_open = R_eff * sqrt(2)

       - 正方形の対角線方向まで含めて完全にカバーするスケール。
         面積等価円よりもやや大きなスケールを見に行く設定となる。

対話プロンプトでは

  - 「R_open を手動指定する」
  - 「R_eff と整合する R_open を自動算出する（面積等価 / 内接円 / 外接円）」

を選択できる。

----------------------------------------------------------------------
■ 外縁マスクの扱い
----------------------------------------------------------------------

  - 比高・標準偏差:
      必要な k×k 窓が完全に取れない外縁は NODATA（共通外縁マスク）とする。

  - 開度:
      レイが指定 R_open[m] まで到達できないセル（DEM 外縁付近）は、
      正しい角度評価ができないため NODATA とする。
      R_open が大きいほど有効領域は内側に狭くなる。

  - 勾配・斜面方位・ラプラシアン・平均曲率（基本版）:
      3×3 オペレータに応じて、1 ピクセル幅の外縁を NODATA とする。

  - スムージング版の勾配・曲率:
      平滑化カーネルの有効半径 R_s[m] と 3×3 オペレータを合成した
      近似的な有効窓サイズに応じて外縁をマスクする。

このようにして、全ての出力レイヤで「外縁における NODATA の扱い」を
できるだけ整合的にそろえる。
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

            z00 = dem[y0, x0]
            z10 = dem[y0, x1]
            z01 = dem[y1, x0]
            z11 = dem[y1, x1]

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
    print("\n[SAGA] ta_lighting モジュールの一覧を確認します...")
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
    print("=== DEM → 8特徴量 一括出力（マルチスケール＋SAGA開度対応＋外縁マスク統一） ===")

    dem_path_str = ask_path("入力DEM GeoTIFF のパス", must_exist=True)
    dem_path = Path(dem_path_str)
    stem = dem_path.stem

    default_out_dir = str(dem_path.with_suffix("").parent / (stem + "_features"))
    out_dir_str = ask_path(f"出力フォルダ [{default_out_dir}]", must_exist=False, default=default_out_dir)
    out_dir = Path(out_dir_str)
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
    print(f"\n[INFO] DEM サイズ: {w} × {h} ピクセル, ピクセルサイズ: {px:.3f} m")

    # SAGA が使えるか判定
    saga_cmd_path = shutil.which("saga_cmd")
    has_saga = saga_cmd_path is not None

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
    print("  r の指定方法を選んでください:")
    print("    1: 距離 [m] で指定（例: 2,5,10）")
    print("    2: ピクセル数 [px] で指定（例: 3,5,11）")
    r_mode = ask_int("番号を選んでください", 1)

    if r_mode == 1:
        relief_r_list = ask_float_list(
            "比高・偏差の計算範囲 R[m]（カンマ区切り可, 例: 2,5,10)\n"
            "  例) 1mDEMで R=2 → 3×3窓, R=5 → 11×11窓",
            [5.0],
        )
    else:
        px_list = ask_int_list(
            "比高・偏差の計算範囲（ピクセル数）r_px（カンマ区切り可, 例: 3,7,15）",
            [3, 7],
        )
        relief_r_list = [p * px for p in px_list]

        print("\n  → ピクセル数から換算した r [m]:")
        for p, R in zip(px_list, relief_r_list):
            print(f"    {p} px → r ≒ {R:.3f} m")

    # R_open_match の求め方（外接円 / 面積等価円 / 内接円）を選択
    print("  1: 外接円（正方形窓の外接円） R_open = √2 × R_eff")
    print("  2: 面積等価円（k×k窓と同面積） R_open = (2/√π) × R_eff")
    print("  3: 内接円（正方形窓の内接円） R_open = R_eff")
    match_mode = ask_int("番号を選んでください", 1)

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

    print("\n[INFO] 指定した R[m] から窓サイズ k と R_open_match を計算します。")
    win_pix_list = []
    r_eff_list = []
    r_open_match_list = []
    for R in relief_r_list:
        k = _winpix_from_meters(R, px, min_win=1, odd=True)
        win_pix_list.append(k)
        R_eff = ((k - 1) / 2.0) * px
        r_eff_list.append(R_eff)
        R_open_match = coeff * R_eff
        r_open_match_list.append(R_open_match)
        print(
            f"  R={R:.3f}m → k={k} (窓サイズ {k}×{k}), "
            f"R_eff≒{R_eff:.3f}m, R_open_match≒{R_open_match:.3f}m"
        )

    print(
        "\n[注意] R が DEM ピクセルサイズに対して大きくなるほど、"
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

        print("\n[注意] 開度 R_open の解釈について"
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
                f"\n[WARN] DEM の総ピクセル数は約 {total_px/1e6:.1f} 百万ピクセルです。"
                "\n       Python 内蔵の開度計算は非常に時間がかかる可能性があります。"
                "\n       必要であれば SAGA の利用や解像度ダウンサンプリングも検討してください。"
            )

    # ------------------------------
    # 勾配・曲率系のスムージング設定
    # ------------------------------
    print("\n[勾配・曲率系のスムージング設定]\n")
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
        k_box = _winpix_from_meters(Rsm, px, min_win=3, odd=True)  # ≈ 2*Rsm/px + 1
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
            #   例）比高 R が1つ（R=6m）のとき: guide_sets = [[2.0, 1.0]]
            #       比高 R が2つ（R=6m, 15m）のとき: guide_sets = [[2.0, 1.0], [5.0, 3.0]]
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
                    # 1スケールのみ指定された場合は、そのスケールの先頭値のみを提示
                    default_ms = [guide_sets[0][0]]
                else:
                    # 複数スケールの場合は、各スケールの先頭値だけを並べて提示
                    default_ms = [g[0] for g in guide_sets]

            smooth_r_list = ask_float_list(
                "平滑化スケール R_smooth[m]（カンマ区切り可, 例: 10,30,50）\n"
                "    ※ここで指定する R_smooth は「平滑化の有効半径[m]」です。\n"
                "      上で表示された『推奨候補（m）』のうち先頭の値を選ぶと、\n"
                "      比高 R と勾配・曲率の平滑化が、ほぼ同じ空間スケール\n"
                "      （窓幅 ≒ 比高の R）を見るように調整されています。\n"
                "      （例：2mDEM で R=6m のとき、R_smooth=2m → 3×3窓（幅≒6m））",
                default_ms,
            )
        else:
            px_list_sm = ask_int_list(
                "平滑化スケール（ピクセル数）r_px_smooth（カンマ区切り可, 例: 5,11,21）\n"
                "    ※ここで指定する r_px_smooth は「有効半径 R_smooth をピクセル単位にした値」です。\n"
                "      Box の場合: R_smooth≒r_px_smooth×px となり、窓サイズは概ね k≒2*r_px_smooth+1。\n"
                "      Gauss の場合: R_smooth≒r_px_smooth×px とみなし、3σ≒R_smooth → σ_px≒r_px_smooth/3。",
                [ _winpix_from_meters((guide_sets[0][0] if guide_sets else 10.0), px, min_win=3, odd=True) ],
            )
            smooth_r_list = [p * px for p in px_list_sm]
            print("\n  → ピクセル数から換算した R_smooth [m]:")
            for p, R in zip(px_list_sm, smooth_r_list):
                print(f"    {p} px → R_smooth ≒ {R:.3f} m")

        print("\n[INFO] スムージング版の勾配/曲率は、指定した R_smooth を")
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
    print(f"\n[INFO] 外縁マスクは共通窓 k={global_mask_win} を使用します。")

    def write_feature(name: str, data: np.ndarray):
        arr = data.astype(np.float32)
        arr = np.where(np.isfinite(arr), arr, nodata).astype(np.float32)
        out_path = out_dir / name
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(arr, 1)

    # --- 比高 / 標準偏差（マルチスケール） ---
    for R_in in relief_r_list:
        # R_in[m] から窓サイズ k と有効半径 R_eff[m] を再計算
        win_pix = _winpix_from_meters(R_in, px, min_win=3, odd=True)
        R_eff = ((win_pix - 1) / 2.0) * px

        print(
            f"\n  > 比高 / 標準偏差: 入力R={R_in:.3f}m, "
            f"k={win_pix}, 有効半径R_eff≒{R_eff:.3f}m ..."
        )

        rel = local_relief(dem, win_pix)
        # 外縁マスクは global_mask_win で他指標と共通化
        rel_m = _apply_valid_mask(rel, global_mask_win)
        tag = f"r{int(round(R_eff))}m"   # ファイル名は有効半径ベース
        write_feature(f"{stem}_relief_{tag}.tif", rel_m)

        std = local_stddev(dem, win_pix)
        std_m = _apply_valid_mask(std, global_mask_win)
        write_feature(f"{stem}_stddev_{tag}.tif", std_m)

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

    print("\n[DONE] 全 8 指標（＋必要に応じてスムージング版勾配/曲率）の出力が完了しました。")


if __name__ == "__main__":
    main()
