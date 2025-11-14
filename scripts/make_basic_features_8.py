"""
単一DEMから 8 つの地形特徴量を一括生成するスクリプト
（マルチスケール＋SAGA開度対応＋外縁マスク統一）

入力:
  - DEM (GeoTIFF, 北が上・等方ピクセル前提, 解像度 px [m])

出力:
  - out_dir/
      {stem}_relief_r{r}m.tif
      {stem}_stddev_r{r}m.tif
      {stem}_slope_deg.tif
      {stem}_aspect_deg.tif
      {stem}_laplacian.tif
      {stem}_mean_curvature.tif
      {stem}_openness_pos_r{r_open}m_nd{n_dirs}dir.tif
      {stem}_openness_neg_r{r_open}m_nd{n_dirs}dir.tif

r, r_open は物理半径 [m]。複数指定可（例: 2,5,10）。

----------------------------------------------------------------------
■ 出力される地形特徴量の定義
----------------------------------------------------------------------
1) 局所比高 (relief_r{R}m)
   - イメージ: k×k 窓に含まれる標高値の ”最大値”-”最小値”
   - 近傍: k×k 窓（8近傍を含む窓内全ピクセル）
   - 定義:
       R(i, j; k) = max_{(u, v) ∈ W_k(i, j)} z(u, v)
                  - min_{(u, v) ∈ W_k(i, j)} z(u, v)

2) 局所標準偏差 (stddev_r{R}m)
   - イメージ: k×k 窓に含まれる標高値の バラつき＝”標準偏差”
   - 近傍: k×k 窓（8近傍を含む窓内全ピクセル）
   - 定義:
       μ(i, j) = (1 / N) Σ_{(u, v) ∈ W_k} z(u, v)
       σ(i, j) = sqrt( (1 / N) Σ_{(u, v) ∈ W_k} z(u, v)^2 - μ(i, j)^2 )

     （N は窓内の有効セル数）

3) 傾斜角 (slope_deg)
   - イメージ: k×k 窓全体を斜平面とみなしたときの ”傾きの大きさ”
   - 近傍: 3×3 の 8近傍
   - Horn (1981) の 3×3 オペレータで gx, gy を求める:
       gx ≈ (1 / (8 px)) * [[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]] * z
       gy ≈ (1 / (8 px)) * [[ 1,  2,  1],
                            [ 0,  0,  0],
                            [-1, -2, -1]] * z
   - 傾斜角:
       slope(i, j) = atan( sqrt(gx^2 + gy^2) )  [rad] → [deg に変換]

4) 斜面方位 (aspect_deg)
   - イメージ: k×k 窓全体を斜平面とみなしたときの ”傾きの向き”
   - 近傍: 勾配と同じ 3×3 の 8近傍
   - 定義（0°=東, 90°=北, 反時計回り）:
       aspect(i, j) = atan2(gy, gx)  [rad] を 0〜360° に正規化

5) ラプラシアン (laplacian)
   - イメージ: 中心セルが、周囲セル（k×k 窓、中心セル除く）の平均からどれだけの ”浮き・沈みか”
   - 近傍: 3×3 の 8近傍
   - 8近傍ラプラシアンカーネル:
       ∇²z(i, j) ≈ (1 / px^2) * (
           z_{i-1,j-1} + z_{i,  j-1} + z_{i+1,j-1} +
           z_{i-1,j  } - 8 z_{i,  j  } + z_{i+1,j  } +
           z_{i-1,j+1} + z_{i,  j+1} + z_{i+1,j+1}
       )

6) 平均曲率 (mean_curvature)
   - イメージ: 中心セルが、k×k 窓の地表面どれだけ丸く ”凸凹しているか”（谷底ほど+、尾根の頂部ほど-、一直線の斜面は0付近）
   - 近傍: 主に 3×3 の 8近傍と、その延長方向の有限差分
   - r や k には依存せず、DEM の解像度 px[m] のみでスケールが決まる
   - 実装では np.gradient を用いて、連続場 z(x, y) の平均曲率 H を
     有限差分で近似している：

       zx  = ∂z/∂x,  zy  = ∂z/∂y
       zxx = ∂²z/∂x², zyy = ∂²z/∂y², zxy = ∂²z/∂x∂y

       H = [ (1 + zy²) zxx - 2 zx zy zxy + (1 + zx²) zyy ]
           / [ 2 (1 + zx² + zy²)^(3/2) ]

     ここで zx, zy, zxx, zyy, zxy は、上下左右およびその周辺セルの
     差分から求めており、「指定半径 r 内のリング」だけを使うような
     ドーナツ型の窓は用いていない。

7) 正開度 / 負開度 (openness_pos / openness_neg)
   - イメージ: 中心セルから周囲360度の設定する方向（スクリプト内では8 or 16）に向け距離rの放射線を飛ばす。
        どの方向でどれだけ見上げ（仰角：正開度）/見下ろす（俯角：負開度）の平均値
   - 近傍: 指定した方向数 n_dirs 本（例: 8 方向）の「レイ」。
           各レイは中心セルから最大距離 r_open [m] までを
           1 ピクセル刻み（Python 内蔵版では stride ピクセル刻み）でサンプリングする。
   - スケール:
       ・比高・標準偏差のような k×k 窓は使わず、
         「各方向に伸びる線分（レイ）上のセル列」を評価する。
       ・r_open（物理距離）と DEM 解像度 px によって、最大ステップ数
           max_step ≈ round(r_open / px)
         が決まり、その 1〜max_step ピクセル先までを使う。
       ・ドーナツ状に内側を除外するのではなく、
         「中心から外側に向かう全ステップ」を追いかけるイメージ。

   - 定義（中心 (x0, y0), 標高 z0、各方向 k、レイ上の点 j、水平距離 d_j）:
       上向き最大仰角（正開度用）:
         α_k = max_j atan( (z_j - z0) / d_j )

       下向き最大俯角（負開度用）:
         β_k = max_j atan( (z0 - z_j) / d_j )

       それぞれの方向ごとの角度を天頂角に変換し、方向平均したものを開度とする:
         Open_pos(i, j) = 90° - mean_k( α_k * 180/π )
         Open_neg(i, j) = 90° - mean_k( β_k * 180/π )

       （ここで d_j は px に応じて 1, 2, 3, ... ピクセル先の水平距離に対応。
         n_dirs は方向分解能を、Python 版の stride はサンプリング間隔を制御する。）

   - 具体例（px = 1 m の DEM の場合）:
       r_open = 1 m → 中心から 1 ピクセル先まで（1 ステップ）
       r_open = 2 m → 中心から 1〜2 ピクセル先まで（2 ステップ）
       r_open = 3 m → 中心から 1〜3 ピクセル先まで（3 ステップ）

     （SAGA 版では ta_lighting:5 に r_open と n_dirs をそのまま渡して計算。
       Python 内蔵版では上式に従って同様の定義で計算する。）     

----------------------------------------------------------------------
■ ピクセル解像度と窓サイズ k の関係
----------------------------------------------------------------------
DEM の水平解像度を px [m] とすると、ユーザが指定した半径 r[m] から
ローカル統計用の窓サイズ k[ピクセル] を

    k = round(r / px)
    k が偶数なら k ← k + 1   （必ず奇数にする）

とし、中心セルを含む k×k の正方形窓を用いる。
物理的な「有効半径」は

    R_phys ≈ ((k - 1) / 2) * px [m]

となる。

※ 比高 / 標準偏差では、この k×k 窓内の「全ての有効セル」を使う
   （リング状に間を飛ばす「ドーナツ窓」ではない）。

----------------------------------------------------------------------
■ r / r_open の指定と窓サイズの具体例
----------------------------------------------------------------------
例1: 1m DEM（px = 1）のとき

  k = round(r / 1), 偶数なら +1 なので:

  - r = 1m → k = 1  → 1×1（中心セルのみ）
  - r = 2m → k = 3  → 3×3
  - r = 4m → k = 5  → 5×5
  - r = 6m → k = 7  → 7×7

  → 1m 解像度で 3×3 の窓にしたい場合は **r = 2** を指定、
     5×5 にしたい場合は **r = 4**、7×7 は **r = 6** を指定する。

例2: 5m DEM（px = 5）のとき

  - r = 5m  → r/px = 1   → k = 1 → 1×1
  - r = 10m → r/px = 2   → k = 3 → 3×3（有効半径 ≒ 5m）
  - r = 20m → r/px = 4   → k = 5 → 5×5（有効半径 ≒ 10m）

  → おおまかには「r / px が 2,4,6,... になるように r を与えると、
     それぞれ 3×3, 5×5, 7×7 の窓になる」と考えてよい。

----------------------------------------------------------------------
■ r / r_open が効く指標と効かない指標
----------------------------------------------------------------------
- 比高 / 標準偏差:
    - ユーザが指定した r[m] から計算した k を使い、
      各セルまわりの k×k 窓に含まれる **全ての有効セル** の標高を使って
      最大値・最小値や平均・分散を計算する。
      内側を飛ばして外側だけを見るような「ドーナツ窓」にはならない。

- 正開度 / 負開度:
    - r_open[m] を「中心セルからの最大距離」として用い、
      各方向に 1, 2, 3, … ピクセル先のセルを順にたどる。
      各レイ上のセルを途中も含めてすべて評価し、その中の最大仰角・俯角から
      開度を算出する（円環の外周だけではなく、中心から外側までの線分全体を見る）。

- 勾配 / 斜面方位 / ラプラシアン / 平均曲率:
    - r および r_open には依存せず、常に DEM 上の **3×3 近傍**
      （中心セル＋周囲 8 セル）の標高だけを使って局所的な傾き・曲率を計算する。


----------------------------------------------------------------------
■ 各特徴量の定義と近傍
----------------------------------------------------------------------
z(i, j) : DEM の標高（行 j, 列 i）
W_k(i, j) : (i, j) を中心とする k×k 窓（有効セルのみを対象）

1) 局所比高 (relief_r{R}m)
   - 近傍: k×k 窓（8近傍を含む窓内全ピクセル）
   - 定義:
       R(i, j; k) = max_{(u, v) ∈ W_k(i, j)} z(u, v)
                  - min_{(u, v) ∈ W_k(i, j)} z(u, v)

2) 局所標準偏差 (stddev_r{R}m)
   - 近傍: k×k 窓（8近傍を含む窓内全ピクセル）
   - 定義:
       μ(i, j) = (1 / N) Σ_{(u, v) ∈ W_k} z(u, v)
       σ(i, j) = sqrt( (1 / N) Σ_{(u, v) ∈ W_k} z(u, v)^2 - μ(i, j)^2 )

     （N は窓内の有効セル数）

3) 勾配角 (slope_deg)
   - 近傍: 3×3 の 8近傍
   - Horn (1981) の 3×3 オペレータで gx, gy を求める:
       gx ≈ (1 / (8 px)) * [[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]] * z
       gy ≈ (1 / (8 px)) * [[ 1,  2,  1],
                            [ 0,  0,  0],
                            [-1, -2, -1]] * z
   - 勾配角:
       slope(i, j) = atan( sqrt(gx^2 + gy^2) )  [rad] → [deg に変換]

4) 斜面方位 (aspect_deg)
   - 近傍: 勾配と同じ 3×3 の 8近傍
   - 定義（0°=東, 90°=北, 反時計回り）:
       aspect(i, j) = atan2(gy, gx)  [rad] を 0〜360° に正規化

5) ラプラシアン (laplacian)
   - 近傍: 3×3 の 8近傍
   - 8近傍ラプラシアンカーネル:
       ∇²z(i, j) ≈ (1 / px^2) * (
           z_{i-1,j-1} + z_{i,  j-1} + z_{i+1,j-1} +
           z_{i-1,j  } - 8 z_{i,  j  } + z_{i+1,j  } +
           z_{i-1,j+1} + z_{i,  j+1} + z_{i+1,j+1}
       )

6) 平均曲率 (mean_curvature)
   - 近傍: 主に 3×3 の 8近傍と、その延長方向の有限差分
   - r や k には依存せず、DEM の解像度 px[m] のみでスケールが決まる
   - 実装では np.gradient を用いて、連続場 z(x, y) の平均曲率 H を
     有限差分で近似している：

       zx  = ∂z/∂x,  zy  = ∂z/∂y
       zxx = ∂²z/∂x², zyy = ∂²z/∂y², zxy = ∂²z/∂x∂y

       H = [ (1 + zy²) zxx - 2 zx zy zxy + (1 + zx²) zyy ]
           / [ 2 (1 + zx² + zy²)^(3/2) ]

     ここで zx, zy, zxx, zyy, zxy は、上下左右およびその周辺セルの
     差分から求めており、「指定半径 r 内のリング」だけを使うような
     ドーナツ型の窓は用いていない。

7) 正開度 / 負開度 (openness_pos / openness_neg)
   - 近傍: 指定した方向数 n_dirs 本（例: 8 方向）の「レイ」。
           各レイは中心セルから最大距離 r_open [m] までを
           1 ピクセル刻み（Python 内蔵版では stride ピクセル刻み）でサンプリングする。
   - スケール:
       ・比高・標準偏差のような k×k 窓は使わず、
         「各方向に伸びる線分（レイ）上のセル列」を評価する。
       ・r_open（物理距離）と DEM 解像度 px によって、最大ステップ数
           max_step ≈ round(r_open / px)
         が決まり、その 1〜max_step ピクセル先までを使う。
       ・ドーナツ状に内側を除外するのではなく、
         「中心から外側に向かう全ステップ」を追いかけるイメージ。

   - 定義（中心 (x0, y0), 標高 z0、各方向 k、レイ上の点 j、水平距離 d_j）:
       上向き最大仰角（正開度用）:
         α_k = max_j atan( (z_j - z0) / d_j )

       下向き最大俯角（負開度用）:
         β_k = max_j atan( (z0 - z_j) / d_j )

       それぞれの方向ごとの角度を天頂角に変換し、方向平均したものを開度とする:
         Open_pos(i, j) = 90° - mean_k( α_k * 180/π )
         Open_neg(i, j) = 90° - mean_k( β_k * 180/π )

       （ここで d_j は px に応じて 1, 2, 3, ... ピクセル先の水平距離に対応。
         n_dirs は方向分解能を、Python 版の stride はサンプリング間隔を制御する。）

   - 具体例（px = 1 m の DEM の場合）:
       r_open = 1 m → 中心から 1 ピクセル先まで（1 ステップ）
       r_open = 2 m → 中心から 1〜2 ピクセル先まで（2 ステップ）
       r_open = 3 m → 中心から 1〜3 ピクセル先まで（3 ステップ）

     （SAGA 版では ta_lighting:5 に r_open と n_dirs をそのまま渡して計算。
       Python 内蔵版では上式に従って同様の定義で計算する。）

----------------------------------------------------------------------
■ 方針
----------------------------------------------------------------------
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
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter, convolve



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


def _horn_gradients(arr, px: float):
    """
    Horn (1981) の 3x3 オペレータで 8近傍を使った勾配を計算する。
    gx: 東向き（+x）方向の上り勾配
    gy: 北向き（+y）方向の上り勾配
    """
    arr = arr.astype(np.float32)
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
    # 現状の定義を維持（0=東, 90=北, 反時計回り）
    aspect = (np.degrees(np.arctan2(gy, gx)) + 360.0) % 360.0
    aspect[~np.isfinite(arr)] = np.nan
    return aspect.astype(np.float32)


def laplacian(arr, px: float):
    """ラプラシアン ∇²z（8近傍版）"""
    arr = arr.astype(np.float32)
    mask = ~np.isfinite(arr)

    # 8近傍ラプラシアンカーネル
    k_lap = np.array([[1, 1, 1],
                      [1,-8, 1],
                      [1, 1, 1]], dtype=np.float32) / (px * px)

    out = convolve(arr, k_lap, mode="nearest").astype(np.float32)
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

# =============== SAGA の開度計算を確認 ===============

def confirm_saga_openness_tool(saga_cmd_path: str, tool_id: int = 5) -> bool:
    """
    SAGA の ta_lighting モジュール一覧を表示し、
    tool_id 番が "Topographic Openness" になっているかを確認する。

    True ならそのまま 5 番を使う。
    False なら Python 内蔵にフォールバックするなどを呼び出し側で判断。
    """
    print("\n[SAGA] ta_lighting モジュールの一覧を確認します...")
    try:
        # -h をつけてヘルプを出す（returncode は無視する）
        res = subprocess.run(
            [saga_cmd_path, "ta_lighting", "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as e:
        print(f"  [ERROR] saga_cmd ta_lighting の実行に失敗しました: {e}")
        return False

    # ★ returncode は気にしないで、とにかく出力を見に行く
    text = (res.stdout or "") + "\n" + (res.stderr or "")

    print("  ---- ta_lighting のツール一覧 ----")
    print(res.stdout)

    expected_label = "Topographic Openness"
    key = f"[{tool_id}]"     # ← ここを "(5)" ではなく "[5]" に
    ok = False
    for line in text.splitlines():
        if key in line and expected_label in line:
            ok = True
            break

    if ok:
        print(f"  → [{tool_id}] {expected_label} を開度計算に使用します。")
        ans = input("    この内容で問題なければ Enter（変更したい場合は n）: ").strip().lower()
        if ans in ("n", "no"):
            return False
        return True
    else:
        print(f"  [WARN] ta_lighting {tool_id} に '{expected_label}' が見つかりません。")
        ans = input("    このまま ta_lighting 5 を使うのではなく、Python 内蔵開度に切り替えますか？ [Y/n]: ").strip().lower()
        return False if ans in ("", "y", "yes") else True

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

                # SAGA 側の NODATA 値（だいたい -99999）
                saga_nodata = src_saga.nodata
                if saga_nodata is None:
                    saga_nodata = -99999.0  # SAGA のデフォルトに合わせる

                # 1) SAGA の NODATA を NaN に
                data = np.where(data == saga_nodata, np.nan, data)

                # 2) NaN / inf を、このスクリプトで使う nodata に統一
                data = np.where(np.isfinite(data), data, nodata).astype(np.float32)

                meta2 = meta.copy()
                meta2.update(
                    width=src_saga.width,
                    height=src_saga.height,
                    transform=src_saga.transform,
                    crs=src_saga.crs,
                    nodata=nodata,  # 念のためここも明示
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

        # ★ SAGA を選んだときだけ ta_lighting 5 を確認する
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

    # 比高・偏差：複数半径
    relief_r_list = ask_float_list(
        "比高・偏差の計算範囲 r[m]（カンマ区切り可, 例: 2,5,10)"
        "\n  例) 1mDEMで r=2 → 3×3窓, r=5 → 11×11窓",
        [5.0],
    )

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

    print("\n[DONE] 全 8 指標（マルチスケール）の出力が完了しました。")


if __name__ == "__main__":
    main()
