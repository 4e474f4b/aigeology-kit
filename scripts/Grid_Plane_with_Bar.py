#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3×3 グリッドの値から最小二乗平面（平均的な斜面）をフィットし、
その平面からのズレを棒で可視化する図を描く。

- 灰色の板: 3×3 のマス（平面 z = a x + b y + c）
- 各マスの中心から棒が生える
    - 平面より高い: 赤い棒が「面から上」に伸びる
    - 平面より低い: 青い棒が「面から下」に伸びる（上端は必ず面上）
    - ほぼ一致   : 黒（わずかに描く）

出力: slope_plane_with_bars.png
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def main():
    # ========================
    # 1. 3x3 の標高データ
    # ========================
    #   5   3   2
    #   4   0  -2
    #   2  -3  -2
    values = np.array([
        [5, 3, 2],
        [4, 0, -2],
        [2, -3, -2]
    ], dtype=float)

    ny, nx = values.shape  # (3, 3)

    # セル中心の座標 (0,1,2)
    cx, cy = np.meshgrid(np.arange(nx), np.arange(ny))
    Xc = cx.ravel()
    Yc = cy.ravel()
    Z = values.ravel()

    # ========================
    # 2. 平面フィット z = a x + b y + c
    # ========================
    A = np.c_[Xc, Yc, np.ones_like(Xc)]
    (a, b, c), *_ = np.linalg.lstsq(A, Z, rcond=None)
    print(f"a={a:.3f}, b={b:.3f}, c={c:.3f}")

    # セル中心での平面の高さ
    Z_plane_center = (a * cx + b * cy + c)
    Zp_center = Z_plane_center.ravel()

    # ========================
    # 3. 棒の配置（面から上 / 面から下）
    # ========================
    # セル幅は 1 とみなし、棒は各セルの中に収まるよう 0.8 幅にする
    dx = dy = 0.8
    xpos = Xc - dx / 2.0
    ypos = Yc - dy / 2.0

    zpos = []   # 棒の始点（足）
    dz = []     # 棒の長さ（常に正の値）
    colors = []

    for z_val, zp_val in zip(Z, Zp_center):
        if z_val > zp_val + 1e-6:
            # 平面より高い → 面から上へ
            zpos.append(zp_val)
            dz.append(z_val - zp_val)
            colors.append("red")
        elif z_val < zp_val - 1e-6:
            # 平面より低い → 面から下へ（上端は平面上）
            zpos.append(z_val)
            dz.append(zp_val - z_val)
            colors.append("blue")
        else:
            # ほぼ平面上
            zpos.append(zp_val)
            dz.append(0.02)  # ほんの少しだけ見える程度
            colors.append("black")

    zpos = np.array(zpos)
    dz = np.array(dz)

    # ========================
    # 4. 平面（3×3 マス）の描画
    # ========================
    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    # 3×3 マスになるように、境界を -0.5, 0.5, 1.5, 2.5 にとる
    xv, yv = np.meshgrid(
        np.linspace(-0.5, 2.5, 4),
        np.linspace(-0.5, 2.5, 4)
    )
    zv = a * xv + b * yv + c

    ax.plot_surface(
        xv, yv, zv,
        color="lightgray",
        alpha=0.8,
        edgecolor="k",   # マス目の線
        linewidth=0.5,
        rstride=1,
        cstride=1,
    )

    # 棒を描画
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)

    # 視点（傾きが分かりやすいように斜め上から）
    ax.view_init(elev=30, azim=-45)

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("slope_plane_with_bars.png", transparent=True)
    plt.show()


if __name__ == "__main__":
    main()
