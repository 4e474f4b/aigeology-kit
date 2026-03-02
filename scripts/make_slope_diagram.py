#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Horn (1981) 3×3 傾斜計算の説明図を生成するスクリプト
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

# --- フォント設定（日本語対応） ---
plt.rcParams["font.family"] = ["Hiragino Sans", "Hiragino Kaku Gothic Pro", "Yu Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# === 配色 ===
COL_BG     = "#F8F9FA"
COL_CENTER = "#2C7BB6"   # 中心セル（注目ピクセル）
COL_EAST   = "#D7191C"   # 東西方向に関係するセル
COL_NORTH  = "#1A9641"   # 南北方向に関係するセル
COL_CORNER = "#F4A582"   # 四隅（両方向に関係）
COL_TEXT   = "#1A1A2E"
COL_FORMULA_BG = "#EFF3FF"
COL_ARROW  = "#555555"

# ===================================================
# セル位置と Horn 重み
# ===================================================
#  z1(NW) z2(N)  z3(NE)
#  z4(W)  z5(C)  z6(E)
#  z7(SW) z8(S)  z9(SE)
#
# gx = [(z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)] / (8·Δx)
# gy = [(z1 + 2*z2 + z3) - (z7 + 2*z8 + z9)] / (8·Δx)

CELL_LABELS = [
    ("z₁\n(NW)", -1,  1),
    ("z₂\n(N)",   0,  1),
    ("z₃\n(NE)",  1,  1),
    ("z₄\n(W)",  -1,  0),
    ("z₅\n(C)",   0,  0),
    ("z₆\n(E)",   1,  0),
    ("z₇\n(SW)", -1, -1),
    ("z₈\n(S)",   0, -1),
    ("z₉\n(SE)",  1, -1),
]

# gx に対する各セルの重み符号  (+正 / −負 / 0なし)
GX_WEIGHTS = {
    "z₁": -1, "z₂": 0, "z₃": +1,
    "z₄": -2, "z₅": 0, "z₆": +2,
    "z₇": -1, "z₈": 0, "z₉": +1,
}
# gy に対する各セルの重み符号
GY_WEIGHTS = {
    "z₁": +1, "z₂": +2, "z₃": +1,
    "z₄":  0, "z₅":  0, "z₆":  0,
    "z₇": -1, "z₈": -2, "z₉": -1,
}

def cell_color(label_key):
    """セルの役割に応じた背景色"""
    if label_key == "z₅":
        return COL_CENTER
    gx = GX_WEIGHTS.get(label_key, 0)
    gy = GY_WEIGHTS.get(label_key, 0)
    if gx != 0 and gy != 0:
        return COL_CORNER
    elif gx != 0:
        return COL_EAST
    elif gy != 0:
        return COL_NORTH
    else:
        return "#CCCCCC"


# ===================================================
# 図全体レイアウト
# ===================================================
fig = plt.figure(figsize=(14, 9), facecolor=COL_BG)
fig.suptitle(
    "傾斜（Slope）の計算方法 ― Horn (1981) 3×3 加重微分オペレータ",
    fontsize=16, fontweight="bold", color=COL_TEXT, y=0.98
)

gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    left=0.04, right=0.98,
    top=0.90, bottom=0.05,
    wspace=0.35, hspace=0.45,
)

# -------------------------------------------------------
# [左上] 3×3 DEMウィンドウ図
# -------------------------------------------------------
ax_dem = fig.add_subplot(gs[0, 0])
ax_dem.set_xlim(-1.6, 1.6)
ax_dem.set_ylim(-1.6, 1.6)
ax_dem.set_aspect("equal")
ax_dem.axis("off")
ax_dem.set_title("① 3×3 DEM ウィンドウ", fontsize=12, fontweight="bold", color=COL_TEXT, pad=8)

CS = 0.9  # セルサイズの半分

for label, cx, cy in CELL_LABELS:
    key = label.split("\n")[0]
    fc = cell_color(key)
    rect = mpatches.FancyBboxPatch(
        (cx - CS/2, cy - CS/2), CS, CS,
        boxstyle="round,pad=0.04",
        facecolor=fc, edgecolor="white", linewidth=2.5, alpha=0.88,
    )
    ax_dem.add_patch(rect)
    txt_color = "white" if fc in (COL_CENTER, COL_EAST, COL_NORTH) else COL_TEXT
    ax_dem.text(cx, cy, label, ha="center", va="center",
                fontsize=10, color=txt_color, fontweight="bold", linespacing=1.4)

# ピクセルサイズ矢印（Δx）
ax_dem.annotate(
    "", xy=(0.5, -1.55), xytext=(-0.5, -1.55),
    arrowprops=dict(arrowstyle="<->", color=COL_ARROW, lw=1.5),
)
ax_dem.text(0, -1.53, "Δx（ピクセルサイズ）", ha="center", va="top",
            fontsize=8, color=COL_ARROW)

# 凡例
legend_items = [
    mpatches.Patch(facecolor=COL_CENTER, label="注目ピクセル（中心）"),
    mpatches.Patch(facecolor=COL_EAST,   label="東西勾配に使用"),
    mpatches.Patch(facecolor=COL_NORTH,  label="南北勾配に使用"),
    mpatches.Patch(facecolor=COL_CORNER, label="両方向に使用（四隅）"),
]
ax_dem.legend(
    handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, -0.14),
    fontsize=8, framealpha=0.9, ncol=2,
)


# -------------------------------------------------------
# [中央上] x方向（東西）重みカーネル
# -------------------------------------------------------
ax_kx = fig.add_subplot(gs[0, 1])
ax_kx.set_xlim(-0.2, 3.2)
ax_kx.set_ylim(-0.2, 3.2)
ax_kx.set_aspect("equal")
ax_kx.axis("off")
ax_kx.set_title("② 東西勾配カーネル（kx）", fontsize=12, fontweight="bold", color=COL_TEXT, pad=8)

KX = [[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]
KX_COLORS = [
    [COL_EAST, "#EEEEEE", COL_EAST],
    [COL_EAST, "#EEEEEE", COL_EAST],
    [COL_EAST, "#EEEEEE", COL_EAST],
]

for row in range(3):
    for col in range(3):
        fc = KX_COLORS[row][col]
        val = KX[row][col]
        rect = mpatches.FancyBboxPatch(
            (col, 2 - row), 1, 1,
            boxstyle="round,pad=0.05",
            facecolor=fc, edgecolor="white", linewidth=2,
        )
        ax_kx.add_patch(rect)
        txt = f"{val:+d}" if val != 0 else "0"
        txt_color = "white" if fc == COL_EAST else "#888888"
        ax_kx.text(col + 0.5, 2 - row + 0.5, txt,
                   ha="center", va="center", fontsize=16, fontweight="bold", color=txt_color)

ax_kx.text(1.5, -0.15, "÷ (8 · Δx)", ha="center", va="top", fontsize=11,
           color=COL_TEXT, style="italic")

# 方向ラベル
ax_kx.text(0.5, 3.1, "西\n(負)", ha="center", va="bottom", fontsize=9, color=COL_EAST, fontweight="bold")
ax_kx.text(1.5, 3.1, "0",       ha="center", va="bottom", fontsize=9, color="#888888")
ax_kx.text(2.5, 3.1, "東\n(正)", ha="center", va="bottom", fontsize=9, color=COL_EAST, fontweight="bold")

ax_kx.annotate("", xy=(2.9, 1.5), xytext=(0.1, 1.5),
               arrowprops=dict(arrowstyle="-|>", color=COL_EAST, lw=2))
ax_kx.text(3.15, 1.5, "東+", ha="left", va="center", fontsize=9, color=COL_EAST, fontweight="bold")


# -------------------------------------------------------
# [右上] y方向（南北）重みカーネル
# -------------------------------------------------------
ax_ky = fig.add_subplot(gs[0, 2])
ax_ky.set_xlim(-0.2, 3.2)
ax_ky.set_ylim(-0.2, 3.2)
ax_ky.set_aspect("equal")
ax_ky.axis("off")
ax_ky.set_title("③ 南北勾配カーネル（ky）", fontsize=12, fontweight="bold", color=COL_TEXT, pad=8)

KY = [[+1, +2, +1], [0, 0, 0], [-1, -2, -1]]
KY_COLORS = [
    [COL_NORTH, COL_NORTH, COL_NORTH],
    ["#EEEEEE",  "#EEEEEE",  "#EEEEEE" ],
    [COL_NORTH, COL_NORTH, COL_NORTH],
]

for row in range(3):
    for col in range(3):
        fc = KY_COLORS[row][col]
        val = KY[row][col]
        rect = mpatches.FancyBboxPatch(
            (col, 2 - row), 1, 1,
            boxstyle="round,pad=0.05",
            facecolor=fc, edgecolor="white", linewidth=2,
        )
        ax_ky.add_patch(rect)
        txt = f"{val:+d}" if val != 0 else "0"
        txt_color = "white" if fc == COL_NORTH else "#888888"
        ax_ky.text(col + 0.5, 2 - row + 0.5, txt,
                   ha="center", va="center", fontsize=16, fontweight="bold", color=txt_color)

ax_ky.text(1.5, -0.15, "÷ (8 · Δx)", ha="center", va="top", fontsize=11,
           color=COL_TEXT, style="italic")

ax_ky.text(3.15, 2.5, "北\n(正)", ha="left", va="center", fontsize=9, color=COL_NORTH, fontweight="bold")
ax_ky.text(3.15, 1.5, "0",       ha="left", va="center", fontsize=9, color="#888888")
ax_ky.text(3.15, 0.5, "南\n(負)", ha="left", va="center", fontsize=9, color=COL_NORTH, fontweight="bold")

ax_ky.annotate("", xy=(1.5, 2.9), xytext=(1.5, 0.1),
               arrowprops=dict(arrowstyle="-|>", color=COL_NORTH, lw=2))
ax_ky.text(1.5, 3.15, "北+", ha="center", va="bottom", fontsize=9, color=COL_NORTH, fontweight="bold")


# -------------------------------------------------------
# [下段全幅] 数式パネル
# -------------------------------------------------------
ax_formula = fig.add_subplot(gs[1, :])
ax_formula.set_xlim(0, 1)
ax_formula.set_ylim(0, 1)
ax_formula.axis("off")
ax_formula.set_facecolor(COL_FORMULA_BG)

# 背景ボックス
formula_bg = mpatches.FancyBboxPatch(
    (0.01, 0.02), 0.98, 0.94,
    boxstyle="round,pad=0.01",
    facecolor=COL_FORMULA_BG, edgecolor="#AABBDD", linewidth=1.5,
    transform=ax_formula.transAxes,
)
ax_formula.add_patch(formula_bg)

ax_formula.set_title("④ 計算式まとめ", fontsize=12, fontweight="bold", color=COL_TEXT, pad=6)

# 数式テキスト配置
STEP_Y = [0.82, 0.60, 0.38, 0.14]

steps = [
    (
        "Step 1  東西・南北方向の勾配を計算",
        r"$g_x = \frac{(z_3 + 2z_6 + z_9) - (z_1 + 2z_4 + z_7)}{8 \cdot \Delta x}$"
        r"     "
        r"$g_y = \frac{(z_1 + 2z_2 + z_3) - (z_7 + 2z_8 + z_9)}{8 \cdot \Delta x}$",
    ),
    (
        "Step 2  合成勾配（勾配の大きさ）を計算",
        r"$|\nabla z| = \sqrt{g_x^2 + g_y^2}$",
    ),
    (
        "Step 3  傾斜角（度）に変換",
        r"$\mathrm{Slope} = \arctan\!\left(\sqrt{g_x^2 + g_y^2}\right) \times \frac{180}{\pi}$",
    ),
    (
        "ポイント：四隅（NW/NE/SW/SE）は重み 1、辺中央（N/S/E/W）は重み 2 — 平均ではなく加重差分",
        None,
    ),
]

for i, (label, formula) in enumerate(steps):
    y = STEP_Y[i]
    if formula is None:
        # 注記
        ax_formula.text(
            0.5, y, label,
            ha="center", va="center",
            fontsize=10, color="#555566",
            style="italic",
            transform=ax_formula.transAxes,
        )
    else:
        ax_formula.text(
            0.02, y + 0.07, label,
            ha="left", va="center",
            fontsize=10, fontweight="bold", color=COL_TEXT,
            transform=ax_formula.transAxes,
        )
        ax_formula.text(
            0.5, y - 0.02, formula,
            ha="center", va="center",
            fontsize=13, color=COL_TEXT,
            transform=ax_formula.transAxes,
        )

# ===================================================
# 出力
# ===================================================
out_path = "/Users/ngok/dev/aigeology-kit/scripts/horn_slope_diagram.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COL_BG)
print(f"[OK] 保存しました: {out_path}")
plt.show()
