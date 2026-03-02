#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3×3（生DEM）と 9×9（平滑化DEM）の傾斜計算を比較する説明図
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

plt.rcParams["font.family"] = ["Hiragino Sans", "Hiragino Kaku Gothic Pro", "Yu Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# === 配色 ===
COL_BG      = "#F8F9FA"
COL_CENTER  = "#2C7BB6"
COL_EAST    = "#D7191C"
COL_NORTH   = "#1A9641"
COL_CORNER  = "#F4A582"
COL_SMOOTH  = "#9B59B6"   # 平滑化ウィンドウ
COL_SMOOTH_L= "#E8D5F5"   # 平滑化ウィンドウ（淡色）
COL_TEXT    = "#1A1A2E"
COL_ARROW   = "#555555"
COL_3X3_BG  = "#EFF6FB"
COL_9X9_BG  = "#F5EFF9"

# ======================================================
# ヘルパ：1セルを描く
# ======================================================
def draw_cell(ax, cx, cy, size, fc, ec="white", lw=2, alpha=0.9, text=None,
              txt_color="white", fontsize=9, fontweight="bold"):
    s = size
    rect = mpatches.FancyBboxPatch(
        (cx - s/2, cy - s/2), s, s,
        boxstyle="round,pad=0.03",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha,
    )
    ax.add_patch(rect)
    if text is not None:
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fontsize, color=txt_color, fontweight=fontweight)


# ======================================================
# 図全体
# ======================================================
fig = plt.figure(figsize=(16, 10), facecolor=COL_BG)
fig.suptitle(
    "傾斜（Slope）の計算：3×3（生DEM）vs 9×9（平滑化DEM） ― 解像度 1 m/pixel",
    fontsize=15, fontweight="bold", color=COL_TEXT, y=0.99,
)

gs = gridspec.GridSpec(
    1, 2,
    figure=fig,
    left=0.03, right=0.97,
    top=0.92, bottom=0.05,
    wspace=0.08,
)

# ======================================================
# 左パネル：3×3 生DEM 版
# ======================================================
ax_l = fig.add_subplot(gs[0, 0])
ax_l.set_facecolor(COL_3X3_BG)
ax_l.set_xlim(-0.5, 9.5)
ax_l.set_ylim(-1.5, 13.0)
ax_l.set_aspect("equal")
ax_l.axis("off")
ax_l.set_title("【A】3×3 ウィンドウ（生DEM）\n　→ 1 m スケールの細かい傾斜",
               fontsize=12, fontweight="bold", color=COL_TEXT, pad=6, loc="center")

# --- DEMグリッド 9×9（背景）---
CS = 0.85
GRID_N = 9
for row in range(GRID_N):
    for col in range(GRID_N):
        cx = col * 1.0 + 0.5
        cy = (GRID_N - 1 - row) * 1.0 + 4.5
        # 中心3×3 を強調
        in_center = (3 <= row <= 5) and (3 <= col <= 5)
        if in_center:
            fc = "#DDEEFF"
        else:
            fc = "#F0F0F0"
        draw_cell(ax_l, cx, cy, CS, fc=fc, ec="#CCCCCC", lw=1, alpha=1.0)

# 9×9 全体の枠
rect_all = mpatches.FancyBboxPatch(
    (0.07, 4.07), 8.86, 8.86,
    boxstyle="round,pad=0.0",
    facecolor="none", edgecolor="#AAAAAA", linewidth=1.5, linestyle="--",
)
ax_l.add_patch(rect_all)
ax_l.text(4.5, 13.05, "DEM（9×9 表示）", ha="center", va="bottom",
          fontsize=9, color="#888888")

# 3×3 ウィンドウの枠
rect_3x3 = mpatches.FancyBboxPatch(
    (3.07, 7.07), 2.86, 2.86,
    boxstyle="round,pad=0.0",
    facecolor="none", edgecolor=COL_CENTER, linewidth=3.0,
)
ax_l.add_patch(rect_3x3)
ax_l.text(4.5, 10.1, "← 3×3 window（3 m × 3 m）→", ha="center", va="bottom",
          fontsize=9, color=COL_CENTER, fontweight="bold")

# 3×3 内のセルにラベル付け
labels_3x3 = [
    ["z₁", "z₂", "z₃"],
    ["z₄", "z₅", "z₆"],
    ["z₇", "z₈", "z₉"],
]
colors_3x3 = [
    [COL_CORNER, COL_NORTH, COL_CORNER],
    [COL_EAST,   COL_CENTER, COL_EAST  ],
    [COL_CORNER, COL_NORTH, COL_CORNER],
]
for ri in range(3):
    for ci in range(3):
        cx = (3 + ci) * 1.0 + 0.5
        cy = (GRID_N - 1 - (3 + ri)) * 1.0 + 4.5
        draw_cell(ax_l, cx, cy, CS, fc=colors_3x3[ri][ci], ec="white", lw=2,
                  text=labels_3x3[ri][ci], txt_color="white", fontsize=9)

# ↓ 矢印（処理フロー）
ax_l.annotate("", xy=(4.5, 3.6), xytext=(4.5, 4.0),
              arrowprops=dict(arrowstyle="-|>", color=COL_ARROW, lw=2.5))

# Horn 3×3 カーネルボックス
ax_l.text(4.5, 3.45, "Horn 3×3 オペレータ",
          ha="center", va="top", fontsize=10, fontweight="bold", color=COL_TEXT)

kx_vals = [[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]
for ri in range(3):
    for ci in range(3):
        cx = (ci + 3) * 0.65 + 1.5
        cy = 2.5 - ri * 0.65
        v = kx_vals[ri][ci]
        fc = COL_EAST if v != 0 else "#DDDDDD"
        txt = f"{v:+d}" if v != 0 else "0"
        tc = "white" if v != 0 else "#999999"
        draw_cell(ax_l, cx, cy, 0.60, fc=fc, ec="white", lw=1.5,
                  text=txt, txt_color=tc, fontsize=9)

ax_l.text(3.3, 1.75, "kx", ha="center", va="center", fontsize=9, color=COL_EAST,
          fontweight="bold")
ax_l.text(5.0, 1.75, "÷(8·Δx)", ha="left", va="center", fontsize=9, color=COL_TEXT)

# ↓ 矢印
ax_l.annotate("", xy=(4.5, 0.5), xytext=(4.5, 1.0),
              arrowprops=dict(arrowstyle="-|>", color=COL_ARROW, lw=2.5))

# 結果ボックス
result_box = mpatches.FancyBboxPatch(
    (1.5, -1.1), 6.0, 1.4,
    boxstyle="round,pad=0.1",
    facecolor=COL_CENTER, edgecolor="white", linewidth=2, alpha=0.9,
)
ax_l.add_patch(result_box)
ax_l.text(4.5, -0.35, "Slope（傾斜角）= arctan(√(gx²＋gy²))",
          ha="center", va="center", fontsize=10, color="white", fontweight="bold")

# スケール注記
ax_l.text(4.5, -1.3, "有効スケール：約 3 m × 3 m（= 3×3 window）",
          ha="center", va="top", fontsize=9, color=COL_CENTER,
          style="italic")


# ======================================================
# 右パネル：9×9 平滑化DEM 版
# ======================================================
ax_r = fig.add_subplot(gs[0, 1])
ax_r.set_facecolor(COL_9X9_BG)
ax_r.set_xlim(-0.5, 9.5)
ax_r.set_ylim(-1.5, 13.0)
ax_r.set_aspect("equal")
ax_r.axis("off")
ax_r.set_title("【B】9×9 ウィンドウ（平滑化DEM）\n　→ 9 m スケールのなだらかな傾斜",
               fontsize=12, fontweight="bold", color=COL_TEXT, pad=6, loc="center")

# --- STEP 1: 9×9 移動平均 ---
# DEMグリッド全体
for row in range(GRID_N):
    for col in range(GRID_N):
        cx = col * 1.0 + 0.5
        cy = (GRID_N - 1 - row) * 1.0 + 4.5
        draw_cell(ax_r, cx, cy, CS, fc=COL_SMOOTH_L, ec=COL_SMOOTH, lw=1, alpha=0.6)

# 9×9 全体を平滑化ウィンドウとして強調
rect_9x9 = mpatches.FancyBboxPatch(
    (0.07, 4.07), 8.86, 8.86,
    boxstyle="round,pad=0.0",
    facecolor="none", edgecolor=COL_SMOOTH, linewidth=3.5,
)
ax_r.add_patch(rect_9x9)
ax_r.text(4.5, 13.05, "← 9×9 moving average window（9 m × 9 m）→",
          ha="center", va="bottom", fontsize=9, color=COL_SMOOTH, fontweight="bold")

# 中心セルに「z̄₅」
draw_cell(ax_r, 4.5, 8.5, CS, fc=COL_CENTER, ec="white", lw=2.5,
          text="z̄₅\n（平均値）", txt_color="white", fontsize=8)

# 矢印と STEP 1 テキスト
ax_r.text(4.5, 4.0, "↑ STEP 1：9×9 窓内（81セル）の移動平均 → 平滑化された標高 z̄",
          ha="center", va="top", fontsize=9, color=COL_SMOOTH, fontweight="bold",
          wrap=True)

# ↓
ax_r.annotate("", xy=(4.5, 3.6), xytext=(4.5, 4.0),
              arrowprops=dict(arrowstyle="-|>", color=COL_ARROW, lw=2.5))

# STEP 2 テキスト
ax_r.text(4.5, 3.45, "STEP 2：平滑化DEMに Horn 3×3 オペレータを適用",
          ha="center", va="top", fontsize=10, fontweight="bold", color=COL_TEXT)

# 平滑化後 DEM の3×3（概念）
smooth_labels = [["z̄₁", "z̄₂", "z̄₃"],
                 ["z̄₄", "z̄₅", "z̄₆"],
                 ["z̄₇", "z̄₈", "z̄₉"]]
smooth_colors = [
    [COL_CORNER, COL_NORTH, COL_CORNER],
    [COL_EAST,   COL_CENTER, COL_EAST  ],
    [COL_CORNER, COL_NORTH, COL_CORNER],
]
for ri in range(3):
    for ci in range(3):
        cx = (ci + 3) * 0.65 + 1.5
        cy = 2.5 - ri * 0.65
        draw_cell(ax_r, cx, cy, 0.60, fc=smooth_colors[ri][ci], ec="white", lw=1.5,
                  text=smooth_labels[ri][ci], txt_color="white", fontsize=8)

ax_r.text(5.1, 1.75, "（各セルは 9×9 移動平均済みの標高値）",
          ha="left", va="center", fontsize=8, color="#666666", style="italic")

# ↓
ax_r.annotate("", xy=(4.5, 0.5), xytext=(4.5, 1.0),
              arrowprops=dict(arrowstyle="-|>", color=COL_ARROW, lw=2.5))

# 結果ボックス
result_box_r = mpatches.FancyBboxPatch(
    (1.5, -1.1), 6.0, 1.4,
    boxstyle="round,pad=0.1",
    facecolor=COL_SMOOTH, edgecolor="white", linewidth=2, alpha=0.9,
)
ax_r.add_patch(result_box_r)
ax_r.text(4.5, -0.35, "Slope（傾斜角）= arctan(√(gx²＋gy²))",
          ha="center", va="center", fontsize=10, color="white", fontweight="bold")

ax_r.text(4.5, -1.3, "有効スケール：約 9 m × 9 m（= 9×9 平滑化の効果）",
          ha="center", va="top", fontsize=9, color=COL_SMOOTH,
          style="italic")


# ======================================================
# 中央境界線 ＋ vs テキスト
# ======================================================
# （gridspec の間なので fig 座標で描く）
fig.text(0.5, 0.5, "vs", ha="center", va="center",
         fontsize=18, fontweight="bold", color="#888888",
         bbox=dict(facecolor=COL_BG, edgecolor="#CCCCCC", boxstyle="round,pad=0.3"))

# 下部共通注記
fig.text(
    0.5, 0.01,
    "※ Horn オペレータ（加重微分）自体は A・B どちらも 3×3 固定。\n"
    "   B は「DEMを事前に平滑化することで、より広域（9 m スケール）の傾斜傾向を抽出」する手法。",
    ha="center", va="bottom", fontsize=10, color="#444455", style="italic",
)

# ======================================================
# 出力
# ======================================================
out_path = "/Users/ngok/dev/aigeology-kit/scripts/horn_slope_diagram_9x9.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COL_BG)
print(f"[OK] 保存しました: {out_path}")
plt.show()
