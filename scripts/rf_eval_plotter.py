#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
rf_eval_plotter.py

rf_geomorph_toolkit.py の予測モードで出力された

  - *_eval_confusion_matrix.csv
  - *_eval_classification_report.csv
  - *_eval_pred_summary.csv

をまとめて図化するスクリプト。

入出力パスはすべてインタラクティブに指定する：
  1) 評価CSVが入っているフォルダ
  2) 図を保存する出力フォルダ（デフォルト: <入力フォルダ>/eval_plots）
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def strip_quotes(s: str) -> str:
    """文字列の前後に付いた " または ' を除去する"""
    if len(s) >= 2 and ((s[0] == s[-1]) and s[0] in ("'", '"')):
        return s[1:-1]
    return s


def choose_cm_file(base_dir: Path) -> Path:
    """フォルダ内の *_eval_confusion_matrix.csv から対象を1つ選ぶ"""
    cm_files = sorted(base_dir.glob("*_eval_confusion_matrix.csv"))
    if not cm_files:
        raise FileNotFoundError("指定フォルダに *_eval_confusion_matrix.csv が見つかりません。")
    if len(cm_files) == 1:
        return cm_files[0]

    print("\n[評価セット候補（混同行列CSV）]")
    for i, f in enumerate(cm_files):
        print(f"  [{i}] {f.name}")
    idx = input(f"使う番号を選んでください [0-{len(cm_files)-1}]（空=0）: ").strip()
    if not idx:
        idx = "0"
    i = int(idx)
    return cm_files[i]


def plot_confusion_matrix(cm_df: pd.DataFrame, out_png: Path):
    """混同行列のヒートマップ"""
    labels = cm_df.index.astype(str).tolist()
    cm = cm_df.values.astype(float)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("予測クラス")
    ax.set_ylabel("真のクラス")
    ax.set_title("混同行列（行: 真, 列: 予測）")

    # クラス数が少ないときだけ値を描画
    if cm.shape[0] <= 10:
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                if val == 0:
                    continue
                color = "white" if val > thresh else "black"
                ax.text(j, i, int(val), ha="center", va="center", color=color, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_class_metrics(rep_df: pd.DataFrame, out_png: Path):
    """
    classification_report(output_dict=True) 形式の CSV から
    クラス別 precision / recall / f1-score を棒グラフで描画
    """
    rep_df = rep_df.copy()
    rep_df.index = rep_df.index.astype(str)

    # 行: クラス / accuracy / macro avg / weighted avg
    mask_class = ~rep_df.index.isin(["accuracy", "macro avg", "weighted avg"])
    cls_df = rep_df[mask_class]

    # support=0 のクラスなどは除外
    if "support" in cls_df.columns:
        cls_df = cls_df[cls_df["support"] > 0]

    labels = cls_df.index.tolist()
    x = np.arange(len(labels))

    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    p = cls_df.get("precision", pd.Series([np.nan] * len(labels), index=labels))
    r = cls_df.get("recall", pd.Series([np.nan] * len(labels), index=labels))
    f1 = cls_df.get("f1-score", pd.Series([np.nan] * len(labels), index=labels))

    ax.bar(x - width, p.values, width, label="precision")
    ax.bar(x,         r.values, width, label="recall")
    ax.bar(x + width, f1.values, width, label="f1")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("スコア")
    ax.set_title("クラス別 precision / recall / f1-score")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_pred_summary(summary_df: pd.DataFrame, out_png: Path):
    """
    pred_summary CSV（class, count, proba_max_mean）から
    クラス別件数＋平均最大確率を図化
    """
    summary_df = summary_df.copy()
    labels = summary_df["class"].astype(str).tolist()
    x = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)

    counts = summary_df["count"].values
    ax1.bar(x, counts)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_ylabel("予測件数")
    ax1.set_title("クラス別 予測件数 ＋ 平均最大確率")

    if "proba_max_mean" in summary_df.columns:
        ax2 = ax1.twinx()
        ax2.plot(x, summary_df["proba_max_mean"].values, marker="o")
        ax2.set_ylabel("平均 最大確率")

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    # 1) 入力フォルダをインタラクティブ指定
    print("=== ランダムフォレスト評価結果の図化ツール ===\n")
    base_dir_str = input(
        "評価CSVがあるフォルダのパスを入力してください\n"
        "（例: D:\\AiGeology\\1115\\predict\\A\\1119, 空Enter=カレントディレクトリ）: "
    ).strip()
    base_dir_str = strip_quotes(base_dir_str)

    if not base_dir_str:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir_str)

    if not base_dir.is_dir():
        raise SystemExit(f"フォルダが見つかりません: {base_dir}")

    print(f"\n[INFO] 評価CSVフォルダ: {base_dir}")

    # 2) 混同行列 CSV を選ぶ → プレフィックスを決定
    cm_file = choose_cm_file(base_dir)
    prefix = cm_file.name.replace("_eval_confusion_matrix.csv", "")
    rep_file = base_dir / f"{prefix}_eval_classification_report.csv"
    summary_file = base_dir / f"{prefix}_eval_pred_summary.csv"

    print(f"\n[INFO] 対象プレフィックス: {prefix}")
    print(f"[INFO] confusion_matrix:         {cm_file.name}")
    print(f"[INFO] classification_report:    {rep_file.name} "
          f"({'exists' if rep_file.exists() else 'MISSING'})")
    print(f"[INFO] pred_summary:             {summary_file.name} "
          f"({'exists' if summary_file.exists() else 'MISSING'})")

    # 3) 出力フォルダをインタラクティブ指定
    default_out_dir = base_dir / "eval_plots"
    out_dir_input = input(
        "\n図の出力先フォルダを指定してください。\n"
        f"  デフォルト: {default_out_dir}\n"
        "  空Enter=デフォルトを使用。パスを入力するとそのフォルダに保存します。\n"
        "出力フォルダ: "
    ).strip()
    out_dir_input = strip_quotes(out_dir_input)

    if not out_dir_input:
        out_dir = default_out_dir
    else:
        out_dir = Path(out_dir_input)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 図の出力先: {out_dir}\n")

    # 4) 混同行列を図化
    cm_df = pd.read_csv(cm_file, index_col=0)
    cm_png = out_dir / f"{prefix}_cm.png"
    plot_confusion_matrix(cm_df, cm_png)
    print(f"[OK] 混同行列を図化: {cm_png}")

    # 5) classification_report を図化
    if rep_file.exists():
        rep_df = pd.read_csv(rep_file, index_col=0)
        rep_png = out_dir / f"{prefix}_class_metrics.png"
        plot_class_metrics(rep_df, rep_png)
        print(f"[OK] クラス別指標を図化: {rep_png}")
    else:
        print("[WARN] classification_report CSV が見つからないため、クラス別指標図はスキップします。")

    # 6) pred_summary を図化
    if summary_file.exists():
        summary_df = pd.read_csv(summary_file)
        ps_png = out_dir / f"{prefix}_pred_summary.png"
        plot_pred_summary(summary_df, ps_png)
        print(f"[OK] 予測サマリを図化: {ps_png}")
    else:
        print("[WARN] pred_summary CSV が見つからないため、予測サマリ図はスキップします。")

    print("\n=== 完了しました ===")


if __name__ == "__main__":
    main()
