#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
feature_distribution_inspector.py

ランダムフォレスト等に与える「特徴量（説明変数）」の分布傾向を確認するためのツール。

機能概要
--------
- gpkg / parquet / csv のいずれか 1 ファイルを読み込み、
- 対話的に列を選択し（番号 + 範囲指定 2-6, 1-3,7 などに対応）、
- 列ごとに「数値 / 数値変換 / カテゴリ / IDライク」を自動判定し、
  - 数値/数値変換列: ヒストグラム + 基本統計量 CSV
  - カテゴリ列: 棒グラフ + value_counts CSV
  - IDライク列: デフォルトではスキップ
- 閾値（数値変換割合・カテゴリ判定のユニーク数）も対話的に設定可能。
- 日本語フォントが見つからない場合は、ユーザーに対応方法を確認。
- 出力フォルダは、デフォルト（入力ファイルと同じ場所にタイムスタンプ付きサブフォルダ）を提示しつつ、
  ユーザーが任意のパスを指定可能。

想定環境
--------
- OS: Windows 11 / macOS
- Python: 3.11
- 仮想環境: terrain-env / aigeology-env など（mamba 管理を想定）
- 主なライブラリ: pandas, pyarrow, geopandas（任意）, matplotlib
"""

import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype

# GUI なし環境でも動作するよう、非インタラクティブ backend を指定
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager, rcParams

try:
    import geopandas as gpd  # GPKG 読み込み用（インストールされていれば使用）
except ImportError:
    gpd = None

# 日本語フォントが無い場合に英語ラベルに切り替えるかどうかのフラグ
USE_ENGLISH_LABELS: bool = False


# ============================================================
# 日本語フォント設定まわり
# ============================================================

def setup_font_and_language() -> None:
    """
    日本語フォントの有無を確認し、フォント設定およびラベル言語モードを決める。

    - 候補フォントが見つかればそれを使用（日本語ラベルのまま）。
    - 見つからない場合は、ユーザーに
        1) そのまま続行（日本語文字化けの可能性あり）
        2) 英語ラベルに切り替える
        3) 処理を中断
      を選んでもらう。
    """
    global USE_ENGLISH_LABELS

    candidates = [
        "IPAexGothic",
        "IPAGothic",
        "Yu Gothic",
        "YuGothic",
        "Meiryo",
        "MS Gothic",
        "Hiragino Sans",
        "Noto Sans CJK JP",
    ]

    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available_fonts:
            rcParams["font.family"] = name
            print(f"[INFO] 日本語フォントを使用します: {name}")
            USE_ENGLISH_LABELS = False
            return

    # ここまで来たら日本語フォント候補が見つからない
    print("\n[WARN] 日本語フォント候補が見つかりません。")
    print("      グラフ内の日本語（タイトル・軸ラベルなど）が文字化けする可能性があります。")
    while True:
        choice = input(
            "対応を選択してください:\n"
            "  1) そのまま続行（日本語ラベルのまま。文字化けの可能性あり）\n"
            "  2) 英語ラベルに切り替える（title/count など）\n"
            "  3) 処理を中断してフォントをインストールする\n"
            "選択 [1/2/3]（Enter=1）: "
        ).strip()
        if choice == "" or choice == "1":
            USE_ENGLISH_LABELS = False
            print("[INFO] 既定フォントのまま続行します（日本語が文字化けする可能性があります）。\n")
            return
        elif choice == "2":
            USE_ENGLISH_LABELS = True
            print("[INFO] タイトル・ラベルを英語表記に切り替えます。\n")
            return
        elif choice == "3":
            print("[INFO] ユーザー指定により処理を中断します。フォントインストール後に再実行してください。")
            sys.exit(0)
        else:
            print("[WARN] 1/2/3 のいずれかを入力してください。")


# ============================================================
# ユーティリティ
# ============================================================

def sanitize_filename(name: str) -> str:
    """
    列名などをファイル名に安全に使えるように変換する。
    - 英数字・日本語・アンダースコア・ハイフン以外はアンダースコアに置換。
    """
    return re.sub(r'[^0-9A-Za-z\u3040-\u30ff\u4e00-\u9faf\-_]+', "_", str(name))


def ask_path() -> Path:
    """入力ファイルパスを対話的に取得する。"""
    print("=== 特徴量分布チェックツール ===")
    print("gpkg / parquet / csv のいずれか 1 ファイルを指定してください。")
    while True:
        path_str = input("入力ファイルのパス: ").strip().strip('"').strip("'")
        if not path_str:
            print("[WARN] 空です。パスを入力してください。")
            continue
        p = Path(path_str)
        if not p.exists():
            print(f"[ERROR] ファイルが見つかりません: {p}")
            continue
        if not p.is_file():
            print(f"[ERROR] ファイルではありません: {p}")
            continue
        return p


def load_table(path: Path) -> pd.DataFrame:
    """拡張子に応じてテーブルを読み込む。"""
    ext = path.suffix.lower()
    print(f"[INFO] 読み込み中: {path}")
    try:
        if ext in (".parquet", ".pq"):
            print("[INFO] parquet として読み込みます。")
            df = pd.read_parquet(path)
        elif ext in (".csv", ".txt"):
            print("[INFO] CSV として読み込みます。")
            df = pd.read_csv(path)
        elif ext == ".gpkg":
            if gpd is None:
                print("[ERROR] geopandas がインポートできません。gpkg を扱うには geopandas が必要です。")
                sys.exit(1)
            print("[INFO] GPKG (ベクタ) として読み込みます。geometry 列は削除します。")
            gdf = gpd.read_file(path)
            if "geometry" in gdf.columns:
                df = gdf.drop(columns="geometry")
            else:
                df = pd.DataFrame(gdf)
        else:
            print(f"[ERROR] 未対応の拡張子です: {ext}")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 読み込みに失敗しました: {e}")
        sys.exit(1)

    print(f"[INFO] 読み込み完了: {df.shape[0]} 行 × {df.shape[1]} 列")
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"[INFO] 推定メモリ使用量: 約 {mem_mb:.1f} MB")
    return df


def show_columns(df: pd.DataFrame) -> None:
    """列一覧を番号付きで表示する。"""
    print("\n=== 列一覧 ===")
    for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes)):
        print(f"[{i:3d}] {col}  (dtype: {dtype})")
    print("====================")


def ask_columns(df: pd.DataFrame) -> list[str]:
    """解析対象とする列を対話的に選択する（番号指定 + 範囲指定対応）。

    対応フォーマット例:
      - 0,2,5
      - 2-6          → 2,3,4,5,6
      - 1-3,7,10-12  → 1,2,3,7,10,11,12
      - all / 空Enter → 全列
    """
    show_columns(df)
    n_cols = len(df.columns)

    while True:
        s = input(
            "解析対象とする列番号をカンマ区切りで入力してください\n"
            "  例: 0,2,5 / 2-6 / 1-3,7,10-12\n"
            "  all または空 Enter: 全列を対象にする\n"
            "入力: "
        ).strip()

        if s == "" or s.lower() == "all":
            selected_cols = list(df.columns)
            print(f"[INFO] 全 {len(selected_cols)} 列を暫定選択しました。")
            return selected_cols

        parts = [x.strip() for x in s.split(",") if x.strip()]
        indices: list[int] = []
        ok = True

        for p in parts:
            # 範囲指定 (例: 2-6, 1:4)
            if "-" in p or ":" in p:
                delim = "-" if "-" in p else ":"
                start_str, end_str = [q.strip() for q in p.split(delim, 1)]
                if not (start_str.isdigit() and end_str.isdigit()):
                    print(f"[WARN] 範囲指定が正しくありません: {p}")
                    ok = False
                    break
                start = int(start_str)
                end = int(end_str)
                if start > end:
                    print(f"[WARN] 範囲指定の開始 > 終了 です: {p}")
                    ok = False
                    break
                if start < 0 or end >= n_cols:
                    print(f"[WARN] 範囲外の番号を含んでいます: {p}  (有効範囲: 0 ～ {n_cols-1})")
                    ok = False
                    break
                indices.extend(range(start, end + 1))
            else:
                # 単一番号
                if not p.isdigit():
                    print(f"[WARN] 数字または範囲ではない入力を検出しました: {p}")
                    ok = False
                    break
                idx = int(p)
                if idx < 0 or idx >= n_cols:
                    print(f"[WARN] 範囲外の番号です: {idx}  (有効範囲: 0 ～ {n_cols-1})")
                    ok = False
                    break
                indices.append(idx)

        if not ok:
            print("[INFO] もう一度入力してください。")
            continue

        if not indices:
            print("[WARN] 列が一つも選択されていません。再入力してください。")
            continue

        # 重複を削除しつつ順序を維持
        unique_indices: list[int] = []
        seen = set()
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)

        selected_cols = [df.columns[i] for i in unique_indices]
        print("[INFO] 選択された列:")
        for c in selected_cols:
            print(f"  - {c}")
        return selected_cols


def ask_float_with_default(prompt: str, default: float, min_value: float, max_value: float) -> float:
    """下限・上限付き float 入力（空なら default）。"""
    while True:
        s = input(f"{prompt} [default={default}]: ").strip()
        if s == "":
            return default
        try:
            v = float(s)
            if v < min_value or v > max_value:
                print(f"[WARN] {min_value} ～ {max_value} の範囲で入力してください。")
                continue
            return v
        except ValueError:
            print("[WARN] 数値を入力してください。")


def ask_int_with_default(prompt: str, default: int, min_value: int, allow_zero: bool = False) -> int:
    """下限付き int 入力（空なら default）。"""
    while True:
        s = input(f"{prompt} [default={default}]: ").strip()
        if s == "":
            return default
        try:
            v = int(s)
            if allow_zero and v == 0:
                return v
            if v < min_value:
                print(f"[WARN] {min_value} 以上の整数を入力してください。")
                continue
            return v
        except ValueError:
            print("[WARN] 整数を入力してください。")


def ask_thresholds() -> Tuple[float, int]:
    """
    列の自動分類に使う閾値を対話的に取得する。

    戻り値:
        numeric_ratio_threshold: 文字列列を数値キャストしてよいとみなす比率（0～1）
        max_categories: カテゴリ列とみなす最大ユニーク数
    """
    print("\n=== 列の自動分類に使う閾値を設定します ===")
    print(
        "- 文字列列を数値に変換したとき、何 % 以上が変換成功なら「数値扱い」にするか\n"
        "  （例: 0.95 → 全体の 95% 以上が数値に変換できれば numeric_cast）"
    )
    numeric_ratio_threshold = ask_float_with_default(
        "数値変換成功率の閾値 (0.0～1.0)", default=0.95, min_value=0.0, max_value=1.0
    )

    print(
        "\n- 文字列列のユニーク値が何種類以下なら「カテゴリ列」とみなすか\n"
        "  （例: 30 以下ならカテゴリ列、それ以上なら ID ライクとしてスキップ）"
    )
    max_categories = ask_int_with_default(
        "カテゴリ列とみなす最大ユニーク数", default=30, min_value=1, allow_zero=False
    )

    print(f"\n[INFO] 閾値設定: numeric_ratio_threshold={numeric_ratio_threshold}, max_categories={max_categories}")
    return numeric_ratio_threshold, max_categories


def ask_bins() -> int:
    """ヒストグラムの階級数を尋ねる。"""
    print("\n=== ヒストグラムの設定 ===")
    return ask_int_with_default("ヒストグラムの階級数（bins）", default=30, min_value=1, allow_zero=False)


def ask_sample_size() -> int:
    """ヒストグラム計算用のサンプリング行数を尋ねる。0 で全件。"""
    print(
        "\n=== サンプリング設定（ヒストグラム用） ===\n"
        "- 0: サンプリングせず、全行を使用（大きなファイルでは時間がかかることがあります）。\n"
        "- 正の整数: 指定行数までランダムサンプリングしてヒストグラムを作成します。\n"
        "  ※ 統計量（平均・分位点など）は常に全行を用いて計算します。\n"
    )
    return ask_int_with_default("ヒストグラム用サンプル行数 (0 = 全件)", default=0, min_value=0, allow_zero=True)


def prepare_output_dir(input_path: Path) -> Path:
    """
    デフォルトの出力フォルダ（入力ファイルと同じフォルダにタイムスタンプ付き）を作成しつつ、
    ユーザーに任意の保存先を指定する機会を与える。
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_dir = input_path.parent / f"{input_path.stem}_hist_{ts}"

    print("\n=== 出力フォルダの設定 ===")
    print(f"[INFO] デフォルトの出力フォルダ候補: {default_dir}")
    while True:
        s = input(
            "出力フォルダを変更する場合はパスを入力してください。\n"
            "  - 空 Enter: 上記のデフォルトフォルダを使用\n"
            "  - 任意のパス: そのフォルダを作成して使用\n"
            "入力: "
        ).strip().strip('"').strip("'")

        if s == "":
            out_dir = default_dir
        else:
            out_dir = Path(s)

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] 出力フォルダ: {out_dir}")
            return out_dir
        except Exception as e:
            print(f"[ERROR] 出力フォルダを作成できません: {e}")
            print("別のパスを指定してください。")


# ============================================================
# 列タイプ判定 & 統計・プロット
# ============================================================

def classify_column_for_plot(
    s: pd.Series,
    numeric_ratio_threshold: float,
    max_categories: int,
) -> Tuple[str, Optional[float], Optional[int]]:
    """
    列をヒストグラム/棒グラフの観点から分類する。

    戻り値:
        col_type: "numeric" / "numeric_cast" / "category" / "id_like"
        numeric_ratio: 文字列→数値変換成功率（numeric_cast 判定に利用した値。数値列では None）
        nunique: ユニーク数（category / id_like 判定に利用した値。数値列では None）
    """
    # もともと数値型なら即 numeric
    if is_numeric_dtype(s):
        return "numeric", None, None

    # 文字列として扱う
    s_str = s.astype(str).str.strip()
    # 数値化トライ
    s_num = pd.to_numeric(s_str, errors="coerce")
    numeric_ratio = float(s_num.notna().mean())

    if numeric_ratio >= numeric_ratio_threshold:
        # ほぼ数値とみなす
        return "numeric_cast", numeric_ratio, None

    # 数値化が難しい → カテゴリ or ID とみなす
    nunique = int(s_str.nunique(dropna=True))
    if nunique <= max_categories:
        return "category", numeric_ratio, nunique
    else:
        return "id_like", numeric_ratio, nunique


def compute_stats(series: pd.Series, col_name: str) -> dict:
    """1 列分の基本統計量を計算して dict で返す。（数値列用）"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {
            "column": col_name,
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "q05": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
            "q95": float("nan"),
            "max": float("nan"),
        }

    return {
        "column": col_name,
        "count": int(s.count()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "q05": float(s.quantile(0.05)),
        "q25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "q75": float(s.quantile(0.75)),
        "q95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


def save_histogram(
    series: pd.Series,
    col_name: str,
    out_dir: Path,
    bins: int,
    sample_n: Optional[int],
) -> None:
    """1 列分のヒストグラムを PNG として保存する。（数値列用）"""
    global USE_ENGLISH_LABELS

    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        print(f"[WARN] 列 '{col_name}' は有効な数値データがありません。ヒストグラムをスキップします。")
        return

    if sample_n is not None and sample_n > 0 and len(s) > sample_n:
        print(f"[INFO] 列 '{col_name}' は {len(s)} 件中 {sample_n} 件をサンプリングしてヒストグラムを作成します。")
        s_plot = s.sample(sample_n, random_state=0)
    else:
        s_plot = s

    safe_name = sanitize_filename(col_name)
    png_path = out_dir / f"hist_{safe_name}.png"

    plt.figure(figsize=(8, 6))
    plt.hist(s_plot, bins=bins)

    if USE_ENGLISH_LABELS:
        title = f"{col_name} histogram"
        xlabel = col_name
        ylabel = "count"
    else:
        title = f"{col_name} のヒストグラム"
        xlabel = col_name
        ylabel = "頻度"

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    print(f"[OK] ヒストグラム保存: {png_path}")


def save_category_bar_and_counts(
    series: pd.Series,
    col_name: str,
    out_dir: Path,
) -> None:
    """
    カテゴリ列について、棒グラフ PNG と value_counts CSV を保存する。
    （既に max_categories 以下である前提）
    """
    global USE_ENGLISH_LABELS

    s_str = series.astype(str)
    vc = s_str.value_counts().sort_values(ascending=False)

    safe_name = sanitize_filename(col_name)

    # 棒グラフ
    png_path = out_dir / f"bar_{safe_name}.png"
    plt.figure(figsize=(8, 6))
    vc.plot(kind="bar")

    if USE_ENGLISH_LABELS:
        title = f"{col_name} category counts"
        xlabel = col_name
        ylabel = "count"
    else:
        title = f"{col_name} のカテゴリ頻度"
        xlabel = col_name
        ylabel = "件数"

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    print(f"[OK] 棒グラフ保存: {png_path}")

    # value_counts CSV
    csv_path = out_dir / f"value_counts_{safe_name}.csv"
    vc.to_csv(csv_path, header=["count"], encoding="utf-8-sig")
    print(f"[OK] value_counts CSV 保存: {csv_path}")


# ============================================================
# メイン処理
# ============================================================

def main() -> None:
    # 0) フォント・ラベル言語設定
    setup_font_and_language()

    # 1) 入力ファイル
    input_path = ask_path()

    # 2) テーブル読み込み
    df = load_table(input_path)

    # 3) 列選択（暫定）
    selected_cols = ask_columns(df)

    # 4) 閾値設定（数値変換成功率・カテゴリ最大ユニーク数）
    numeric_ratio_threshold, max_categories = ask_thresholds()

    # 5) ヒストグラム設定
    bins = ask_bins()
    sample_n = ask_sample_size()

    # 6) 出力フォルダ準備（ユーザーが保存先を選択可能）
    out_dir = prepare_output_dir(input_path)

    # 7) 列ごとの分類結果を保存するためのリスト
    stats_list: list[dict] = []
    col_info_list: list[dict] = []

    print("\n[INFO] 列ごとの判定と処理を開始します。\n")

    for col in selected_cols:
        print(f"=== 列 '{col}' の判定 ===")
        s = df[col]

        col_type, numeric_ratio, nunique = classify_column_for_plot(
            s, numeric_ratio_threshold=numeric_ratio_threshold, max_categories=max_categories
        )

        # 判定結果をログ用リストに追加
        col_info = {
            "column": col,
            "dtype": str(s.dtype),
            "col_type": col_type,
            "numeric_ratio": numeric_ratio,
            "nunique": nunique,
        }
        col_info_list.append(col_info)

        if col_type == "numeric":
            print(f"[INFO] 列 '{col}' は数値列 (dtype={s.dtype}) と判定されました。")
            stats = compute_stats(s, col)
            stats_list.append(stats)
            save_histogram(s, col, out_dir, bins=bins, sample_n=sample_n)

        elif col_type == "numeric_cast":
            print(
                f"[INFO] 列 '{col}' は文字列ですが、数値変換成功率 {numeric_ratio:.3f} により "
                f"'numeric_cast' と判定されました。数値に変換して処理します。"
            )
            s_num = pd.to_numeric(s.astype(str).str.strip(), errors="coerce")
            stats = compute_stats(s_num, col)
            stats_list.append(stats)
            save_histogram(s_num, col, out_dir, bins=bins, sample_n=sample_n)

        elif col_type == "category":
            print(
                f"[INFO] 列 '{col}' はカテゴリ列と判定されました。"
                f"(numeric_ratio={numeric_ratio:.3f}, nunique={nunique})"
            )
            save_category_bar_and_counts(s, col, out_dir)

        elif col_type == "id_like":
            print(
                f"[INFO] 列 '{col}' は ID / 自由記述などの文字列列と推定されました。"
                f"(numeric_ratio={numeric_ratio:.3f}, nunique={nunique})"
            )
            print("[INFO] デフォルト設定ではヒストグラム・棒グラフの作成をスキップします。")

        else:
            print(f"[WARN] 列 '{col}' の判定で想定外のタイプが返されました: {col_type}")
            print("[INFO] この列の処理はスキップします。")

        print("")

    # 8) 統計量 CSV 出力（数値/数値キャスト列のみ）
    if stats_list:
        stats_df = pd.DataFrame(stats_list)
        stats_csv_path = out_dir / "summary_stats.csv"
        stats_df.to_csv(stats_csv_path, index=False, encoding="utf-8-sig")
        print(f"[OK] 数値列の統計量 CSV を保存しました: {stats_csv_path}")
    else:
        print("[INFO] 数値列（numeric / numeric_cast）がなかったため、summary_stats.csv は出力しません。")

    # 9) 列判定結果ログ CSV
    col_info_df = pd.DataFrame(col_info_list)
    col_info_csv_path = out_dir / "column_type_summary.csv"
    col_info_df.to_csv(col_info_csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 列の判定結果ログ CSV を保存しました: {col_info_csv_path}")

    print("\n=== 処理完了 ===")
    print("・出力フォルダ内の PNG / CSV を確認してください。")
    print("  - summary_stats.csv : 数値列の基本統計量")
    print("  - hist_*.png        : 数値列のヒストグラム")
    print("  - bar_*.png         : カテゴリ列の棒グラフ")
    print("  - value_counts_*.csv: カテゴリ列の出現頻度")
    print("  - column_type_summary.csv: 列ごとの判定結果の一覧")


if __name__ == "__main__":
    main()
