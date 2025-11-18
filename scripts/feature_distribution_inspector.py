#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特徴量分布の事前チェック用ツール。

モード:
  1) 単変量分布の確認
      - 数値列: ヒストグラム PNG + 基本統計量 CSV
      - 文字列列: value_counts CSV + 棒グラフ PNG
  2) ペアワイズ WMW + AUC 解析
      - ラベル列を 1 つ選び、クラスペアごと × 特徴量ごとに
        Mann-Whitney U 検定 + AUC（+オプションで Cliff's δ）を計算
      - 結果をロング形式 CSV + AUC 行列 CSV として出力

【モード1（単変量分布）の出力ファイル仕様】

  入力ファイルごとに 1 つの出力フォルダが作成され、その中に
  各列ごとの画像 / CSV が保存される。

  1. 数値列のヒストグラム PNG

     - ファイル名:
         <列名>_hist.png
       例: TEST1mDEM1k_epsg6673_slope_deg_r25m_gauss_b1_hist.png

     - 中身:
         x軸 : 対象列の値
         y軸 : 度数（件数）
         タイトル : 「ヒストグラム: 列名」

     - データ:
         NaN を除外した数値のみを対象とし、
         指定したビン数（デフォルト 50）でヒストグラム化している。

  2. 数値列の基本統計量 CSV

     - ファイル名:
         numeric_summary_stats.csv

     - 1 行 = 1 数値列。主な列の意味:

         - column : 列名
         - count  : 有効な数値の件数（NaN 除外後）
         - mean   : 平均値
         - std    : 標準偏差
         - min    : 最小値
         - q05    : 5 パーセンタイル（下位 5%）
         - q25    : 25 パーセンタイル（第1四分位）
         - q50    : 50 パーセンタイル（中央値）
         - q75    : 75 パーセンタイル（第3四分位）
         - q95    : 95 パーセンタイル（上位 5%）
         - max    : 最大値

       → 各特徴量のレンジ・代表値・分布の偏りをざっくり把握するための表。

  3. 文字列 / カテゴリ列の頻度表 CSV

     - ファイル名:
         <列名>_value_counts.csv
       例: 地形分類_value_counts.csv

     - 中身:
         列名: 元の列名（カテゴリ値）
         count: そのカテゴリ値の出現回数

       ※ NaN（欠損）の頻度も含めて集計している。

  4. 文字列 / カテゴリ列の棒グラフ PNG

     - ファイル名:
         <列名>_bar.png
       例: 地形分類_bar.png

     - 中身:
         x軸 : 上位 N カテゴリ（デフォルト 20 件）をラベル表示
         y軸 : 件数
         タイトル : 「カテゴリ頻度（上位 N）: 列名」

       → カテゴリ分布の偏り（特定クラスに極端に集中していないか）を
         見るための概観用グラフ。


【モード2（WMW + AUC）の出力ファイル仕様】

  1. wmw_results_long.csv  （「縦長」形式：1行 = 特徴量 × クラスペア）

     各行は「ある特徴量が、あるクラスペア（group1 vs group2）に対して
     どれくらいクラス分離に効いているか」を表す。

     主な列の意味:

       - label_col
           WMW 解析に用いたラベル列名。
           例: "地形分類コード"

       - feature
           対象とした数値特徴量の列名。
           例: "TEST1mDEM1k_epsg6673_slope_deg_r25m_gauss_b1"

       - group1, group2
           比較対象となる 2 クラスのラベル値。
           例: group1=1010101, group2=3010101
           （内部的には、同じペアが重複しないように小さい方を group1 として並べる）

       - n_group1, n_group2
           各グループで、有効な数値（NaN 除外）として WMW に使われたサンプル数。

       - mean_group1, mean_group2
           各グループの平均値。

       - median_group1, median_group2
           各グループの中央値。

       - u_stat
           Mann-Whitney U 検定の U 統計量。

       - p_value
           U 統計量に対応する p 値。
           alternative（two-sided / greater / less）は対話的に選択する。

       - auc
           効果量としての AUC（0.5 = ランダム, 1.0 = 完全分離）。
           ここでは「group1 の値が group2 より大きい確率 + 0.5 * 同値の確率」に対応。

       - cliffs_delta
           Cliff's delta（オプション）。2 * AUC - 1 に相当する値。
           （calc_cliffs_delta=OFF の場合は NaN）

       - note
           サンプル数不足などでスキップされた場合の理由。

  2. auc_matrix_by_feature.csv  （「行: 特徴量 × 列: クラスペア」の AUC 行列）

     このファイルは AUC だけを抜き出して「マトリクス状」に並べたもの。

       - 行 index: feature
           特徴量名。

       - 列: class_pair
           "group1_vs_group2" という文字列で表現されたクラスペア。
           例: "1010101_vs_3010101"

       - セルの値: AUC
           wmw_results_long.csv の auc を、feature × class_pair で pivot したもの。
           Excel 等で開き、条件付き書式（例: AUC>=0.7 を色付け）をかけることで、
           どの特徴量がどのクラス境界に対して効いているかを一覧できる。

  3. wmw_result_auc_ge.<thr>.csv / wmw_result_auc_lt_<thr>.csv

     AUC の閾値 thr （デフォルト 0.70）を対話的に指定し、閾値以上の行と閾値未満の行を別々の csv に書き出す。

       - wmw_results_auc_ge_<thr>.csv
           → AUC >= thr の行だけを含む。

       - wmw_results_auc_lt_<thr>.csv
           → AUC < thr の行だけを含む。

     いずれも wmw_results_long.csv のサブセットであり、ログに表示される SUMMARY は「それぞれ20行のダイジェスト」のみ。

       - 列: class_pair
           "group1_vs_group2" という文字列で表現されたクラスペア。
           例: "1010101_vs_3010101"


【結果の読み方の目安（経験則）】

  ● モード1（単変量分布）

    - numeric_summary_stats.csv
        - q05, q95 付近が「人間が想定しているレンジ」と大きく違う場合、
          外れ値・単位ミス・スケーリングミスの可能性を疑う。
        - q50（中央値）と mean（平均）が大きくずれている場合、
          片側に長い「裾」がある（右長 / 左長）と考えられる。
          → ログ変換 / クリッピング候補。
        - std が極端に大きいのに、AUC や feature importance が低い場合は、
          「ノイズが多いだけの特徴量」の可能性がある。

    - カテゴリ列の頻度（*_value_counts.csv + *_bar.png）
        - 1〜2カテゴリにほとんどのサンプルが集中している場合、
          そのカテゴリ列はモデルに入れてもほとんど効かない（ほぼ定数）可能性がある。
        - 逆にカテゴリ種類が多すぎる（高カーディナリティ）場合、
          one-hot などにすると次元爆発するため、事前にまとめる / ビン分けを検討する。


  ● モード2（WMW + AUC）

    - AUC（wmw_results_long.csv / auc_matrix_by_feature.csv）
        - AUC ≒ 0.5
            → 「ランダムと同等」。その特徴量単独では、そのクラスペアをほとんど分けられない。
        - AUC ≳ 0.6
            → 弱い分離。単独では心許ないが、他特徴量と組み合わせると効く可能性。
        - AUC ≳ 0.7
            → 中程度〜強い分離。該当クラスペアに対して「それなりに効く」特徴量。
        - AUC ≳ 0.8
            → かなり強い分離。ラベル付けが信頼できるなら、その境界に対して核となる特徴量候補。
        ※ あくまで経験的な目安であり、データサイズ・ラベル品質によって解釈は変えること。

    - p_value
        - p_value < 0.05 なら「統計的には有意」とされることが多いが、
          サンプル数が非常に多い場合は、わずかな差でも有意になりやすい。
        - 実務的には「p値の有意性」よりも、「AUC や Cliff's δ が十分大きいか」を重視する。

    - Cliff's delta（cliffs_delta）
        - おおよその目安（文献でよく使われる基準の一例）:
            |δ| < 0.147   → 効果ほぼなし
            0.147〜0.33   → 小さい効果
            0.33〜0.474   → 中くらいの効果
            0.474〜1.0    → 大きい効果
        - sign(δ)
            δ > 0  → group1 > group2 の方向に大きい（group1 の方が値が大きい傾向）
            δ < 0  → group1 < group2 の方向に大きい

    - 実務上の使い方のイメージ
        - まず auc_matrix_by_feature.csv を Excel で開き、
          条件付き書式（例: AUC >= 0.7 を濃い色、0.6〜0.7 を薄い色）をかける。
        - すると「どの特徴量が / どのクラスペアに対して効いているか」が一目で分かる。
        - その上で、地形学的な解釈・常識（例: 開度は尾根 vs 谷で効いていて欲しい）
          と整合しているかを確認し、RF の feature importance の結果とも照らし合わせる。


想定用途:
  - ランダムフォレストに与える前の特徴量の分布をざっくり確認
  - ラベル列と特徴量の関係（どの特徴量がどのクラス境界に対して効いていそうか）を
    WMW + AUC で定量的に把握する
"""

from __future__ import annotations

import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import geopandas as gpd  # type: ignore
    HAS_GPD = True
except ImportError:
    HAS_GPD = False

try:
    from scipy.stats import mannwhitneyu  # type: ignore
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================
# 共通ユーティリティ
# ============================================================


def human_readable_bytes(num_bytes: float) -> str:
    """バイト数を読みやすい文字列に変換する。"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:,.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def ask_file_path() -> Path:
    """入力ファイルパスを対話的に取得する。"""
    print("gpkg / parquet / csv のいずれか 1 ファイルを指定してください。")
    while True:
        s = input("入力ファイルのパス: ").strip().strip('"').strip("'")
        if not s:
            print("[WARN] 空の入力です。パスを入力してください。")
            continue
        p = Path(s)
        if not p.exists():
            print(f"[ERROR] ファイルが見つかりません: {p}")
            continue
        if not p.is_file():
            print(f"[ERROR] 通常のファイルではありません: {p}")
            continue
        return p


def prepare_output_dir(input_path: Path, suffix: str) -> Path:
    """
    デフォルトの出力フォルダを作成しつつ、
    ユーザーに任意の保存先を指定する機会を与える。
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_dir = input_path.parent / f"{input_path.stem}_{suffix}_{ts}"

    print("\n=== 出力フォルダの設定 ===")
    print(f"[INFO] デフォルトの出力フォルダ候補: {default_dir}")
    while True:
        s = input(
            "出力フォルダを変更する場合はパスを入力してください。\n"
            "  - 空 Enter: 上記のデフォルトフォルダを使用\n"
            "  - 任意のパス: そのフォルダを作成して使用\n"
            "出力パス: "
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


def detect_file_type(path: Path) -> str:
    """拡張子からファイル種別を判定するだけの簡易関数。"""
    ext = path.suffix.lower()
    if ext in [".gpkg", ".shp", ".geojson"]:
        return "vector"
    if ext in [".parquet"]:
        return "parquet"
    if ext in [".csv", ".txt"]:
        return "csv"
    return "unknown"


def load_table(path: Path) -> pd.DataFrame:
    """gpkg / parquet / csv を DataFrame として読み込む。"""
    ftype = detect_file_type(path)
    print(f"[INFO] 読み込み中: {path}")

    if ftype == "vector":
        if not HAS_GPD:
            print("[ERROR] geopandas がインポートできません。`mamba install geopandas` 等でインストールしてください。")
            sys.exit(1)
        gdf = gpd.read_file(path)
        if "geometry" in gdf.columns:
            print("[INFO] GPKG (ベクタ) として読み込みます。geometry 列は削除します。")
            df = pd.DataFrame(gdf.drop(columns=["geometry"]))
        else:
            df = pd.DataFrame(gdf)
    elif ftype == "parquet":
        df = pd.read_parquet(path)
    elif ftype == "csv":
        df = pd.read_csv(path)
    else:
        print("[ERROR] 未対応の拡張子です。gpkg / parquet / csv を指定してください。")
        sys.exit(1)

    print(f"[INFO] 読み込み完了: {len(df):,} 行 × {len(df.columns)} 列")
    try:
        mem = df.memory_usage(deep=True).sum()
        print(f"[INFO] 推定メモリ使用量: 約 {human_readable_bytes(mem)}")
    except Exception:
        pass
    return df


def list_columns(df: pd.DataFrame) -> None:
    """列一覧を表示する。"""
    print("\n=== 列一覧 ===")
    for i, c in enumerate(df.columns):
        dtype = df[c].dtype
        print(f"[{i:3d}] {c}  (dtype: {dtype})")
    print("====================")


def parse_column_selection(s: str, max_index: int) -> List[int]:
    """
    列番号入力文字列をパースする。
    例:
      "0,2,5" / "2-6" / "1-3,7"
    """
    s = s.strip()
    if not s or s.lower() == "all":
        return list(range(max_index + 1))

    indices: List[int] = []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                start = int(a)
                end = int(b)
            except ValueError:
                print(f"[WARN] 無効な範囲指定をスキップします: {part}")
                continue
            if start > end:
                start, end = end, start
            for idx in range(start, end + 1):
                if 0 <= idx <= max_index:
                    indices.append(idx)
        else:
            try:
                idx = int(part)
            except ValueError:
                print(f"[WARN] 無効な番号をスキップします: {part}")
                continue
            if 0 <= idx <= max_index:
                indices.append(idx)
    # 重複削除
    indices = sorted(set(indices))
    return indices


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s.dtype)


def is_string_series(s: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(s.dtype) or pd.api.types.is_categorical_dtype(s.dtype)


# ============================================================
# モード1: 単変量分布の確認（既存ロジック）
# ============================================================


def ask_bins() -> int:
    """数値用ヒストグラムのビン数を尋ねる。"""
    while True:
        s = input("ヒストグラムの階級数（ビン数）を指定してください [default=50]: ").strip()
        if not s:
            return 50
        try:
            v = int(s)
            if v <= 0:
                raise ValueError
            return v
        except ValueError:
            print("[WARN] 正の整数を入力してください。")


def ask_top_k() -> int:
    """カテゴリ頻度の上位何件を棒グラフにするか。"""
    while True:
        s = input("カテゴリ頻度の上位何件を棒グラフにしますか？ [default=20]: ").strip()
        if not s:
            return 20
        try:
            v = int(s)
            if v <= 0:
                raise ValueError
            return v
        except ValueError:
            print("[WARN] 正の整数を入力してください。")


def plot_numeric_hist(series: pd.Series, colname: str, out_dir: Path, bins: int) -> Dict[str, Any]:
    """数値列のヒストグラムを作成し、PNG 保存＋統計量を返す。"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    stats: Dict[str, Any] = {}
    stats["count"] = int(s.shape[0])
    if s.empty:
        print(f"[WARN] 列 '{colname}': 有効な数値データがありません。")
        return stats

    stats["mean"] = float(s.mean())
    stats["std"] = float(s.std())
    stats["min"] = float(s.min())
    stats["max"] = float(s.max())
    stats["q05"] = float(s.quantile(0.05))
    stats["q25"] = float(s.quantile(0.25))
    stats["q50"] = float(s.quantile(0.50))
    stats["q75"] = float(s.quantile(0.75))
    stats["q95"] = float(s.quantile(0.95))

    plt.figure(figsize=(8, 5))
    plt.hist(s.values, bins=bins, edgecolor="black")
    plt.title(f"ヒストグラム: {colname}")
    plt.xlabel(colname)
    plt.ylabel("頻度")
    plt.tight_layout()
    out_path = out_dir / f"{colname}_hist.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] 数値列 '{colname}' のヒストグラムを保存: {out_path}")
    return stats


def plot_categorical_bar(series: pd.Series, colname: str, out_dir: Path, top_k: int) -> pd.DataFrame:
    """文字列 / カテゴリ列の value_counts を棒グラフ＋CSV に出力する。"""
    s = series.astype("string")
    vc = s.value_counts(dropna=False)
    vc_df = vc.reset_index()
    vc_df.columns = [colname, "count"]

    # CSV 出力
    csv_path = out_dir / f"{colname}_value_counts.csv"
    vc_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 列 '{colname}' の頻度表 CSV を保存: {csv_path}")

    top = vc.head(top_k)
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(top)), top.values)
    plt.xticks(range(len(top)), [str(x) for x in top.index], rotation=45, ha="right")
    plt.title(f"カテゴリ頻度（上位 {top_k}）: {colname}")
    plt.ylabel("件数")
    plt.tight_layout()
    out_path = out_dir / f"{colname}_bar.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] 列 '{colname}' の棒グラフを保存: {out_path}")
    return vc_df


def run_hist_mode(df: pd.DataFrame, input_path: Path) -> None:
    """モード1: 単変量分布の確認。"""
    list_columns(df)
    print(
        textwrap.dedent(
            """
            解析対象とする列番号をカンマ区切りで入力してください
              例: 0,2,5 / 2-6 / 1-3,7
              all または空 Enter: 全列を対象にする
            """
        ).strip()
    )
    sel = input("入力: ")
    indices = parse_column_selection(sel, len(df.columns) - 1)

    bins = ask_bins()
    top_k = ask_top_k()

    out_dir = prepare_output_dir(input_path, suffix="hist")

    stats_rows: List[Dict[str, Any]] = []

    for idx in indices:
        col = df.columns[idx]
        s = df[col]
        if is_numeric_series(s):
            print(f"\n[NUMERIC] 列 '{col}' を解析します。")
            stats = plot_numeric_hist(s, col, out_dir, bins=bins)
            stats["column"] = col
            stats_rows.append(stats)
        elif is_string_series(s):
            print(f"\n[STRING] 列 '{col}' を解析します。")
            plot_categorical_bar(s, col, out_dir, top_k=top_k)
        else:
            print(f"\n[SKIP] 列 '{col}' (dtype: {s.dtype}) は数値でも文字列でもないためスキップします。")

    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        stats_df = stats_df[["column", "count", "mean", "std", "min", "q05", "q25", "q50", "q75", "q95", "max"]]
        csv_path = out_dir / "numeric_summary_stats.csv"
        stats_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n[OK] 数値列の基本統計量 CSV を保存しました: {csv_path}")

    print("\n=== モード1: 単変量分布の確認 完了 ===")


# ============================================================
# モード2: ペアワイズ WMW + AUC 解析
# ============================================================


def ask_label_column(df: pd.DataFrame) -> str:
    """ラベル列（クラス列）を 1 つ選択させる。"""
    list_columns(df)
    while True:
        s = input(
            "WMW解析に使う「ラベル列」（クラスを表す列）の番号を 1 つ入力してください（例: 5）: "
        ).strip()
        try:
            idx = int(s)
        except ValueError:
            print("[WARN] 整数で列番号を指定してください。")
            continue
        if not (0 <= idx < len(df.columns)):
            print("[WARN] 範囲外の列番号です。")
            continue
        col = df.columns[idx]
        print(f"[INFO] ラベル列: {col}")
        return col


def ask_class_pairs(labels: List[Any]) -> List[Tuple[Any, Any]]:
    """クラスペアの選び方を対話的に決める。"""
    uniq = list(sorted(labels))
    print("\n=== クラス値の確認 ===")
    print("ラベル列のユニーク値（最大20件まで表示）:")
    for i, v in enumerate(uniq[:20]):
        print(f"  {i}: {v}")
    if len(uniq) > 20:
        print(f"  ...（他 {len(uniq) - 20} 種類）")

    from itertools import combinations

    while True:
        print(
            textwrap.dedent(
                """
                クラスペアの選び方を選択してください:
                  1) 全てのクラスペアを対象にする（C(n,2) 通り）
                  2) 特定のクラスペアのみ指定する
                """
            ).strip()
        )
        mode = input("番号を入力してください [1-2]（Enter=1）: ").strip()
        if not mode:
            mode = "1"
        if mode == "1":
            pairs = list(combinations(uniq, 2))
            print("[INFO] 対象クラスペア:")
            for p in pairs:
                print(f"  - {p}")
            return pairs
        elif mode == "2":
            print(
                "比較したいクラスペアをカンマ区切りで指定してください（例: 10-20,10-30）\n"
                "ラベル値には上記のユニーク値をそのまま指定してください。"
            )
            s = input("入力: ").strip()
            if not s:
                print("[WARN] 空の入力です。やり直してください。")
                continue
            pairs: List[Tuple[Any, Any]] = []
            for part in s.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" not in part:
                    print(f"[WARN] '{part}' は 'A-B' 形式ではありません。スキップします。")
                    continue
                a_str, b_str = part.split("-", 1)
                a_str, b_str = a_str.strip(), b_str.strip()
                # 文字列として比較する（int でも str でも OK）
                # 入力とラベルの型が違う場合があるので、文字列表現でマッチング
                def match_label(token: str) -> Optional[Any]:
                    # 完全一致を優先
                    for v in uniq:
                        if str(v) == token:
                            return v
                    return None

                a_val = match_label(a_str)
                b_val = match_label(b_str)
                if a_val is None or b_val is None:
                    print(f"[WARN] '{part}' に含まれるクラス値がユニーク値と一致しません。スキップします。")
                    continue
                if a_val == b_val:
                    print(f"[WARN] 同じ値同士のペア '{part}' はスキップします。")
                    continue
                # 順序を固定（小さい方を group1 とする）
                if str(a_val) <= str(b_val):
                    pairs.append((a_val, b_val))
                else:
                    pairs.append((b_val, a_val))
            pairs = sorted(set(pairs))
            if not pairs:
                print("[WARN] 有効なクラスペアがありません。再入力してください。")
                continue
            print("[INFO] 対象クラスペア:")
            for p in pairs:
                print(f"  - {p}")
            return pairs
        else:
            print("[WARN] 1 または 2 を入力してください。")


def ask_numeric_feature_columns(df: pd.DataFrame, label_col: str) -> List[str]:
    """数値特徴量列を選択させる。"""
    numeric_cols = [c for c in df.columns if is_numeric_series(df[c]) and c != label_col]
    if not numeric_cols:
        print("[ERROR] ラベル列以外に数値列がありません。")
        sys.exit(1)

    print("\n=== 数値特徴量列の候補 ===")
    for i, c in enumerate(numeric_cols):
        print(f"[{i:3d}] {c}")
    print("====================")
    print(
        textwrap.dedent(
            """
            解析対象とする列番号をカンマ区切りで入力してください
              例: 0,2,5 / 2-6 / 1-3,7
              all または空 Enter: 上記の全列を対象にする
            """
        ).strip()
    )
    sel = input("入力: ")
    indices = parse_column_selection(sel, len(numeric_cols) - 1)
    selected = [numeric_cols[i] for i in indices]
    print("[INFO] 対象特徴量列:")
    for c in selected:
        print(f"  - {c}")
    return selected


@dataclass
class WMWSettings:
    alternative: str
    max_sample_per_group: int
    min_n_per_group: int
    calc_cliffs_delta: bool

def ask_auc_threshold(default: float = 0.7) -> float:
    """AUCのしきい値を尋ねる。空Enterならデフォルト値を返す。"""
    print("\n[AUCフィルタ設定]")
    print("  AUC のしきい値を決めて、閾値以上と閾値未満で CSV を分けて保存します。")
    print(f"  何も入力せず Enter すると、デフォルト値 {default:.2f} を使います。")
    print("  ※ 0.5〜1.0 の範囲で指定してください。")

    s = input(f"  AUC しきい値 [default={default:.2f}]: ").strip()
    if not s:
        print(f"  -> デフォルト値 {default:.2f} を使用します。")
        return default
    try:
        val = float(s)
    except ValueError:
        print(f"[WARN] 数値として解釈できませんでした。デフォルト値 {default:.2f} を使用します。")
        return default
    if not (0.5 <= val <= 1.0):
        print(f"[WARN] 0.5〜1.0 の範囲外です。デフォルト値 {default:.2f} を使用します。")
        return default
    return val


def ask_wmw_settings() -> WMWSettings:
    """WMW 検定の設定を対話的に取得する。"""
    # alternative
    while True:
        s = input(
            "検定の方向を選んでください [two-sided/greater/less]（Enter=two-sided）: "
        ).strip()
        if not s:
            s = "two-sided"
        if s not in ("two-sided", "greater", "less"):
            print("[WARN] two-sided / greater / less のいずれかを指定してください。")
            continue
        alternative = s
        break

    # max_sample_per_group
    while True:
        s = input(
            "各グループからの最大サンプル数 (0 = 全件使用) [default=50000]: "
        ).strip()
        if not s:
            max_sample = 50000
            break
        try:
            v = int(s)
            if v < 0:
                raise ValueError
            max_sample = v
            break
        except ValueError:
            print("[WARN] 0 以上の整数を入力してください。")

    # min_n_per_group
    while True:
        s = input(
            "最小サンプル数 min_n_per_group [default=30]: "
        ).strip()
        if not s:
            min_n = 30
            break
        try:
            v = int(s)
            if v <= 0:
                raise ValueError
            min_n = v
            break
        except ValueError:
            print("[WARN] 正の整数を入力してください。")

    # Cliff's delta
    s = input("Cliff's delta を計算しますか？ [y/N]: ").strip().lower()
    calc_delta = s == "y"

    print(
        f"[INFO] alternative={alternative}, "
        f"max_sample_per_group={max_sample}, "
        f"min_n_per_group={min_n}, "
        f"Cliff's delta={'ON' if calc_delta else 'OFF'}"
    )
    return WMWSettings(
        alternative=alternative,
        max_sample_per_group=max_sample,
        min_n_per_group=min_n,
        calc_cliffs_delta=calc_delta,
    )


def run_wmw_pairwise_mode(df: pd.DataFrame, input_path: Path) -> None:
    """モード2: ペアワイズ WMW + AUC 解析。"""
    if not HAS_SCIPY:
        print(
            "[ERROR] scipy がインポートできません。\n"
            "mamba あるいは pip で scipy をインストールしてから再実行してください。\n"
            "  例) mamba install scipy\n"
        )
        sys.exit(1)

    # ラベル列の選択
    label_col = ask_label_column(df)
    label_series = df[label_col]
    labels = label_series.dropna().unique().tolist()
    if len(labels) < 2:
        print("[ERROR] ラベル列に 2 種類以上の値が必要です。")
        sys.exit(1)

    # クラスペア
    pairs = ask_class_pairs(labels)

    # 特徴量列（数値）選択
    feature_cols = ask_numeric_feature_columns(df, label_col)

    # WMW 設定
    settings = ask_wmw_settings()

    # 出力フォルダ
    out_dir = prepare_output_dir(input_path, suffix="WMW")

    # 結果格納
    rows: List[Dict[str, Any]] = []

    rng = np.random.default_rng()

    print("\n[INFO] 特徴量 × クラスペアごとに WMW + AUC を計算します。")

    for feat in feature_cols:
        print(f"\n=== 特徴量 '{feat}' ===")
        x = pd.to_numeric(df[feat], errors="coerce")
        for g1, g2 in pairs:
            mask1 = label_series == g1
            mask2 = label_series == g2
            s1 = x[mask1].dropna().to_numpy()
            s2 = x[mask2].dropna().to_numpy()

            n1, n2 = len(s1), len(s2)
            row: Dict[str, Any] = {
                "label_col": label_col,
                "feature": feat,
                "group1": g1,
                "group2": g2,
                "n_group1": n1,
                "n_group2": n2,
                "mean_group1": float(np.mean(s1)) if n1 > 0 else np.nan,
                "mean_group2": float(np.mean(s2)) if n2 > 0 else np.nan,
                "median_group1": float(np.median(s1)) if n1 > 0 else np.nan,
                "median_group2": float(np.median(s2)) if n2 > 0 else np.nan,
            }

            if n1 < settings.min_n_per_group or n2 < settings.min_n_per_group:
                msg = f"n too small (n1={n1}, n2={n2} < {settings.min_n_per_group})"
                print(f"[WARN] ({g1} vs {g2}): {msg} → スキップ")
                row.update(
                    {
                        "u_stat": np.nan,
                        "p_value": np.nan,
                        "auc": np.nan,
                        "cliffs_delta": np.nan,
                        "note": msg,
                    }
                )
                rows.append(row)
                continue

            # サンプリング
            if settings.max_sample_per_group > 0:
                if n1 > settings.max_sample_per_group:
                    s1 = rng.choice(s1, size=settings.max_sample_per_group, replace=False)
                    n1 = len(s1)
                if n2 > settings.max_sample_per_group:
                    s2 = rng.choice(s2, size=settings.max_sample_per_group, replace=False)
                    n2 = len(s2)

            # WMW
            try:
                res = mannwhitneyu(s1, s2, alternative=settings.alternative)
                u_stat = float(res.statistic)
                p_value = float(res.pvalue)
            except Exception as e:
                msg = f"mannwhitneyu error: {e}"
                print(f"[ERROR] ({g1} vs {g2}): {msg}")
                row.update(
                    {
                        "u_stat": np.nan,
                        "p_value": np.nan,
                        "auc": np.nan,
                        "cliffs_delta": np.nan,
                        "note": msg,
                    }
                )
                rows.append(row)
                continue

            # AUC (s1 が group1, s2 が group2)
            # U / (n1*n2) が P(X1 > X2) + 0.5 P(=) に対応
            auc = u_stat / (n1 * n2)

            # Cliff's delta（近似）
            if settings.calc_cliffs_delta:
                delta = 2.0 * auc - 1.0
            else:
                delta = np.nan

            row.update(
                {
                    "u_stat": u_stat,
                    "p_value": p_value,
                    "auc": auc,
                    "cliffs_delta": delta,
                    "note": "",
                }
            )
            print(f"  ({g1} vs {g2}): n1={n1}, n2={n2}, U={u_stat:.3g}, p={p_value:.3g}, AUC={auc:.3f}")
            rows.append(row)

    if not rows:
        print("[WARN] 有効な結果が 1 行もありませんでした。")
        return

    result_df = pd.DataFrame(rows)
    long_path = out_dir / "wmw_results_long.csv"
    result_df.to_csv(long_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] WMW 結果（ロング形式）CSV を保存しました: {long_path}")

    # AUC 行列 (feature × classpair)
    # 列名は "g1_vs_g2"
    mat_df = result_df.copy()
    mat_df["class_pair"] = mat_df.apply(
        lambda r: f"{r['group1']}_vs_{r['group2']}", axis=1
    )
    auc_matrix = mat_df.pivot_table(
        index="feature",
        columns="class_pair",
        values="auc",
        aggfunc="mean",
    )
    auc_matrix_path = out_dir / "auc_matrix_by_feature.csv"
    auc_matrix.to_csv(auc_matrix_path, encoding="utf-8-sig")
    print(f"[OK] AUC 行列 CSV を保存しました: {auc_matrix_path}")

    # AUC しきい値でフィルタした結果を出力
    thr = ask_auc_threshold()

    # 閾値以上
    high_auc = result_df[result_df["auc"] >= thr].copy()
    if high_auc.empty:
        print(f"[INFO] AUC >= {thr:.3f} の行は 1 件もありませんでした。")
    else:
        filt_ge_path = out_dir / f"wmw_results_auc_ge_{thr:.2f}.csv"
        high_auc.to_csv(filt_ge_path, index=False, encoding="utf-8-sig")
        print(
            f"[OK] AUC >= {thr:.3f} の結果だけを集めた CSV を保存しました: {filt_ge_path}"
        )

        # 上位の組み合わせをログに要約表示
        print("\n[SUMMARY] AUC が高い組み合わせ（上位 20 件まで表示・出力は全件）:")
        top = high_auc.sort_values("auc", ascending=False).head(20)
        for _, r in top.iterrows():
            print(
                f"  {r['feature']} | {r['group1']} vs {r['group2']} | "
                f"AUC={r['auc']:.3f}, p={r['p_value']:.3g}, "
                f"n1={int(r['n_group1'])}, n2={int(r['n_group2'])}"
            )

    # 閾値未満
    low_auc = result_df[result_df["auc"] < thr].copy()
    if low_auc.empty:
        print(f"[INFO] AUC < {thr:.3f} の行は 1 件もありませんでした。")
    else:
        filt_lt_path = out_dir / f"wmw_results_auc_lt_{thr:.2f}.csv"
        low_auc.to_csv(filt_lt_path, index=False, encoding="utf-8-sig")
        print(
            f"[OK] AUC < {thr:.3f} の結果だけを集めた CSV を保存しました: {filt_lt_path}"
        )

        # AUC が低い組み合わせをログに要約表示（下位 20 件）
        print("\n[SUMMARY] AUC が低い組み合わせ（下位 20 件まで表示・出力は全件）:")
        top_low = low_auc.sort_values("auc", ascending=True).head(20)
        for _, r in top_low.iterrows():
            print(
                f"  {r['feature']} | {r['group1']} vs {r['group2']} | "
                f"AUC={r['auc']:.3f}, p={r['p_value']:.3g}, "
                f"n1={int(r['n_group1'])}, n2={int(r['n_group2'])}"
            )

    print("\n=== モード2: ペアワイズ WMW + AUC 解析 完了 ===")


# ============================================================
# メイン
# ============================================================


def main() -> None:
    print("=== 特徴量分布チェックツール ===\n")

    # 入力ファイル
    input_path = ask_file_path()
    df = load_table(input_path)

    # モード選択
    print(
        textwrap.dedent(
            """
            モードを選んでください:
              1) 単変量分布の確認（ヒストグラム + 統計量 / カテゴリ頻度）
              2) ペアワイズ WMW + AUC 解析（2クラス間の特徴量の効き具合を定量評価）
              0) 終了
            """
        ).strip()
    )
    while True:
        mode = input("番号を入力してください [0-2]: ").strip()
        if not mode:
            mode = "1"
        if mode not in ("0", "1", "2"):
            print("[WARN] 0 / 1 / 2 のいずれかを入力してください。")
            continue
        break

    if mode == "0":
        print("終了します。")
        return
    elif mode == "1":
        run_hist_mode(df, input_path)
    elif mode == "2":
        run_wmw_pairwise_mode(df, input_path)


if __name__ == "__main__":
    main()