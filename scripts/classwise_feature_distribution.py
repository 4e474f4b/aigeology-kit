#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
クラス別 特徴量分布チェックツール

対象フォーマット:
  - CSV
  - Parquet
  - GPKG（GeoPackage）

機能概要:
  1) 入力ファイル（CSV / Parquet / GPKG）を DataFrame として読み込み
  2) 「クラス・階級ラベル」列を 1 つ指定
      - その列からユニークなクラスラベルを自動抽出
  3) 解析対象の数値列を選択（空 Enter で「すべて」）
  4) ヒストグラムの階級数（bin 数）を指定
  5) クラスごと × 数値列ごとに:
      - 基本統計量を計算して CSV にまとめて出力
      - ヒストグラム PNG を出力

統計量として出力する指標（ classwise_numeric_summary_stats.csv ）:
  - label:     クラスラベル
  - feature:   特徴量名（列名）
  - count:     有効データ数（NaN を除く）
  - total:     全行数（NaN 含む）
  - missing:   欠損値数
  - missing_ratio: 欠損率（missing / total）
  - mean:      平均
  - std:       標準偏差
  - min, max:  最小値・最大値
  - q05, q25, q50, q75, q95: 分位点（5, 25, 50, 75, 95%）
  - skew:      歪度（skewness）

ヒストグラム出力:
  - ファイル名: {feature}__class_{label_sanitized}_hist.png
    （クラスラベル中の / \ : * ? " < > | などは _ に置換）
"""

import os
import sys
import math
import re
import pandas as pd
import numpy as np

try:
    import cupy as cp  # type: ignore[import]
    HAS_CUPY = True
except Exception:
    cp = None  # type: ignore[assignment]
    HAS_CUPY = False

import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Any, Tuple


# =========================================================
# GPU / CPU 自動切り替えヘルパ
# =========================================================

def get_array_module() -> Tuple[Any, str]:
    """利用可能なら CuPy(GPU)、そうでなければ NumPy(CPU) を返す。
    戻り値: (xp, mode) で、mode は "gpu" または "cpu"。
    """
    if HAS_CUPY:
        try:
            # GPU デバイスが 1 台以上あるか確認
            ndev = cp.cuda.runtime.getDeviceCount()
            if ndev > 0:
                dev = cp.cuda.Device(0)
                dev.use()
                try:
                    name = dev.name
                except Exception:
                    name = "GPU"
                print(f"[INFO] CuPy GPU モードで実行します (device 0 = {name}, count={ndev}).")
                return cp, "gpu"
        except Exception as e:
            print(f"[WARN] CuPy は import されていますが GPU 初期化に失敗しました: {e}")
    print("[INFO] NumPy CPU モードで実行します。")
    return np, "cpu"

def print_environment_info(mode: str) -> None:
    """OS / Python / 計算環境 をまとめて表示するヘルパ関数。

    Parameters
    ----------
    mode : {"gpu", "cpu"}
        get_array_module() から返ってくるモード文字列。
    """
    os_name = platform.system()
    os_release = platform.release()
    py_version = sys.version.split()[0]

    if mode == "gpu":
        compute_env = "GPU（CuPy 使用）"
    else:
        if HAS_CUPY:
            # CuPy 自体はインポート可能だが、GPU が使えない / 使っていないケース
            compute_env = "CPU（CuPy 実行）"
        else:
            compute_env = "CPU のみ（CuPy 未使用）"

    print("=== 実行環境情報 ===")
    print(f"- OS         : {os_name} {os_release}")
    print(f"- Python     : {py_version}")
    print(f"- 計算環境   : {compute_env}")
    print("====================\\n")



# =========================================================
# ユーティリティ関数
# =========================================================

def strip_quotes(s: str) -> str:
    """前後のシングル / ダブルクォートを削除"""
    if not isinstance(s, str):
        return s
    s = s.strip()
    if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ("'", '"')):
        return s[1:-1]
    return s


def safe_filename(s: str) -> str:
    """ファイル名に使えない文字を _ に置換"""
    s = str(s)
    # Windows 禁止文字: \ / : * ? " < > |
    return re.sub(r'[\\/:*?"<>|]', "_", s)


def detect_file_type(path: Path) -> str:
    """拡張子からファイルタイプを判定"""
    ext = path.suffix.lower()
    if ext == ".csv":
        return "csv"
    elif ext in (".parquet", ".pq"):
        return "parquet"
    elif ext in (".gpkg", ".geopackage"):
        return "gpkg"
    else:
        return "unknown"


def read_table_any(path: Path) -> pd.DataFrame:
    """CSV / Parquet / GPKG を DataFrame として読み込む"""
    ftype = detect_file_type(path)

    if ftype == "csv":
        print(f"[INFO] CSV として読み込み: {path}")
        df = pd.read_csv(path)
        return df

    elif ftype == "parquet":
        print(f"[INFO] Parquet として読み込み: {path}")
        df = pd.read_parquet(path)
        return df

    elif ftype == "gpkg":
        print(f"[INFO] GPKG として読み込み: {path}")
        try:
            import geopandas as gpd
        except ImportError:
            print("[ERROR] geopandas がインストールされていません。")
            print("        pip / conda / mamba などで geopandas をインストールしてください。")
            sys.exit(1)

        # レイヤーが 1 つだけである前提で読み込む
        # （複数レイヤーを使いたい場合は必要に応じて拡張）
        df = gpd.read_file(path)
        if "geometry" in df.columns:
            df = df.drop(columns=["geometry"])
        # geopandas.DataFrame -> pandas.DataFrame
        return pd.DataFrame(df)

    else:
        print(f"[ERROR] 未対応のファイル拡張子です: {path.suffix}")
        sys.exit(1)


def print_columns_with_dtypes(df: pd.DataFrame) -> None:
    """列名と dtype を一覧表示"""
    print("\n[INFO] 列一覧（index: dtype）")
    for i, (col, dtype) in enumerate(df.dtypes.items()):
        print(f"  {i:3d}: {col}  ({dtype})")


def parse_column_selection(user_input: str, candidates: List[str]) -> List[str]:
    """
    ユーザー入力文字列から実際に使用する列名リストを返す。

    対応している指定方法:
      - 単一 index:        "10"
      - index 範囲:        "2-30"  (両端とも含む)
      - 列名:              "WS03_RES01m_slope_deg_b1"
      - これらの組み合わせ: "2-10, 15, WS03_RES01m_slope_deg_b1"

    ※ index は「直前に表示した候補リストの index（0〜N-1）」です。
    空文字の場合は candidates 全体を返します。
    """
    user_input = user_input.strip()
    if not user_input:
        return candidates

    parts = [p.strip() for p in user_input.split(",") if p.strip()]
    selected_indices: List[int] = []

    # 列名 → index の対応表
    name_to_idx = {name: idx for idx, name in enumerate(candidates)}

    for part in parts:
        # 範囲指定 "a-b"
        if "-" in part:
            left, right = part.split("-", 1)
            left = left.strip()
            right = right.strip()
            try:
                start_idx = int(left)
                end_idx = int(right)
            except ValueError:
                print(f"[WARN] 範囲 '{part}' は index として解釈できません。無視します。")
                continue

            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx

            for idx in range(start_idx, end_idx + 1):
                if 0 <= idx < len(candidates):
                    selected_indices.append(idx)
                else:
                    print(
                        f"[WARN] index {idx} は有効範囲外です（0〜{len(candidates)-1}）。無視します。"
                    )
            continue

        # 単一 index かどうか
        try:
            idx = int(part)
        except ValueError:
            # index でなければ列名として扱う
            if part in name_to_idx:
                selected_indices.append(name_to_idx[part])
            else:
                print(f"[WARN] 列名 '{part}' は候補に存在しません。無視します。")
            continue

        # index の範囲チェック
        if 0 <= idx < len(candidates):
            selected_indices.append(idx)
        else:
            print(
                f"[WARN] index {idx} は有効範囲外です（0〜{len(candidates)-1}）。無視します。"
            )

    # 重複を除去しつつ順序を維持
    unique_indices: List[int] = []
    seen = set()
    for idx in selected_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    return [candidates[i] for i in unique_indices]


def compute_classwise_stats(
    df: pd.DataFrame,
    label_col: str,
    numeric_cols: list[str],
    xp: Any,
) -> pd.DataFrame:
    """クラス列 label_col ごとに numeric_cols の統計量を計算し、ロング形式 DataFrame を返す。
    xp には NumPy または CuPy を渡す。
    """
    results: list[dict] = []

    # groupby でクラスごとに処理
    groups = df.groupby(label_col, dropna=False)

    use_gpu = xp is not np

    for label_value, sub in groups:
        print(f"[INFO] クラス '{label_value}' を処理中... (行数: {len(sub)})")

        for feature in numeric_cols:
            series_all = sub[feature]
            total = len(series_all)
            series = series_all.dropna()

            rec = {
                "label": label_value,
                "feature": feature,
                "total": int(total),
                "count": int(series.size),
                "missing": int(total - series.size),
                "missing_ratio": float((total - series.size) / total) if total > 0 else float("nan"),
            }

            if series.empty:
                # データが空の場合
                rec.update(
                    {
                        "mean": np.nan,
                        "std": np.nan,
                        "min": np.nan,
                        "max": np.nan,
                        "q05": np.nan,
                        "q25": np.nan,
                        "q50": np.nan,
                        "q75": np.nan,
                        "q95": np.nan,
                        "skew": np.nan,
                    }
                )
            else:
                # GPU / CPU 共通のベクトル演算で統計量を計算
                arr = series.to_numpy(dtype=float)
                if use_gpu:
                    arr_x = xp.asarray(arr)
                else:
                    arr_x = arr  # NumPy 配列として扱う

                # 基本統計量
                mean = float(xp.mean(arr_x))
                std = float(xp.std(arr_x, ddof=1)) if arr_x.size > 1 else float("nan")
                amin = float(xp.min(arr_x))
                amax = float(xp.max(arr_x))

                # パーセンタイル
                q = xp.percentile(arr_x, [5, 25, 50, 75, 95])
                q05, q25, q50, q75, q95 = [float(v) for v in (q[0], q[1], q[2], q[3], q[4])]

                # 歪度 (skewness) を自前で計算
                if arr_x.size > 1 and std not in (0.0, -0.0):
                    # (x - mean)^3 の平均 / std^3
                    m3 = xp.mean((arr_x - mean) ** 3)
                    skew = float(m3 / (std ** 3))
                else:
                    skew = float("nan")

                rec.update(
                    {
                        "mean": mean,
                        "std": std,
                        "min": amin,
                        "max": amax,
                        "q05": q05,
                        "q25": q25,
                        "q50": q50,
                        "q75": q75,
                        "q95": q95,
                        "skew": skew,
                    }
                )

            results.append(rec)

    if not results:
        print("[WARN] 計算結果が空です。指定列やデータを確認してください。")
        return pd.DataFrame()

    stats_df = pd.DataFrame(results)
    stats_df = stats_df[
        [
            "label",
            "feature",
            "total",
            "count",
            "missing",
            "missing_ratio",
            "mean",
            "std",
            "min",
            "max",
            "q05",
            "q25",
            "q50",
            "q75",
            "q95",
            "skew",
        ]
    ]
    return stats_df


def export_stats_per_class(stats_df: pd.DataFrame, out_dir: Path) -> None:
    """classwise_numeric_summary_stats をクラスごとに分割しサブフォルダに CSV 出力する"""
    if stats_df.empty:
        return

    if "label" not in stats_df.columns:
        print("[WARN] 'label' 列が見つからないため、クラス別統計量の分割出力はスキップします。")
        return

    # クラスごとに out_dir / class_<label_safe> / summary_stats.csv を出力
    for label_value, sub in stats_df.groupby("label", dropna=False):
        # NaN クラスも別扱いで出力する
        if pd.isna(label_value):
            label_str = "NaN"
        else:
            label_str = str(label_value)

        label_safe = safe_filename(label_str)
        class_dir = out_dir / f"class_{label_safe}"
        class_dir.mkdir(parents=True, exist_ok=True)

        out_csv = class_dir / "summary_stats.csv"
        sub.to_csv(out_csv, index=False)
        print(f"[OK] クラス '{label_str}' の統計量 CSV を出力: {out_csv}")


def plot_histograms_per_class(
    df: pd.DataFrame,
    label_col: str,
    numeric_cols: list[str],
    bins: int,
    out_dir: Path,
    xp: Any,
) -> None:
    """クラス別 × 数値列別にヒストグラム PNG / CSV を出力する。
    xp には NumPy または CuPy を渡す。
    """
    groups = df.groupby(label_col, dropna=False)
    use_gpu = xp is not np

    for label_value, sub in groups:
        # クラスラベル文字列（NaN の場合も含め安全な文字列にする）
        if pd.isna(label_value):
            label_str = "NaN"
        else:
            label_str = str(label_value)

        # ファイル名に使えるようにサニタイズ
        # ※ 既存の safe_filename() を利用
        label_safe = safe_filename(label_str)

        # クラスごとの出力ディレクトリ
        class_dir = out_dir / f"class_{label_safe}"
        class_dir.mkdir(parents=True, exist_ok=True)

        for feature in numeric_cols:
            series = sub[feature].dropna()

            if series.empty:
                print(f"[WARN] クラス '{label_str}' / 特徴量 '{feature}' は有効データがありません。スキップします。")
                continue

            # --- ヒストグラム元データ（counts, bin_edges） ---
            arr = series.to_numpy(dtype=float)
            if use_gpu:
                arr_x = xp.asarray(arr)
                counts, bin_edges = xp.histogram(arr_x, bins=bins)
                # matplotlib / pandas 用に CPU 側に戻す
                counts = np.asarray(counts)
                bin_edges = np.asarray(bin_edges)
            else:
                counts, bin_edges = np.histogram(arr, bins=bins)

            bin_left = bin_edges[:-1]
            bin_right = bin_edges[1:]

            hist_df = pd.DataFrame(
                {
                    "bin_left": bin_left,
                    "bin_right": bin_right,
                    "count": counts,
                }
            )

            csv_name = f"{feature}__class_{label_safe}_hist.csv"
            csv_path = class_dir / csv_name
            hist_df.to_csv(csv_path, index=False)
            print(f"[OK] ヒストグラム元データ CSV 出力: {csv_path}")

            # --- ヒストグラム PNG 出力 ---
            plt.figure()
            # bins は bin_edges を使うことで CPU/GPU どちらでも同じ階級幅になる
            plt.hist(series, bins=bin_edges, edgecolor="black", alpha=0.7)
            plt.title(f"{feature} (class = {label_str})")
            plt.xlabel(feature)
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            png_name = f"{feature}__class_{label_safe}_hist.png"
            png_path = class_dir / png_name
            plt.savefig(png_path, dpi=150)
            plt.close()

            print(f"[OK] ヒストグラム出力: {png_path}")

# =========================================================
# メイン処理
# =========================================================

def main():
    print("=== クラス別 特徴量分布チェックツール ===")

    # 起動直後にCPU/GPUモードを表示
    xp, xp_mode = get_array_module()
    print_environment_info(xp_mode)


    in_path_str = input("入力ファイルのパス（CSV / Parquet / GPKG）: ").strip()
    in_path_str = strip_quotes(in_path_str)
    if not in_path_str:
        print("[ERROR] 入力ファイルが指定されていません。終了します。")
        return

    in_path = Path(in_path_str)
    if not in_path.exists():
        print(f"[ERROR] ファイルが見つかりません: {in_path}")
        return

    # 出力ディレクトリ
    default_out_dir = in_path.with_suffix("").name + "_classdist"
    out_dir_str = input(f"出力フォルダ名（空 = {default_out_dir}）: ").strip()
    out_dir_str = strip_quotes(out_dir_str) or default_out_dir

    out_dir = in_path.parent / out_dir_str
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 出力フォルダ: {out_dir}")

    # ファイル読み込み
    df = read_table_any(in_path)

    if df.empty:
        print("[ERROR] DataFrame が空です。入力データを確認してください。")
        return

    print_columns_with_dtypes(df)

    # クラス列の指定
    label_col = input("\n[必須] クラス / 階級ラベル列の名前または index を入力: ").strip()
    label_col = strip_quotes(label_col)

    if not label_col:
        print("[ERROR] クラス列が指定されていません。終了します。")
        return

    cols_list = list(df.columns)

    # index 指定か判定
    if label_col.isdigit():
        idx = int(label_col)
        if 0 <= idx < len(cols_list):
            label_col = cols_list[idx]
        else:
            print(f"[ERROR] 列 index {idx} は範囲外です。終了します。")
            return
    else:
        if label_col not in df.columns:
            print(f"[ERROR] 列名 '{label_col}' が見つかりません。終了します。")
            return

    print(f"[INFO] クラス列: {label_col}")

    # クラスラベルを表示
    unique_labels = df[label_col].drop_duplicates()
    print(f"[INFO] クラスラベル一覧（{len(unique_labels)} 種類）:")
    # 多すぎる場合は一部のみ表示
    max_show = 50
    for i, v in enumerate(unique_labels.head(max_show)):
        print(f"  {i:3d}: {v}")
    if len(unique_labels) > max_show:
        print(f"  ... 省略 ...（合計 {len(unique_labels)} クラス）")

    # 数値列の候補
    numeric_candidates = df.select_dtypes(include=["number"]).columns.tolist()
    if label_col in numeric_candidates:
        numeric_candidates.remove(label_col)

    if not numeric_candidates:
        print("[ERROR] クラス列以外に数値列が見つかりません。終了します。")
        return

    print("\n[INFO] 数値列候補（index: 列名）:")
    for i, c in enumerate(numeric_candidates):
        print(f"  {i:3d}: {c}")

    num_sel = input(
        "\n解析対象の数値列をカンマ区切りで指定してください（index / index-index / 列名, 空=すべて）: "
    )
    target_numeric_cols = parse_column_selection(num_sel, numeric_candidates)

    if not target_numeric_cols:
        print("[ERROR] 解析対象の数値列が 0 件です。終了します。")
        return

    print("[INFO] 解析対象の数値列:")
    for c in target_numeric_cols:
        print(f"  - {c}")

    # bin 数の指定
    bins_str = input("\nヒストグラムの階級数（bin 数, 空 = 30）: ").strip()
    if not bins_str:
        bins = 30
    else:
        try:
            bins = int(bins_str)
            if bins <= 0:
                raise ValueError()
        except ValueError:
            print("[WARN] bin 数が不正なため、30 を使用します。")
            bins = 30
    print(f"[INFO] ヒストグラムの bin 数: {bins}")

    # 統計量の計算
    print("\n[INFO] クラス別統計量を計算しています...")
    stats_df = compute_classwise_stats(df, label_col, target_numeric_cols, xp)

    if not stats_df.empty:
        stats_path = out_dir / "classwise_numeric_summary_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"[OK] 統計量 CSV 出力: {stats_path}")

        # クラス別の統計量をサブフォルダに分けて出力
        export_stats_per_class(stats_df, out_dir)
    else:
        print("[WARN] 統計量 DataFrame が空のため CSV 出力をスキップします。")

    # ヒストグラムの出力
    print("\n[INFO] クラス別ヒストグラムを作成しています...")
    plot_histograms_per_class(df, label_col, target_numeric_cols, bins, out_dir, xp)

    print("\n[INFO] すべての処理が完了しました。")


if __name__ == "__main__":
    main()
