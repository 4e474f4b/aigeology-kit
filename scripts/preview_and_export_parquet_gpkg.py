#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
parquet / gpkg / csv の一部を表示し、CSV にエクスポートするツール

【機能概要】
- 入力: GPKG / Parquet / CSV ファイル（対話式でパスを入力）
- ヘッダ情報（列名・型・行数の概算）と先頭 10 行を表示
- エクスポート方法を選択して CSV を出力
    1) 全件を 1 ファイルに CSV 出力
    2) 行番号順に N 分割して複数 CSV 出力（id列を追加）
    3) ランダムサンプリング（全体の一定割合を 1 ファイルに出力）

【注意】
- 数 GB ～ 数十 GB のファイルでは、メモリに載せられない場合があります。
- その場合はメモリ増設や、事前の空間クリップなどで対応してください。
"""

import os
import sys
import math
import platform
from typing import Tuple

import pandas as pd

try:
    import geopandas as gpd
except ImportError:
    gpd = None

# GPU (CuPy) チェック
try:
    import cupy as cp  # type: ignore
    try:
        _gpu_device_count = cp.cuda.runtime.getDeviceCount()  # type: ignore
        GPU_AVAILABLE = _gpu_device_count > 0
    except Exception:
        GPU_AVAILABLE = False
except ImportError:
    cp = None  # type: ignore
    GPU_AVAILABLE = False


# ==============================
# ユーティリティ関数
# ==============================

def print_environment_info() -> None:
    """OS / Python / CPU or GPU 利用可否など環境情報を表示する。"""
    print("=== 実行環境情報 ===")
    print(f"- OS         : {platform.system()} {platform.release()}")
    print(f"- Python     : {platform.python_version()}")
    if GPU_AVAILABLE:
        print("- 計算環境   : GPU (CuPy 利用可能)")
    else:
        print("- 計算環境   : CPU のみ（CuPy 未使用）")
    print("====================\n")


def strip_quotes(path: str) -> str:
    """パス文字列の前後に付いた ' または \" を取り除く。"""
    path = path.strip()
    if (path.startswith('"') and path.endswith('"')) or (
        path.startswith("'") and path.endswith("'")
    ):
        return path[1:-1]
    return path


def human_readable_size(size_bytes: int) -> str:
    """バイト数を MB / GB などに変換して返す。"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    for unit in ["KB", "MB", "GB", "TB"]:
        size_bytes /= 1024.0
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
    return f"{size_bytes:.2f} PB"


def detect_file_type(path: str) -> str:
    """拡張子からファイルタイプを判定する。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return "parquet"
    elif ext == ".gpkg":
        return "gpkg"
    elif ext == ".csv":
        return "csv"
    else:
        return "unknown"


# ==============================
# 読み込み処理
# ==============================

def load_head_only(path: str, file_type: str) -> Tuple[pd.DataFrame, int]:
    """
    先頭 10 行だけ読み込み、DataFrame と総行数（わからなければ -1）を返す。
    総行数が高速に取得できない場合は -1 を返す。
    """
    if file_type == "parquet":
        # parquet は一気読み + head
        try:
            df = pd.read_parquet(path)
            total_rows = len(df)
            head_df = df.head(10)
            return head_df, total_rows
        except Exception as e:
            raise RuntimeError(f"parquet 読み込み時にエラーが発生しました: {e}")

    elif file_type == "gpkg":
        if gpd is None:
            raise RuntimeError("geopandas がインストールされていないため、GPKG を読み込めません。")
        try:
            gdf = gpd.read_file(path)
            total_rows = len(gdf)
            head_df = gdf.head(10)
            return head_df, total_rows
        except Exception as e:
            raise RuntimeError(f"GPKG 読み込み時にエラーが発生しました: {e}")

    elif file_type == "csv":
        try:
            head_df = pd.read_csv(path, nrows=10)
            # CSV は行数カウントしようとすると重いので、ここでは -1 にしておく
            total_rows = -1
            return head_df, total_rows
        except Exception as e:
            raise RuntimeError(f"CSV 読み込み時にエラーが発生しました: {e}")

    else:
        raise ValueError("対応していないファイルタイプです。")


def load_full_dataframe(path: str, file_type: str) -> pd.DataFrame:
    """ファイル全体を DataFrame / GeoDataFrame として読み込む。"""
    if file_type == "parquet":
        print("[INFO] parquet を DataFrame として読み込みます（全件）...")
        df = pd.read_parquet(path)
        print(f"[INFO] 読み込み完了: {len(df):,} 行 × {len(df.columns):,} 列")
        return df

    elif file_type == "gpkg":
        if gpd is None:
            raise RuntimeError("geopandas がインストールされていないため、GPKG を読み込めません。")
        print("[INFO] GPKG を GeoDataFrame として読み込みます（全件）...")
        gdf = gpd.read_file(path)
        print(f"[INFO] 読み込み完了: {len(gdf):,} 行 × {len(gdf.columns):,} 列")
        return gdf

    elif file_type == "csv":
        print("[INFO] CSV を DataFrame として読み込みます（全件）...")
        df = pd.read_csv(path)
        print(f"[INFO] 読み込み完了: {len(df):,} 行 × {len(df.columns):,} 列")
        return df

    else:
        raise ValueError("対応していないファイルタイプです。")


# ==============================
# サンプリング・分割処理
# ==============================

def random_sample_indices(n_rows: int, frac: float, use_gpu: bool) -> pd.Index:
    """
    行数 n_rows から frac の割合でランダムサンプルするためのインデックスを返す。
    use_gpu=True かつ GPU_AVAILABLE=True の場合、CuPy でインデックスを生成。
    """
    if frac <= 0.0 or frac > 1.0:
        raise ValueError("サンプリング割合 frac は 0 < frac <= 1 で指定してください。")

    sample_n = max(1, int(n_rows * frac))
    if sample_n > n_rows:
        sample_n = n_rows

    if use_gpu and GPU_AVAILABLE and cp is not None:
        print(f"[INFO] GPU (CuPy) を用いてサンプリングインデックスを生成します (n={sample_n})")
        idx_gpu = cp.random.choice(n_rows, size=sample_n, replace=False)  # type: ignore
        idx_np = cp.asnumpy(idx_gpu)
        return pd.Index(idx_np)
    else:
        print(f"[INFO] CPU を用いてサンプリングインデックスを生成します (n={sample_n})")
        import numpy as np
        idx_np = np.random.choice(n_rows, size=sample_n, replace=False)
        return pd.Index(idx_np)


def export_csv_split(df: pd.DataFrame, out_dir: str, base_name: str, n_splits: int) -> None:
    """
    行番号順に n_splits 個の CSV に分割して出力する。

    - 左端に id 列を追加（1 〜 総行数）
    - 各分割ファイルには、対応する id が含まれる
    """
    n_rows = len(df)
    if n_splits < 2:
        print("[WARN] 分割数は 2 以上を指定してください。処理を中止します。")
        return

    # 念のためインデックスを 0 ～ n_rows-1 に振り直しておく
    df = df.reset_index(drop=True)

    chunk_size = int(math.ceil(n_rows / n_splits))
    print(f"[INFO] 総行数: {n_rows:,} 行 を {n_splits} 分割（1ファイルあたり最大 {chunk_size:,} 行）")

    os.makedirs(out_dir, exist_ok=True)

    for i in range(n_splits):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_rows)
        if start >= end:
            break

        # このチャンクに対応する id を 1-origin で振る
        # 例: 全体 40000 行で start=0,end=10000 → id=1〜10000
        #     start=10000,end=20000 → id=10001〜20000
        part_df = df.iloc[start:end].copy()
        id_values = range(start + 1, end + 1)
        part_df.insert(0, "id", id_values)

        out_path = os.path.join(out_dir, f"{base_name}_part{i+1:02d}.csv")
        print(f"[INFO] 出力中: {out_path} (id={start+1:,} ～ {end:,})")
        part_df.to_csv(out_path, index=False)

    print("[OK] 分割 CSV 出力が完了しました。")


def export_csv_random(df: pd.DataFrame, out_dir: str, base_name: str, frac: float, use_gpu: bool) -> None:
    """全体の frac 割合をランダムサンプリングして CSV を 1 ファイル出力する。"""
    n_rows = len(df)
    idx = random_sample_indices(n_rows, frac, use_gpu=use_gpu)
    sampled = df.iloc[idx]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base_name}_random_{frac:.4f}.csv")
    print(f"[INFO] ランダムサンプル {len(sampled):,} 行を {out_path} に出力します。")
    sampled.to_csv(out_path, index=False)
    print("[OK] ランダムサンプル CSV 出力が完了しました。")


# ==============================
# メイン処理フロー
# ==============================

def main() -> None:
    print_environment_info()

    print("=== parquet / gpkg / csv 部分表示 & CSV エクスポートツール ===")
    print("gpkg / parquet / csv のいずれか 1 ファイルを指定してください。")

    in_path = strip_quotes(input("入力ファイルのパス（空で終了）: ").strip())
    if not in_path:
        print("入力が空のため終了します。")
        return

    if not os.path.isfile(in_path):
        print(f"[ERROR] ファイルが見つかりません: {in_path}")
        return

    file_type = detect_file_type(in_path)
    if file_type == "unknown":
        print("[ERROR] .gpkg / .parquet / .csv のみ対応しています。")
        return

    size_bytes = os.path.getsize(in_path)
    print(f"[INFO] 入力ファイル: {in_path}")
    print(f"[INFO] ファイルサイズ: {human_readable_size(size_bytes)}")

    # まずは先頭 10 行だけ読み込んで概要を表示
    try:
        head_df, total_rows_est = load_head_only(in_path, file_type)
    except Exception as e:
        print(f"[ERROR] ヘッダ読み込み中にエラーが発生しました: {e}")
        return

    print("\n=== 列情報（ヘッダ） ===")
    print(f"列数: {len(head_df.columns)} 列")
    print("列名と推定 dtype:")
    for col in head_df.columns:
        print(f" - {col}: {head_df[col].dtype}")

    if total_rows_est >= 0:
        print(f"\n推定総行数: {total_rows_est:,} 行")
    else:
        print("\n推定総行数: 不明")

    print("\n=== 先頭 10 行プレビュー ===")
    with pd.option_context("display.max_columns", 50, "display.width", 200):
        print(head_df.head(10))

    # フルデータ読み込み
    try:
        df_full = load_full_dataframe(in_path, file_type)
    except Exception as e:
        print(f"[ERROR] フルデータ読み込み中にエラーが発生しました: {e}")
        return

    n_rows = len(df_full)

    print("\n=== エクスポート方法を選択してください ===")
    print("  1) 全件を 1 つの CSV に出力")
    print("  2) 行番号順に N 分割して CSV 出力（id列を追加）")
    print("  3) ランダムサンプリングで 1 つの CSV 出力")
    print("  0) 何も出力せず終了")
    print("[注意] CSV エクスポートにはファイル全体をメモリに読み込みます。")
    mode = input("番号を選んでください [0-3]: ").strip()

    if mode == "0":
        print("CSV エクスポートは行わず終了します。")
        return

    # 出力ディレクトリとベース名（mode==0 以外）
    default_out_dir = os.path.dirname(os.path.abspath(in_path))
    out_dir = strip_quotes(input(f"\n出力ディレクトリ（空で入力ファイルと同じ: {default_out_dir}）: ").strip())
    if not out_dir:
        out_dir = default_out_dir

    base_name_default = os.path.splitext(os.path.basename(in_path))[0]
    base_name = input(f"出力ファイルのベース名（拡張子不要 / 空で {base_name_default}）: ").strip()
    if not base_name:
        base_name = base_name_default

    if mode == "1":
        # 全件出力
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base_name}.csv")
        print(f"[INFO] 全 {n_rows:,} 行を {out_path} に出力します。")
        try:
            df_full.to_csv(out_path, index=False)
            print("[OK] 全件 CSV 出力が完了しました。")
        except Exception as e:
            print(f"[ERROR] CSV 出力中にエラーが発生しました: {e}")
        return

    elif mode == "2":
        # N 分割出力（id 列付き）
        try:
            n_splits_str = input("何分割しますか？（例: 4）: ").strip()
            n_splits = int(n_splits_str)
        except ValueError:
            print("[ERROR] 整数値として解釈できませんでした。処理を中止します。")
            return

        try:
            export_csv_split(df_full, out_dir, base_name, n_splits)
        except Exception as e:
            print(f"[ERROR] 分割 CSV 出力中にエラーが発生しました: {e}")
        return

    elif mode == "3":
        # ランダムサンプリング出力
        frac_str = input("サンプリング割合を指定してください（例: 0.1 または 1/10）: ").strip()
        try:
            if "/" in frac_str:
                num_str, den_str = frac_str.split("/", 1)
                num = float(num_str)
                den = float(den_str)
                frac = num / den
            else:
                frac = float(frac_str)

            if frac <= 0 or frac > 1:
                raise ValueError
        except Exception:
            print("[ERROR] サンプリング割合は 0 < 値 <= 1 で指定してください（例: 0.1 または 1/10）。")
            return

        use_gpu = GPU_AVAILABLE
        if GPU_AVAILABLE:
            ans = input("GPU (CuPy) を使ってインデックスを生成しますか？ [Y/n]: ").strip().lower()
            if ans and not ans.startswith("y"):
                use_gpu = False

        try:
            export_csv_random(df_full, out_dir, base_name, frac, use_gpu=use_gpu)
        except Exception as e:
            print(f"[ERROR] ランダムサンプル CSV 出力中にエラーが発生しました: {e}")
        return

    else:
        print("[ERROR] 不正な番号が指定されました。処理を中止します。")
        return


if __name__ == "__main__":
    main()
