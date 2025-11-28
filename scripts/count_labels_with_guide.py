#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
指定列のカテゴリ件数・比率を効率的に集計するツール（条件付き）

- 対応フォーマット:
    - CSV
    - Parquet
    - GPKG（GeoPackage）

- 特徴:
    - 入力ファイルは 1 回だけ読み込んで集計（逐次 for ループは使わない）
        - CSV / Parquet: pandas のベクトル演算 + value_counts
        - GPKG: SQLite に直接 GROUP BY クエリを投げて集計
    - 検索対象列を選択した際に、その列の推定型・サンプル値・
      条件指定のヒントを表示
    - 列の型（数値 / 文字列 / 日時など）に応じて、
      条件設定で選べる演算子を自動で制限

- 典型的な用途:
    - ランダムフォレストの正解ラベル（地形分類コードなど）の
      出現比率を事前に把握しておく
"""

import os
import sys
import sqlite3
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

try:
    import pyarrow.parquet as pq  # noqa: F401
    _HAS_PYARROW = True
except Exception:
    _HAS_PYARROW = False


# =========================================================
# 共通ヘルパ
# =========================================================

def strip_quotes(s: str) -> str:
    """パスなどの前後に付いた " や ' を削除する。"""
    return s.strip().strip('"').strip("'")


def ask_yes_no(prompt: str, default: bool | None = None) -> bool:
    """
    y/n を聞く簡易プロンプト。
    default:
        True  → [Y/n] で Enter = Yes
        False → [y/N] で Enter = No
        None  → [y/n] で必ず入力
    """
    while True:
        if default is None:
            s = input(f"{prompt} [y/n]: ").strip().lower()
        elif default:
            s = input(f"{prompt} [Y/n]: ").strip().lower()
            if s == "":
                return True
        else:
            s = input(f"{prompt} [y/N]: ").strip().lower()
            if s == "":
                return False

        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False
        print("  'y' か 'n' で答えてください。")


def choose_column(columns: List[str], title: str = "カラム選択") -> str:
    """カラム一覧を表示して 1 つ選ばせる。"""
    print(f"\n=== {title} ===")
    for i, c in enumerate(columns):
        print(f"[{i:03d}] {c}")
    max_idx = len(columns) - 1

    while True:
        idx_s = input(f"使用するカラム番号を入力してください [0-{max_idx}]: ").strip()
        try:
            idx = int(idx_s)
        except ValueError:
            print("  整数で入力してください。")
            continue
        if 0 <= idx <= max_idx:
            return columns[idx]
        print("  範囲外です。")


# =========================================================
# 型判定 / ガイド表示（pandas 用）
# =========================================================

def get_kind_from_dtype(dtype) -> str:
    """
    pandas の dtype から「型カテゴリ」を判定する。
    戻り値は 'numeric' / 'string' / 'datetime' / 'other' のいずれか。
    """
    if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype) \
            or pd.api.types.is_object_dtype(dtype):
        return "string"
    return "other"


def print_column_guide_pandas(df: pd.DataFrame, col: str) -> None:
    """pandas 用: 列の型とサンプル値、条件指定のヒントを表示する。"""
    s = df[col]
    dtype = s.dtype
    kind = get_kind_from_dtype(dtype)

    samples = s.dropna().unique()[:5]

    print("\n--- 列の型ガイド（pandas） ---")
    print(f"列名   : {col}")
    print(f"dtype  : {dtype}  → 推定型: {kind}")
    print(f"サンプル値: {list(samples)}")

    print("\n[条件指定のヒント]")
    if kind == "numeric":
        print("  - 等しい:          == 10")
        print("  - 範囲（between）: 5 ～ 30")
        print("  - 候補リスト:      10, 20, 30 など（in）")
    elif kind == "datetime":
        print("  - 等しい:          2025-01-01")
        print("  - 範囲:            2025-01-01 ～ 2025-01-31（between）")
        print("  ※ CSV の場合は、事前に日時型へ変換済みであることを推奨します。")
    elif kind == "string":
        print("  - 等しい:          平野, 谷底平野 など")
        print("  - 候補リスト:      平野, 山地 など（in）")
        print("  - ≠ 条件:          != 山地")
    else:
        print("  - 型が混在している可能性があります。少し注意して条件を指定してください。")


# =========================================================
# 型判定 / ガイド表示（SQLite/GPKG 用）
# =========================================================

def get_kind_from_sql_type(sql_type: str) -> str:
    """
    SQLite/GPKG の列型（TEXT, INTEGER, REAL, ...）から型カテゴリを判定。
    戻り値は 'numeric' / 'string' / 'datetime' / 'other' のいずれか。
    """
    if sql_type is None:
        return "other"
    t = sql_type.upper()
    if "INT" in t or "REAL" in t or "DOUBLE" in t or "FLOAT" in t or "NUM" in t:
        return "numeric"
    if "DATE" in t or "TIME" in t:
        return "datetime"
    if "CHAR" in t or "TEXT" in t or "CLOB" in t:
        return "string"
    return "other"


def print_column_guide_sqlite(conn: sqlite3.Connection, table: str, col: str, sql_type: str) -> None:
    """SQLite 用: 列の型とサンプル値、条件指定のヒントを表示する。"""
    kind = get_kind_from_sql_type(sql_type)

    cur = conn.cursor()
    try:
        cur.execute(f'SELECT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL LIMIT 5')
        rows = cur.fetchall()
        samples = [r[0] for r in rows]
    except Exception:
        samples = []

    print("\n--- 列の型ガイド（SQLite / GPKG） ---")
    print(f"列名      : {col}")
    print(f"SQL 型    : {sql_type}  → 推定型: {kind}")
    print(f"サンプル値: {samples}")

    print("\n[条件指定のヒント]")
    if kind == "numeric":
        print("  - 等しい:          == 10")
        print("  - 範囲（between）: 5 ～ 30")
        print("  - 候補リスト:      10, 20, 30 など（in）")
    elif kind == "datetime":
        print("  - 等しい:          2025-01-01")
        print("  - 範囲:            2025-01-01 ～ 2025-01-31（between）")
        print("  ※ DATE/TIME 型の扱いはテーブル定義に依存します。")
    elif kind == "string":
        print("  - 等しい:          平野, 谷底平野 など")
        print("  - 候補リスト:      平野, 山地 など（in）")
        print("  - ≠ 条件:          != 山地")
    else:
        print("  - 型が混在している可能性があります。少し注意して条件を指定してください。")


# =========================================================
# 条件指定まわり（共通）
# =========================================================

def choose_basic_pattern_for_target(target_col: str) -> List[Dict[str, Any]] | None:
    """
    対象列そのものに対する「基本パターン」を選ぶメニュー。

    戻り値:
      - []       : 条件なし（全件）
      - [cond..] : target_col に対する in 条件 1 個
      - None     : 詳細条件モード（build_conditions_interactive を呼ぶ）
    """
    print("\n=== 抽出条件の設定（基本パターン） ===")
    print(f"対象列: {target_col}")
    print("この列に対する集計パターンを選んでください:")
    print("  1) この列の全カテゴリを集計（条件なし・value_counts）")
    print("  2) この列の「指定した値リスト」だけを集計（in 条件）")
    print("  3) 別の列も含めて、詳細な条件を設定する（高度なモード）")

    while True:
        choice = input("番号を入力 [1-3]: ").strip()
        if choice not in ("1", "2", "3"):
            print("  1〜3 の番号で入力してください。")
            continue
        break

    if choice == "1":
        # 条件なし
        print("  → 条件なしで全件を対象に集計します。")
        return []

    if choice == "2":
        # target_col に対する in 条件を 1 つだけ作る
        print(f"\n対象列 '{target_col}' に対して in 条件を指定します。")
        raw = input("候補値をカンマ区切りで入力してください（例: 1010101,1010201,2010101）: ").strip()
        values = [v.strip() for v in raw.split(",") if v.strip() != ""]
        if not values:
            print("  値が入力されていないため、条件なし扱いにします。")
            return []
        cond = {"col": target_col, "op": "in", "values": values}
        print(f"  → 設定された条件: {cond['col']} {cond['op']} {cond['values']}")
        return [cond]

    # choice == "3": 詳細条件モード
    print("  → 別の列も含めた詳細条件モードに進みます。")
    return None


def build_conditions_interactive(columns: List[str], kind_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    抽出条件を対話式で複数指定する（詳細モード）。
    kind_map: カラム名 → 'numeric' / 'string' / 'datetime' / 'other'
    """
    print("\n=== 抽出条件の詳細設定（複数列・複数条件） ===")
    conds: List[Dict[str, Any]] = []

    while True:
        print("\n[条件を 1 つ追加]")
        col = choose_column(columns, title="条件に使うカラムの選択")
        kind = kind_map.get(col, "other")

        # 型に応じて演算子候補を制限
        if kind == "numeric":
            ops_menu = [
                ("1", "==（等しい）", "=="),
                ("2", "!=（等しくない）", "!="),
                ("3", " >（より大きい）", ">"),
                ("4", " >=（以上）", ">="),
                ("5", " <（より小さい）", "<"),
                ("6", " <=（以下）", "<="),
                ("7", "between（範囲指定）", "between"),
                ("8", "in（候補のいずれか）", "in"),
            ]
        elif kind == "string":
            ops_menu = [
                ("1", "==（等しい）", "=="),
                ("2", "!=（等しくない）", "!="),
                ("8", "in（候補のいずれか）", "in"),
            ]
        elif kind == "datetime":
            ops_menu = [
                ("1", "==（等しい）", "=="),
                ("7", "between（日付範囲）", "between"),
            ]
        else:
            # よくわからない型 → ひとまず ==, !=, in のみに絞る
            ops_menu = [
                ("1", "==（等しい）", "=="),
                ("2", "!=（等しくない）", "!="),
                ("8", "in（候補のいずれか）", "in"),
            ]

        print(f"  対象カラム: {col}（推定型: {kind}）")
        print("  使用する演算子を選んでください:")
        for key, label, _ in ops_menu:
            print(f"    {key}) {label}")

        choice = input("  番号を入力: ").strip()
        op = None
        for key, _, o in ops_menu:
            if choice == key:
                op = o
                break
        if op is None:
            print("  ⚠ 無効な番号です。この条件はスキップします。")
            if not ask_yes_no("別の条件を続けますか？", default=True):
                break
            else:
                continue

        # 値を入力
        if op == "between":
            v1 = input("  下限値（v1）: ").strip()
            v2 = input("  上限値（v2）: ").strip()
            conds.append({"col": col, "op": op, "values": [v1, v2]})
        elif op == "in":
            v = input("  候補値をカンマ区切りで入力してください（例: 1010101,1010201,2010101）: ").strip()
            values = [x.strip() for x in v.split(",") if x.strip() != ""]
            conds.append({"col": col, "op": op, "values": values})
        else:
            v = input("  比較する値: ").strip()
            conds.append({"col": col, "op": op, "values": [v]})

        if not ask_yes_no("さらに条件を追加しますか？", default=False):
            break

    print("\n[INFO] 設定された条件:")
    if not conds:
        print("  （なし）")
    else:
        for i, c in enumerate(conds):
            print(f"  ({i+1}) {c['col']} {c['op']} {c['values']}")
    return conds


# =========================================================
# pandas 側の条件適用（CSV / Parquet 用）
# =========================================================

def _try_cast_value_to_dtype(value: str, series: pd.Series):
    """
    series の dtype を見て、数値列なら float にキャストを試みる。
    失敗したら文字列のまま返す。
    """
    if pd.api.types.is_numeric_dtype(series.dtype):
        try:
            return float(value)
        except Exception:
            return value
    # ここで datetime まで厳密に扱いたい場合は、pd.to_datetime を使った拡張もあり
    return value


def apply_conditions_pandas(df: pd.DataFrame, conds: List[Dict[str, Any]]) -> pd.Series:
    """
    conds を pandas の boolean マスクに変換。
    DataFrame に対して 1 回のマスク適用で済ませる。
    """
    if not conds:
        return pd.Series([True] * len(df), index=df.index)

    mask = pd.Series([True] * len(df), index=df.index)

    for cond in conds:
        col = cond["col"]
        op = cond["op"]
        values = cond["values"]
        if col not in df.columns:
            print(f"  ⚠ 条件列 {col} が存在しないため、この条件を無視します。")
            continue

        s = df[col]
        if op == "between":
            v1_raw, v2_raw = values
            v1 = _try_cast_value_to_dtype(v1_raw, s)
            v2 = _try_cast_value_to_dtype(v2_raw, s)
            mask &= (s >= v1) & (s <= v2)
        elif op == "in":
            casted = [_try_cast_value_to_dtype(v, s) for v in values]
            mask &= s.isin(casted)
        else:
            v_raw = values[0]
            v = _try_cast_value_to_dtype(v_raw, s)
            if op == "==":
                mask &= (s == v)
            elif op == "!=":
                mask &= (s != v)
            elif op == ">":
                mask &= (s > v)
            elif op == ">=":
                mask &= (s >= v)
            elif op == "<":
                mask &= (s < v)
            elif op == "<=":
                mask &= (s <= v)
            else:
                print(f"  ⚠ 未対応の演算子 {op} のため、この条件を無視します。")

    return mask


# =========================================================
# SQLite (GPKG) の WHERE 句生成
# =========================================================

def build_sql_where_and_params(conds: List[Dict[str, Any]]) -> tuple[str, list[Any]]:
    """
    conds を SQL WHERE 句とパラメータリストに変換。
    値は常にパラメータとして渡し、SQL インジェクションを避ける。
    """
    if not conds:
        return "", []

    parts = []
    params: list[Any] = []
    for cond in conds:
        col = cond["col"]
        op = cond["op"]
        values = cond["values"]

        if op == "between":
            parts.append(f'"{col}" BETWEEN ? AND ?')
            params.extend(values)
        elif op == "in":
            if not values:
                continue
            placeholders = ", ".join(["?"] * len(values))
            parts.append(f'"{col}" IN ({placeholders})')
            params.extend(values)
        else:
            if op not in ("==", "!=", ">", ">=", "<", "<="):
                continue
            sql_op = "=" if op == "==" else op
            parts.append(f'"{col}" {sql_op} ?')
            params.append(values[0])

    if not parts:
        return "", []

    where_clause = " WHERE " + " AND ".join(parts)
    return where_clause, params


# =========================================================
# CSV / Parquet 集計（pandas）
# =========================================================

def process_csv_or_parquet(path: str,
                           target_col: str,
                           conds: List[Dict[str, Any]],
                           df: pd.DataFrame) -> List[str]:
    """
    CSV / Parquet を pandas で一度だけ読み込んで集計。
    df はすでに読み込まれている前提。
    戻り値: 出力テキストの各行（ログ用）
    """
    logs: List[str] = []
    logs.append(f"[INFO] 読み込み済み: {path}")
    logs.append(f"  行数: {len(df):,}  列数: {len(df.columns)}")

    if target_col not in df.columns:
        msg = f"[ERROR] 指定されたカラム {target_col} が存在しません。"
        print(msg)
        logs.append(msg)
        return logs

    mask = apply_conditions_pandas(df, conds)
    df_sub = df.loc[mask]
    logs.append(f"[INFO] 条件適用後の行数: {len(df_sub):,}")

    vc = df_sub[target_col].value_counts(dropna=False).sort_index()
    total = int(vc.sum())

    logs.append("\n=== 集計結果 ===")
    logs.append(f"対象カラム: {target_col}")
    logs.append(f"総件数（条件適用後）: {total:,}\n")

    header = f"{'値':>20}  {'件数':>10}  {'割合(%)':>10}"
    logs.append(header)
    logs.append("-" * len(header))

    for val, cnt in vc.items():
        ratio = 100.0 * cnt / total if total > 0 else 0.0
        logs.append(f"{str(val):>20}  {cnt:10d}  {ratio:10.2f}")

    for line in logs:
        print(line)

    return logs


# =========================================================
# GPKG 集計（SQLite）
# =========================================================

def process_gpkg(path: str,
                 target_col: str,
                 conds: List[Dict[str, Any]],
                 layer_name: str | None = None) -> List[str]:
    """
    GPKG を SQLite の GROUP BY で直接集計。
    戻り値: 出力テキストの各行（ログ用）
    """
    logs: List[str] = []
    logs.append(f"[INFO] GPKG を SQLite 経由で集計します: {path}")

    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()

        # レイヤ名（テーブル名）未指定なら gpkg_contents から 1 つ拾う
        if layer_name is None:
            try:
                cur.execute("SELECT table_name FROM gpkg_contents LIMIT 1;")
                row = cur.fetchone()
                if row:
                    layer_name = row[0]
                    logs.append(f"[INFO] 使用レイヤ: {layer_name}")
                else:
                    msg = "[ERROR] gpkg_contents が空です。レイヤが見つかりません。"
                    print(msg)
                    logs.append(msg)
                    return logs
            except Exception as e:
                msg = f"[ERROR] レイヤ名の取得に失敗しました: {e}"
                print(msg)
                logs.append(msg)
                return logs

        # カラム一覧と型
        cur.execute(f'PRAGMA table_info("{layer_name}")')
        cols_info = cur.fetchall()
        col_names = [r[1] for r in cols_info]
        type_map = {r[1]: (r[2] or "") for r in cols_info}

        if target_col not in col_names:
            msg = f"[ERROR] 指定されたカラム {target_col} はレイヤ {layer_name} に存在しません。"
            print(msg)
            logs.append(msg)
            return logs

        # WHERE 句とパラメータ
        where_clause, params = build_sql_where_and_params(conds)

        # 集計クエリ
        sql = f'''
            SELECT "{target_col}", COUNT(*) as cnt
            FROM "{layer_name}"
            {where_clause}
            GROUP BY "{target_col}"
            ORDER BY "{target_col}"
        '''
        logs.append("[DEBUG] 実行クエリ:")
        logs.append("  " + " ".join(sql.split()))
        logs.append(f"[DEBUG] パラメータ: {params}")

        cur.execute(sql, params)
        rows = cur.fetchall()

        # 総件数
        sql_total = f'''
            SELECT COUNT(*) FROM "{layer_name}" {where_clause}
        '''
        cur.execute(sql_total, params)
        total_row = cur.fetchone()
        total = int(total_row[0]) if total_row else 0

    finally:
        conn.close()

    logs.append("\n=== 集計結果 ===")
    logs.append(f"対象カラム: {target_col}")
    logs.append(f"総件数（条件適用後）: {total:,}\n")

    header = f"{'値':>20}  {'件数':>10}  {'割合(%)':>10}"
    logs.append(header)
    logs.append("-" * len(header))

    for val, cnt in rows:
        ratio = 100.0 * cnt / total if total > 0 else 0.0
        logs.append(f"{str(val):>20}  {cnt:10d}  {ratio:10.2f}")

    for line in logs:
        print(line)

    return logs


# =========================================================
# メインフロー
# =========================================================

def main():
    print("\n=== 指定列のカテゴリ別カウントツール（条件付き & 型ガイド付き） ===")

    path = strip_quotes(input("入力ファイルのパス（CSV / Parquet / GPKG）: ").strip())
    if not path:
        print("ファイルパスが指定されていません。終了します。")
        sys.exit(1)
    if not os.path.exists(path):
        print(f"ファイルが見つかりません: {path}")
        sys.exit(1)

    suffix = Path(path).suffix.lower()

    # -----------------------------------------------------
    # CSV / Parquet の場合: ここで一度だけ DataFrame を読み込む
    # -----------------------------------------------------
    df = None
    col_list: List[str] = []
    kind_map: Dict[str, str] = {}

    if suffix == ".csv":
        print(f"[INFO] CSV を読み込み中です（ファイルサイズによっては時間がかかります）: {path}")
        df = pd.read_csv(path)
        col_list = df.columns.tolist()
        for c in col_list:
            kind_map[c] = get_kind_from_dtype(df[c].dtype)

    elif suffix in (".parquet", ".pq"):
        print(f"[INFO] Parquet を読み込み中です: {path}")
        df = pd.read_parquet(path)
        col_list = df.columns.tolist()
        for c in col_list:
            kind_map[c] = get_kind_from_dtype(df[c].dtype)

    # -----------------------------------------------------
    # GPKG の場合: SQLite から列名と型を取得
    # -----------------------------------------------------
    elif suffix == ".gpkg":
        conn = sqlite3.connect(path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT table_name FROM gpkg_contents LIMIT 1;")
            row = cur.fetchone()
            if not row:
                print("[ERROR] gpkg_contents が空です。終了します。")
                return
            layer = row[0]

            cur.execute(f'PRAGMA table_info("{layer}")')
            cols_info = cur.fetchall()
            col_list = [r[1] for r in cols_info]
            type_map = {r[1]: (r[2] or "") for r in cols_info}
            for c in col_list:
                kind_map[c] = get_kind_from_sql_type(type_map[c])

            print(f"[INFO] GPKG レイヤ: {layer}")
        finally:
            conn.close()
    else:
        print(f"[ERROR] 未対応の拡張子です: {suffix}")
        sys.exit(1)

    # 対象カラム（カウントしたい列）を選択
    target_col = choose_column(col_list, title="カウント対象カラムの選択")

    # 型ガイド表示
    if suffix in (".csv", ".parquet", ".pq"):
        print_column_guide_pandas(df, target_col)
    elif suffix == ".gpkg":
        # GPKG の場合はもう一度 conn を開いてサンプルを取得
        conn = sqlite3.connect(path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT table_name FROM gpkg_contents LIMIT 1;")
            row = cur.fetchone()
            layer = row[0] if row else None

            cur.execute(f'PRAGMA table_info("{layer}")')
            cols_info = cur.fetchall()
            type_map = {r[1]: (r[2] or "") for r in cols_info}
            sql_type = type_map.get(target_col, "")
            print_column_guide_sqlite(conn, layer, target_col, sql_type)
        finally:
            conn.close()

    # まず対象列に対する基本パターンを選択
    basic_conds = choose_basic_pattern_for_target(target_col)

    if basic_conds is None:
        # 詳細条件モード（他の列も条件に使う）
        conds = build_conditions_interactive(col_list, kind_map)
    else:
        # 基本パターンから決まった条件をそのまま使う
        conds = basic_conds

    # 本処理
    if suffix in (".csv", ".parquet", ".pq"):
        logs = process_csv_or_parquet(path, target_col, conds, df)
    elif suffix == ".gpkg":
        logs = process_gpkg(path, target_col, conds)
    else:
        print("[ERROR] 想定外の拡張子です。終了します。")
        return

    # ログを txt に保存
    out_base = Path(path)
    out_txt = out_base.parent / f"{out_base.stem}_count_report.txt"
    try:
        with open(out_txt, "w", encoding="utf-8") as f:
            for line in logs:
                f.write(str(line) + "\n")
        print(f"\n[保存] 結果レポートをテキストに保存しました: {out_txt}")
    except Exception as e:
        print(f"[WARN] テキスト出力に失敗しました: {e}")


if __name__ == "__main__":
    main()
