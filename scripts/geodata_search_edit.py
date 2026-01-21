#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeoParquet / GeoPackage / CSV の属性フィールドを条件検索（正規表現）して一括編集するインタラクティブCLI。

要件:
- 入力: .parquet/.geoparquet, .gpkg, .csv
- 検索: 指定フィールドの値に対して正規表現で一致判定
- 値一覧: 指定フィールドのユニーク値をナンバー付きで表示し、番号選択 or 直接正規表現入力
- 編集: 一致行に対して所定の編集（複数パターンを用意）
- 出力: 入力と同形式 or 別形式で保存（出力パスを対話指定）

依存:
- geopandas, pandas, pyarrow
  pip install geopandas pyarrow pandas
"""

from __future__ import annotations

import os
import re
import sys
import math
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    import pyarrow as pa  # noqa: F401
    import pyarrow.parquet as pq  # noqa: F401
except Exception:
    pass


# -----------------------------
# Utilities
# -----------------------------
def strip_quotes(s: str) -> str:
    return s.strip().strip('"').strip("'")


def prompt_path(msg: str) -> Path:
    while True:
        s = input(msg).strip()
        s = strip_quotes(s)
        if not s:
            print("[ERROR] 空文字です。")
            continue
        p = Path(s)
        return p


def infer_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in [".parquet", ".geoparquet"]:
        return "parquet"
    if ext == ".gpkg":
        return "gpkg"
    if ext == ".csv":
        return "csv"
    return "unknown"


def is_geo_df(df: Any) -> bool:
    return gpd is not None and isinstance(df, gpd.GeoDataFrame)


def safe_to_str_series(s: pd.Series) -> pd.Series:
    # NaN を "nan" にしたくないので空文字化
    return s.astype("string").fillna(pd.NA).astype("string")

def summarize_series_kind(s: pd.Series, numeric_threshold: float = 0.95) -> Tuple[str, float, str]:
    """
    列の「実用上の型」を推定して返す。
    returns: (kind_label, numeric_parse_rate, dtype_str)
      kind_label: "numeric" / "string" / "mixed"
    """
    dtype_str = str(s.dtype)
    num_try = pd.to_numeric(s, errors="coerce")
    parse_rate = float(num_try.notna().mean()) if len(s) else 0.0
    if parse_rate >= numeric_threshold:
        kind = "numeric"
    elif parse_rate <= 0.05:
        kind = "string"
    else:
        kind = "mixed"
    return kind, parse_rate, dtype_str

def print_kind_hint(col_name: str, kind: str, parse_rate: float, dtype_str: str) -> None:
    # 人間向け表現（pandas dtype を隠しすぎない程度に併記）
    if kind == "numeric":
        kmsg = "数値"
    elif kind == "string":
        kmsg = "文字列"
    else:
        kmsg = "混在列（文字列/数値が混ざっている可能性）"
    # 後続表示（列一覧など）と視覚的に分離
    print(f"[INFO] 列型推定: {col_name} = {kmsg}  (dtype={dtype_str}, numeric_rate={parse_rate:.3f})")
    print("")

def is_missing_value(v: Any) -> bool:
    # pandas.NA / None / NaN を欠損として扱う
    if v is pd.NA:
        return True
    if v is None:
        return True
    try:
        # float('nan') 判定
        return isinstance(v, float) and math.isnan(v)
    except Exception:
        return False


def print_numbered(items: List[str], counts: Optional[List[int]] = None, limit: int = 50) -> None:
    n = len(items)
    m = min(n, limit)
    for i in range(m):
        if counts is None:
            print(f"{i+1:>3}) {items[i]}")
        else:
            print(f"{i+1:>3}) {items[i]}  (count={counts[i]})")
    if n > limit:
        print(f"... 省略: {n-limit} 件")


def choose_from_list(title: str, options: List[str]) -> int:
    print(title)
    for i, opt in enumerate(options, start=1):
        print(f"{i}) {opt}")
    while True:
        s = input("番号を選択: ").strip()
        if not s.isdigit():
            print("[ERROR] 数字のみ。")
            continue
        k = int(s)
        if 1 <= k <= len(options):
            return k
        print("[ERROR] 範囲外。")


def parse_indices(s: str, n: int) -> List[int]:
    # "1,2,5-7"
    s = s.replace(" ", "")
    if not s:
        return []
    out: List[int] = []
    parts = s.split(",")
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            if not (a.isdigit() and b.isdigit()):
                raise ValueError("range")
            ia, ib = int(a), int(b)
            if ia > ib:
                ia, ib = ib, ia
            for v in range(ia, ib + 1):
                if 1 <= v <= n:
                    out.append(v)
        else:
            if not part.isdigit():
                raise ValueError("index")
            v = int(part)
            if 1 <= v <= n:
                out.append(v)
    # 1-based -> 0-based unique preserve order
    seen = set()
    out0 = []
    for v in out:
        if v not in seen:
            seen.add(v)
            out0.append(v - 1)
    return out0


def ensure_gpd() -> None:
    if gpd is None:
        raise RuntimeError("geopandas が import できません。pip install geopandas を実行してください。")


# -----------------------------
# IO
# -----------------------------
def read_table(path: Path) -> Tuple[pd.DataFrame, str]:
    fmt = infer_format(path)
    if fmt == "parquet":
        # geopandas は GeoParquet を read_parquet で読める（環境依存あり）
        # 失敗したら pandas で読む（この場合 geometry は通常の列になる）
        try:
            ensure_gpd()
            df = gpd.read_parquet(path)
            return df, "parquet"
        except Exception:
            df = pd.read_parquet(path)
            return df, "parquet"
    if fmt == "gpkg":
        ensure_gpd()
        # レイヤ選択
        layers = gpd.list_layers(path)
        # layers: DataFrame(name, geometry_type)
        names = layers["name"].tolist()
        if not names:
            raise RuntimeError("GPKG にレイヤがありません。")
        if len(names) == 1:
            layer = names[0]
        else:
            print("=== レイヤ一覧 ===")
            for i, nm in enumerate(names, start=1):
                gtype = layers.loc[layers["name"] == nm, "geometry_type"].values
                gtype = gtype[0] if len(gtype) else ""
                print(f"{i}) {nm}  ({gtype})")
            while True:
                s = input("使用するレイヤ番号: ").strip()
                if s.isdigit() and 1 <= int(s) <= len(names):
                    layer = names[int(s) - 1]
                    break
                print("[ERROR] 範囲外。")
        df = gpd.read_file(path, layer=layer)
        # layer 名はメタとして持つ
        df.attrs["_gpkg_layer"] = layer
        return df, "gpkg"
    if fmt == "csv":
        df = pd.read_csv(path)
        return df, "csv"
    raise ValueError(f"未対応の拡張子: {path.suffix}")


def write_table(df: pd.DataFrame, out_path: Path, in_fmt: str) -> None:
    out_fmt = infer_format(out_path)
    if out_fmt == "unknown":
        raise ValueError("出力拡張子が未対応です（.parquet/.geoparquet/.gpkg/.csv）。")

    if out_fmt == "csv":
        if is_geo_df(df):
            print("CSV 出力: geometry の扱いを選択")
            k = choose_from_list("選択:", ["geometry を破棄して出力", "geometry を WKT 列（geometry_wkt）として出力"])
            if k == 1:
                tmp = pd.DataFrame(df.drop(columns=[df.geometry.name], errors="ignore"))
            else:
                tmp = pd.DataFrame(df.copy())
                gcol = df.geometry.name
                tmp["geometry_wkt"] = df[gcol].to_wkt()
                tmp = tmp.drop(columns=[gcol], errors="ignore")
            tmp.to_csv(out_path, index=False, encoding="utf-8-sig")
        else:
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return

    if out_fmt == "parquet":
        # GeoParquet を維持できる場合は geopandas で
        if is_geo_df(df):
            df.to_parquet(out_path, index=False)
        else:
            df.to_parquet(out_path, index=False)
        return

    if out_fmt == "gpkg":
        ensure_gpd()
        if not is_geo_df(df):
            raise ValueError("GPKG 出力は GeoDataFrame が必要です（入力がCSV等でgeometry無し）。")
        layer_default = df.attrs.get("_gpkg_layer", "layer")
        # 空入力は「layer指定なし（= None）」にする（= 引数を渡さない）
        s = input(f"出力レイヤ名（空=指定しない / 例: {layer_default}）: ").strip()
        if s:
            df.to_file(out_path, layer=s, driver="GPKG")
        else:
            # layer 引数を省略（環境により既定名が付与される）
            df.to_file(out_path, driver="GPKG")
        return


def _print_cols_page(cols: List[str], page: int, page_size: int) -> Tuple[int, int, int]:
    total = len(cols)
    if total <= 0:
        return 0, 0, 0
    max_page = (total - 1) // page_size
    page = max(0, min(page, max_page))
    start = page * page_size
    end = min(start + page_size, total)
    print(f"=== 列一覧 ({start+1}–{end} / {total}) ===")
    for i in range(start, end):
        print(f"{i+1:>4}) {cols[i]}")
    return page, max_page, total

def select_column_interactive(cols: List[str], prompt_label: str) -> str:
    """
    1000列超を想定し、ページング/正規表現フィルタで列を選択する。
    - ページング: n/p/q/番号(グローバル番号)
    - regex: 列名を正規表現で絞り込み → 結果を番号選択（必要なら再フィルタ/ページング）
    """
    if not cols:
        raise RuntimeError("列が空です。")

    page_size = 50

    while True:
        print(f"=== {prompt_label}: 列選択 ===")
        print("列選択方法:")
        print("  1) ページング表示（n=次, p=前, q=中止, 番号=選択）")
        print("  2) 正規表現で絞り込み（列名に対して re.search）")
        m = input("番号を選択 [1-2]: ").strip()
        if m in ("1", "2"):
            break
        print("[ERROR] 範囲外。")

    if m == "1":
        page = 0
        while True:
            page, max_page, total = _print_cols_page(cols, page, page_size)
            s = input("操作（n/p/q/番号）: ").strip().lower()
            if s == "n":
                page = min(page + 1, max_page)
                continue
            if s == "p":
                page = max(page - 1, 0)
                continue
            if s == "q":
                raise KeyboardInterrupt()
            if s.isdigit():
                k = int(s)
                if 1 <= k <= total:
                    return cols[k - 1]
                print("[ERROR] 範囲外。")
                continue
            print("[ERROR] 入力不正。")

    # m == "2"（regex フィルタ）
    base = cols
    while True:
        pat = input("列名フィルタ用 正規表現（空=中止）: ").strip()
        if not pat:
            raise KeyboardInterrupt()
        try:
            rx = re.compile(pat)
        except re.error as e:
            print(f"[ERROR] 正規表現エラー: {e}")
            continue
        hits = [c for c in base if rx.search(c)]
        if not hits:
            print("[INFO] 一致なし。再入力。")
            continue

        print(f"=== マッチ列: {len(hits)} ===")
        # マッチ数が多い場合はページングで見せる
        if len(hits) <= 200:
            for i, c in enumerate(hits, start=1):
                print(f"{i:>4}) {c}")
        else:
            # hits もページング（グローバル番号で選択）
            page = 0
            while True:
                page, max_page, total = _print_cols_page(hits, page, page_size)
                s = input("操作（n/p/r=再フィルタ/q/番号）: ").strip().lower()
                if s == "n":
                    page = min(page + 1, max_page)
                    continue
                if s == "p":
                    page = max(page - 1, 0)
                    continue
                if s == "r":
                    break  # 再フィルタへ
                if s == "q":
                    raise KeyboardInterrupt()
                if s.isdigit():
                    k = int(s)
                    if 1 <= k <= total:
                        return hits[k - 1]
                    print("[ERROR] 範囲外。")
                    continue
                print("[ERROR] 入力不正。")
            continue  # 再フィルタ

        # hits が少ない場合の単純選択
        while True:
            s = input("番号で選択（r=再フィルタ, q=中止）: ").strip().lower()
            if s == "r":
                break
            if s == "q":
                raise KeyboardInterrupt()
            if s.isdigit() and 1 <= int(s) <= len(hits):
                return hits[int(s) - 1]
            print("[ERROR] 範囲外。")


# -----------------------------
# Main edit flow
# -----------------------------
def main() -> int:
    print("=== 属性一括編集（GeoParquet / GeoPackage / CSV）===")
    in_path = prompt_path("入力ファイルパス（.parquet/.geoparquet/.gpkg/.csv）: ")
    if not in_path.exists():
        print(f"[ERROR] 入力が存在しません: {in_path}")
        return 1

    df, in_fmt = read_table(in_path)
    print(f"[INFO] 読み込み完了: rows={len(df):,}, cols={len(df.columns)}")

    # フィールド選択（検索対象）
    cols = list(map(str, df.columns.tolist()))
    if not cols:
        print("[ERROR] 列がありません。")
        return 1

    try:
        filter_col = select_column_interactive(cols, prompt_label="検索対象列")
    except KeyboardInterrupt:
        print("[INFO] 中断。")
        return 0

    # ユニーク値一覧
    raw_col = pd.Series(df[filter_col])
    ser = safe_to_str_series(raw_col)

    # 列型推定（人間向け）
    f_kind, f_rate, f_dtype = summarize_series_kind(raw_col, numeric_threshold=0.95)
    print_kind_hint(filter_col, f_kind, f_rate, f_dtype)

    # 表示は上位（頻度順）に寄せる
    # 欠損も含めるため raw_col で集計（表示は <NA> に統一）
    vc_raw = raw_col.value_counts(dropna=False)
    raw_values = vc_raw.index.tolist()  # 実値（pd.NA / NaN / None を含み得る）
    values = pd.Index(raw_values).astype("string").fillna("<NA>").tolist()  # 表示用
    counts = vc_raw.values.tolist()

    print(f"=== ユニーク値一覧: {filter_col}（上位50表示）===")
    print_numbered(values, counts=counts, limit=50)

    print("選択方法:")
    print("  - 番号指定: 例 1,2,5-7（選択値のいずれかに“完全一致”で検索）")
    print("  - 0 を入力: 正規表現を直接入力して検索（部分一致）")

    pattern: str
    want_na = False
    while True:
        s = input("番号（または 0）: ").strip()
        if s == "0":
            pattern = input("正規表現（Python re、部分一致）: ").strip()
            if not pattern:
                print("[ERROR] 空です。")
                continue
            try:
                re.compile(pattern)
            except re.error as e:
                print(f"[ERROR] 正規表現エラー: {e}")
                continue
            mode = "regex"
            break
        try:
            idxs = parse_indices(s, n=len(values))
        except ValueError:
            print("[ERROR] 番号形式が不正です。")
            continue
        if not idxs:
            print("[ERROR] 選択が空です。")
            continue
        # 完全一致: ^(v1|v2|... )$
        # ただし <NA> は実体が欠損なので isna() で別扱い
        selected_display = [values[i] for i in idxs]
        selected_raw = [raw_values[i] for i in idxs]
        want_na = any(is_missing_value(v) for v in selected_raw)

        # 文字列一致対象（<NA> は除外）
        selected_vals = [v for v in selected_display if v != "<NA>"]
        if selected_vals:
            escaped = [re.escape(v) for v in selected_vals]
            pattern = r"^(?:" + "|".join(escaped) + r")$"
        else:
            # <NA> のみ選択時：文字列マッチ側は全Falseにして isna() 側で拾う
            pattern = r"(?!)"
        mode = "exact"
        break

    # マッチ行
    #  - 文字列一致（regex/exact）は ser（string化済）で判定
    #  - 欠損一致は raw_col.isna() で判定（<NA> 選択時のみ）
    mask = ser.str.contains(pattern, regex=True, na=False)
    if mode == "exact" and want_na:
        mask = mask | raw_col.isna()
    n_match = int(mask.sum())
    print(f"[INFO] 一致行数: {n_match:,} / {len(df):,}  (mode={mode})")
    if n_match == 0:
        print("[INFO] 一致なし。終了。")
        return 0

    # 編集内容選択
    # ここでは edit_col 未選択なので、操作メニューは「後で列型により向き不向きがある」ことだけ示す
    ops = [
        "一致行の edit_col を定数で上書き（文字/数値）",
        "一致行の edit_col を正規表現置換（文字列向け）",
        "一致行の edit_col に数値加算（数値向け）",
        "一致行の edit_col に数値乗算（数値向け）",
        "一致行（地物行）を削除",
    ]
    op = choose_from_list("=== 編集操作 ===", ops)

    df2 = df.copy()

    # 出力前 CRS/EPSG 指定（GeoDataFrame の場合のみ）
    # - EPSGを変更する場合は「座標変換(to_crs)」か「ラベル上書き(set_crs)」を必ず選択させる
    if is_geo_df(df2):
        cur = df2.crs.to_string() if df2.crs is not None else "None"
        s = input(f"出力EPSG指定（空=維持: {cur} / 例: 6674 or EPSG:6674）: ").strip()
        if s:
            # 入力を crs として解釈
            if s.isdigit():
                target_crs: Any = int(s)
            else:
                target_crs = s

            print("CRSの扱いを選択:")
            print("  1) 座標変換する（to_crs）")
            print("  2) CRSラベルのみ上書き（set_crs, allow_override=True）")
            while True:
                m = input("番号を選択 [1-2]: ").strip()
                if m in ("1", "2"):
                    break
                print("[ERROR] 範囲外。")

            if m == "1":
                # to_crs は元CRSが必須
                if df2.crs is None:
                    print("[ERROR] 元CRSが未定義のため座標変換できません。")
                    print("        先に入力データのCRSを正しく設定するか、2) ラベル上書きを選択してください。")
                    return 1
                # geopandas は epsg=int or crs str のどちらも受ける
                df2 = df2.to_crs(epsg=target_crs) if isinstance(target_crs, int) else df2.to_crs(target_crs)
            else:
                df2 = df2.set_crs(epsg=target_crs, allow_override=True) if isinstance(target_crs, int) else df2.set_crs(target_crs, allow_override=True)

    if op == 5:
        yn = input(f"一致行 {int(mask.sum()):,} 件を削除します（y/N）: ").strip().lower()
        if yn != "y":
            print("[INFO] 中止。")
            return 0
        df2 = df2.loc[~mask].copy()
        # geometry 列がある場合も維持される
        # 出力へ
        out_path = prompt_path("出力ファイルパス（.parquet/.geoparquet/.gpkg/.csv）: ")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_table(df2, out_path, in_fmt=in_fmt)
        print(f"[INFO] 保存完了: {out_path}")
        return 0

    # ここからは「列を書き換える」編集
    # 編集対象列
    try:
        edit_col = select_column_interactive(cols, prompt_label="編集対象列（書き換え先）")
    except KeyboardInterrupt:
        print("[INFO] 中断。")
        return 0

    # edit_col の列型推定と、操作との対応表示
    e_raw = pd.Series(df2[edit_col])
    e_kind, e_rate, e_dtype = summarize_series_kind(e_raw, numeric_threshold=0.95)
    print_kind_hint(edit_col, e_kind, e_rate, e_dtype)
    if e_kind == "numeric":
        print("[INFO] 推奨操作: 1(上書き), 3(加算), 4(乗算)  / 注意: 2(置換)は文字列化されます")
    elif e_kind == "string":
        print("[INFO] 推奨操作: 1(上書き), 2(置換)  / 注意: 3,4 は数値化できない行が NaN になり得ます")
    else:
        print("[INFO] 注意: 混在列のため 2/3/4 の挙動に注意（文字列化/数値化で欠損が発生し得ます）")

    if op == 1:
        raw = input("上書きする値（そのまま文字列。数値にしたい場合は例: 12.3）: ").strip()
        # 数値にできるなら数値、できなければ文字列
        val: Any = raw
        if raw == "":
            val = ""
        else:
            try:
                if "." in raw:
                    val = float(raw)
                else:
                    val = int(raw)
            except Exception:
                val = raw
        df2.loc[mask, edit_col] = val

    elif op == 2:
        repl_pat = input("置換対象の正規表現: ").strip()
        if not repl_pat:
            print("[ERROR] 空です。")
            return 1
        try:
            re.compile(repl_pat)
        except re.error as e:
            print(f"[ERROR] 正規表現エラー: {e}")
            return 1
        repl_to = input("置換後文字列: ")
        # 文字列化して置換
        s_edit = safe_to_str_series(pd.Series(df2[edit_col]))
        df2.loc[mask, edit_col] = s_edit.loc[mask].str.replace(repl_pat, repl_to, regex=True)

    elif op == 3:
        if e_kind == "string":
            print("[WARN] edit_col が文字列列推定です。数値化できない行は NaN になります。")
        delta_s = input("加算値（例: 1.5）: ").strip()
        try:
            delta = float(delta_s)
        except Exception:
            print("[ERROR] 数値に変換できません。")
            return 1
        # 数値化して加算
        num = pd.to_numeric(df2[edit_col], errors="coerce")
        num.loc[mask] = num.loc[mask] + delta
        df2[edit_col] = num

    elif op == 4:
        if e_kind == "string":
            print("[WARN] edit_col が文字列列推定です。数値化できない行は NaN になります。")
        mul_s = input("乗算係数（例: 0.1）: ").strip()
        try:
            mul = float(mul_s)
        except Exception:
            print("[ERROR] 数値に変換できません。")
            return 1
        num = pd.to_numeric(df2[edit_col], errors="coerce")
        num.loc[mask] = num.loc[mask] * mul
        df2[edit_col] = num

    else:
        print("[ERROR] 未対応。")
        return 1

    # 出力
    out_path = prompt_path("出力ファイルパス（.parquet/.geoparquet/.gpkg/.csv）: ")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_table(df2, out_path, in_fmt=in_fmt)
    print(f"[INFO] 保存完了: {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[INFO] 中断。")
        raise
