#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPKG / Parquet / CSV 相互変換ツール（対話式）

機能概要
--------
- GPKG → Parquet / CSV
- Parquet → GPKG / CSV
- CSV → GPKG / Parquet

特徴
----
- ジオメトリ（geometry）を WKB HEX or WKT or X/Y から復元・変換可能
- EPSG コードを列として保存 / 復元
- GPKG 複数レイヤにも対応（レイヤ選択プロンプト）
- 入出力ファイルパスは対話式入力（ドラッグ & ドロップ可）
- 変換前に shape / columns / 先頭数行をプレビュー表示

依存ライブラリ
--------------
- pandas
- geopandas
- shapely
- fiona

使い方（例）
------------
$ python convert_gpkg_parquet_csv_interactive.py

メニューに従って番号を選択してください。
"""

import os
import sys
import textwrap
from typing import Optional, List, Tuple

import pandas as pd
import geopandas as gpd
from shapely import wkb, wkt

try:
    import fiona
except ImportError:
    fiona = None


# ==========================
# 共通ユーティリティ
# ==========================

def ask_path(prompt: str) -> str:
    """パス入力補助（ドラッグ & ドロップ → 余分なクォート除去）"""
    path = input(prompt).strip()
    # ドラッグ & ドロップ時の二重クォートを削除
    if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
        path = path[1:-1]
    return os.path.expanduser(path)


def guess_output_path(input_path: str, new_ext: str) -> str:
    """拡張子だけ差し替えた出力パスを返す"""
    base, _ = os.path.splitext(input_path)
    return base + new_ext


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60 + "\n")


def preview_df(df: pd.DataFrame, max_rows: int = 5) -> None:
    """DataFrame / GeoDataFrame の簡易プレビュー"""
    print(f"[PREVIEW] shape={df.shape[0]} rows x {df.shape[1]} cols")
    print(f"[PREVIEW] columns: {list(df.columns)}")
    print(f"[PREVIEW] head({max_rows}):")
    print(df.head(max_rows))
    print("-" * 60)


def ask_yes_no(msg: str, default_yes: bool = True) -> bool:
    """Y/n プロンプト"""
    default = "Y/n" if default_yes else "y/N"
    while True:
        ans = input(f"{msg} [{default}]: ").strip().lower()
        if ans == "" and default_yes:
            return True
        if ans == "" and not default_yes:
            return False
        if ans in ("y", "yes", "はい"):
            return True
        if ans in ("n", "no", "いいえ"):
            return False
        print("[WARN] y または n で答えてください。")


# ==========================
# GPKG 関連
# ==========================

def list_gpkg_layers(path: str) -> List[str]:
    """GPKG 内のレイヤ一覧を取得"""
    if fiona is None:
        return []

    try:
        return list(fiona.listlayers(path))
    except Exception as e:
        print(f"[WARN] レイヤ一覧取得に失敗しました: {e}")
        return []


def select_gpkg_layer(path: str) -> Optional[str]:
    """GPKG のレイヤをユーザーに選択させる"""
    layers = list_gpkg_layers(path)
    if not layers:
        # fiona なし or 取得失敗 → None（デフォルトレイヤ扱い）
        return None
    if len(layers) == 1:
        print(f"[INFO] 単一レイヤのため自動選択: {layers[0]}")
        return layers[0]

    print("[INFO] GPKG には複数レイヤがあります。番号を選んでください。")
    for i, name in enumerate(layers, start=1):
        print(f"  {i}) {name}")

    while True:
        ans = input(f"レイヤ番号 [1-{len(layers)}]（空=1）: ").strip()
        if ans == "":
            return layers[0]
        if ans.isdigit():
            idx = int(ans)
            if 1 <= idx <= len(layers):
                return layers[idx - 1]
        print("[WARN] 正しい番号を入力してください。")


def read_gpkg(path: str) -> gpd.GeoDataFrame:
    """GPKG を読み込む（レイヤ選択つき）"""
    layer = select_gpkg_layer(path)
    if layer:
        print(f"[INFO] レイヤ '{layer}' を読み込みます...")
        gdf = gpd.read_file(path, layer=layer)
    else:
        print(f"[INFO] レイヤ指定なしで読み込みます...")
        gdf = gpd.read_file(path)
    preview_df(gdf)
    return gdf


# ==========================
# ジオメトリ変換ヘルパー
# ==========================

def export_geometry_from_gdf(
    gdf: gpd.GeoDataFrame
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    GeoDataFrame から geometry をエンコードして DataFrame を返す。
    戻り値: (df, geom_col_name または None)
    """
    if gdf.geometry is None or gdf.geometry.name not in gdf.columns:
        print("[INFO] ジオメトリ列が見つかりませんでした。通常の DataFrame として扱います。")
        df = pd.DataFrame(gdf.drop(columns=gdf.geometry.name, errors="ignore"))
        return df, None

    geom_col = gdf.geometry.name
    print(f"[INFO] ジオメトリ列: '{geom_col}'")

    print(textwrap.dedent("""
        [ジオメトリ出力形式を選択してください]
          1) WKB HEX 文字列（再インポート向け推奨）
          2) WKT 文字列
          3) X/Y 列（ポイントのみ想定）
    """).strip())

    while True:
        mode = input("番号を選択 [1-3]（空=1）: ").strip()
        if mode == "":
            mode = "1"
        if mode not in ("1", "2", "3"):
            print("[WARN] 1〜3 の番号を入力してください。")
            continue
        break

    df = pd.DataFrame(gdf.drop(columns=geom_col))

    if mode == "1":
        out_col = "geometry_wkb"
        print(f"[INFO] WKB HEX として '{out_col}' 列に書き出します。")
        df[out_col] = gdf.geometry.apply(
            lambda g: wkb.dumps(g, hex=True) if g is not None else None
        )
    elif mode == "2":
        out_col = "geometry_wkt"
        print(f"[INFO] WKT として '{out_col}' 列に書き出します。")
        df[out_col] = gdf.geometry.apply(
            lambda g: g.wkt if g is not None else None
        )
    else:
        out_col = None
        print("[INFO] X/Y 列を追加します（Point のみ想定）。")
        try:
            df["x"] = gdf.geometry.x
            df["y"] = gdf.geometry.y
        except Exception as e:
            print(f"[WARN] X/Y 取得に失敗しました: {e}")
            print("[WARN] geometry を破棄して通常の DataFrame として扱います。")
        # out_col は None のまま返す

    # EPSG 列を付与するか
    epsg = None
    if gdf.crs is not None:
        try:
            epsg = gdf.crs.to_epsg()
        except Exception:
            epsg = None

    if epsg is not None:
        if ask_yes_no(f"[INFO] CRS の EPSG={epsg} を 'epsg' 列として保存しますか？", default_yes=True):
            df["epsg"] = epsg

    preview_df(df)
    return df, out_col


def restore_geometry_to_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    DataFrame から GeoDataFrame にジオメトリを復元。
    - WKB HEX 列
    - WKT 列
    - X/Y 列
    - ジオメトリなし（属性テーブルとして GPKG に保存）

    戻り値: GeoDataFrame
    """
    print(textwrap.dedent("""
        [ジオメトリ復元方法を選択してください]
          1) WKB HEX 列から復元
          2) WKT 列から復元
          3) X/Y 列から Point を生成
          4) ジオメトリなし（属性テーブルとして保存）
    """).strip())

    while True:
        mode = input("番号を選択 [1-4]（空=4）: ").strip()
        if mode == "":
            mode = "4"
        if mode not in ("1", "2", "3", "4"):
            print("[WARN] 1〜4 の番号を入力してください。")
            continue
        break

    if mode == "4":
        print("[INFO] ジオメトリなしで GeoDataFrame を作成します。")
        gdf = gpd.GeoDataFrame(df.copy(), geometry=None)
        # EPSG 列があれば、最頻値か先頭値を CRS として採用
        epsg = extract_epsg_from_df(df)
        if epsg is not None:
            gdf = gdf.set_crs(epsg, allow_override=True)
        preview_df(gdf)
        return gdf

    # モード 1〜3 → 何らかのジオメトリを作る
    if mode == "1":
        candidates = [c for c in df.columns if "wkb" in c.lower() or "geom" in c.lower()]
        geom_col = ask_column_name(df, "WKB HEX 列名", candidates)
        print(f"[INFO] '{geom_col}' を WKB HEX として解釈します。")

        def _to_geom(v):
            if pd.isna(v):
                return None
            try:
                return wkb.loads(v, hex=True)
            except Exception:
                return None

        geom = df[geom_col].apply(_to_geom)

    elif mode == "2":
        candidates = [c for c in df.columns if "wkt" in c.lower() or "geom" in c.lower()]
        geom_col = ask_column_name(df, "WKT 列名", candidates)
        print(f"[INFO] '{geom_col}' を WKT として解釈します。")

        def _to_geom(v):
            if pd.isna(v):
                return None
            try:
                return wkt.loads(v)
            except Exception:
                return None

        geom = df[geom_col].apply(_to_geom)

    else:
        # X/Y → Point
        x_candidates = [c for c in df.columns if c.lower() in ("x", "lon", "longitude", "easting")]
        y_candidates = [c for c in df.columns if c.lower() in ("y", "lat", "latitude", "northing")]

        x_col = ask_column_name(df, "X 座標列名", x_candidates)
        y_col = ask_column_name(df, "Y 座標列名", y_candidates)

        print(f"[INFO] X='{x_col}', Y='{y_col}' から Point を生成します。")
        geom = gpd.points_from_xy(df[x_col], df[y_col])

    # GeoDataFrame 化
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geom)

    epsg = extract_epsg_from_df(df)
    if epsg is not None:
        print(f"[INFO] EPSG={epsg} を CRS として設定します。")
        gdf = gdf.set_crs(epsg, allow_override=True)
    else:
        # ユーザーに聞く
        epsg = ask_epsg_manual()
        if epsg is not None:
            print(f"[INFO] EPSG={epsg} を CRS として設定します。")
            gdf = gdf.set_crs(epsg, allow_override=True)

    preview_df(gdf)
    return gdf


def ask_column_name(df: pd.DataFrame, label: str, candidates: List[str]) -> str:
    """列名をユーザーに選ばせるヘルパー"""
    print(f"[INFO] {label} を指定してください。")
    if candidates:
        print(f"  候補: {candidates}")
    while True:
        col = input(f"{label}: ").strip()
        if col in df.columns:
            return col
        if col == "" and len(candidates) == 1:
            print(f"[INFO] 空入力のため候補 '{candidates[0]}' を採用します。")
            return candidates[0]
        print(f"[WARN] 列 '{col}' は存在しません。存在する列から選んでください。")


def extract_epsg_from_df(df: pd.DataFrame) -> Optional[int]:
    """epsg / srid 列から EPSG を推定"""
    candidates = [c for c in df.columns if c.lower() in ("epsg", "srid", "code")]
    if not candidates:
        return None

    col = candidates[0]
    vals = df[col].dropna().unique()
    if len(vals) == 0:
        return None

    try:
        epsg = int(vals[0])
        if len(vals) > 1:
            print(f"[WARN] '{col}' 列に複数の値がありますが、最初の値 {epsg} を採用します。")
        return epsg
    except Exception:
        return None


def ask_epsg_manual() -> Optional[int]:
    """ユーザーに EPSG を入力させる（空=未設定）"""
    while True:
        ans = input("CRS の EPSG コードを入力してください（空=設定しない）: ").strip()
        if ans == "":
            return None
        if ans.isdigit():
            return int(ans)
        print("[WARN] 数値で入力してください。")


# ==========================
# 各種フォーマット読み書き
# ==========================

def read_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    preview_df(df)
    return df


def read_csv_smart(path: str) -> pd.DataFrame:
    """CSV をエンコーディング自動トライで読み込み"""
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[INFO] encoding='{enc}' で読み込み成功。")
            preview_df(df)
            return df
        except Exception:
            continue
    print("[ERROR] CSV の読み込みに失敗しました（encoding を手動で指定してください）。")
    raise RuntimeError("CSV load failed")


def write_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[OK] Parquet を出力しました: {path}")


def write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] CSV を出力しました: {path}")


def write_gpkg(gdf: gpd.GeoDataFrame, path: str, layer: Optional[str] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if layer is None:
        layer = os.path.splitext(os.path.basename(path))[0]

    # 既存ファイルに同名レイヤがある場合の扱い
    if os.path.exists(path) and fiona is not None:
        try:
            layers = list(fiona.listlayers(path))
            if layer in layers:
                if not ask_yes_no(f"[WARN] GPKG '{path}' にレイヤ '{layer}' が既に存在します。上書きしますか？", default_yes=False):
                    print("[INFO] ユーザーが上書きをキャンセルしました。")
                    return
        except Exception:
            pass

    gdf.to_file(path, layer=layer, driver="GPKG")
    print(f"[OK] GPKG を出力しました: {path} (layer='{layer}')")


# ==========================
# 変換ルーチン
# ==========================

def convert_gpkg_to_parquet():
    print_header("GPKG → Parquet 変換")
    in_path = ask_path("入力 GPKG ファイルパス: ")
    if not os.path.isfile(in_path):
        print("[ERROR] ファイルが見つかりません。")
        return

    gdf = read_gpkg(in_path)
    df, _ = export_geometry_from_gdf(gdf)

    default_out = guess_output_path(in_path, ".parquet")
    out_path = ask_path(f"出力 Parquet パス（空={default_out}）: ")
    if out_path == "":
        out_path = default_out

    write_parquet(df, out_path)


def convert_gpkg_to_csv():
    print_header("GPKG → CSV 変換")
    in_path = ask_path("入力 GPKG ファイルパス: ")
    if not os.path.isfile(in_path):
        print("[ERROR] ファイルが見つかりません。")
        return

    gdf = read_gpkg(in_path)
    df, _ = export_geometry_from_gdf(gdf)

    default_out = guess_output_path(in_path, ".csv")
    out_path = ask_path(f"出力 CSV パス（空={default_out}）: ")
    if out_path == "":
        out_path = default_out

    write_csv(df, out_path)


def convert_parquet_to_gpkg():
    print_header("Parquet → GPKG 変換")
    in_path = ask_path("入力 Parquet ファイルパス: ")
    if not os.path.isfile(in_path):
        print("[ERROR] ファイルが見つかりません。")
        return

    df = read_parquet(in_path)
    gdf = restore_geometry_to_gdf(df)

    default_out = guess_output_path(in_path, ".gpkg")
    out_path = ask_path(f"出力 GPKG パス（空={default_out}）: ")
    if out_path == "":
        out_path = default_out

    layer = input("出力レイヤ名（空=ファイル名から自動）: ").strip() or None
    write_gpkg(gdf, out_path, layer)


def convert_parquet_to_csv():
    print_header("Parquet → CSV 変換")
    in_path = ask_path("入力 Parquet ファイルパス: ")
    if not os.path.isfile(in_path):
        print("[ERROR] ファイルが見つかりません。")
        return

    df = read_parquet(in_path)
    default_out = guess_output_path(in_path, ".csv")
    out_path = ask_path(f"出力 CSV パス（空={default_out}）: ")
    if out_path == "":
        out_path = default_out

    write_csv(df, out_path)


def convert_csv_to_gpkg():
    print_header("CSV → GPKG 変換")
    in_path = ask_path("入力 CSV ファイルパス: ")
    if not os.path.isfile(in_path):
        print("[ERROR] ファイルが見つかりません。")
        return

    df = read_csv_smart(in_path)
    gdf = restore_geometry_to_gdf(df)

    default_out = guess_output_path(in_path, ".gpkg")
    out_path = ask_path(f"出力 GPKG パス（空={default_out}）: ")
    if out_path == "":
        out_path = default_out

    layer = input("出力レイヤ名（空=ファイル名から自動）: ").strip() or None
    write_gpkg(gdf, out_path, layer)


def convert_csv_to_parquet():
    print_header("CSV → Parquet 変換")
    in_path = ask_path("入力 CSV ファイルパス: ")
    if not os.path.isfile(in_path):
        print("[ERROR] ファイルが見つかりません。")
        return

    df = read_csv_smart(in_path)

    default_out = guess_output_path(in_path, ".parquet")
    out_path = ask_path(f"出力 Parquet パス（空={default_out}）: ")
    if out_path == "":
        out_path = default_out

    write_parquet(df, out_path)


# ==========================
# メインメニュー
# ==========================

def main():
    print_header("GPKG / Parquet / CSV 相互変換ツール（対話式）")

    menu = textwrap.dedent("""
        変換モードを選択してください:

          1) GPKG   → Parquet
          2) GPKG   → CSV
          3) Parquet → GPKG
          4) Parquet → CSV
          5) CSV    → GPKG
          6) CSV    → Parquet
          0) 終了
    """).strip()

    while True:
        print(menu)
        choice = input("番号を選んでください [0-6]: ").strip()

        if choice == "0":
            print("[INFO] 終了します。")
            break
        elif choice == "1":
            convert_gpkg_to_parquet()
        elif choice == "2":
            convert_gpkg_to_csv()
        elif choice == "3":
            convert_parquet_to_gpkg()
        elif choice == "4":
            convert_parquet_to_csv()
        elif choice == "5":
            convert_csv_to_gpkg()
        elif choice == "6":
            convert_csv_to_parquet()
        else:
            print("[WARN] 0〜6 の番号を入力してください。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C で中断されました。")
        sys.exit(1)
