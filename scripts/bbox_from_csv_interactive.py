#!/usr/bin/env python3
"""
name,Xmin,Ymin,Xmax,Ymax のテキストデータから BBOX ポリゴンを作成し、
GPKG / GeoJSON / Shapefile などに出力する対話式スクリプト。

例）
name,Xmin,Ymin,Xmax,Ymax
A,20400,-114850,21400,-114750
B,22150,-116000,23150,-115000
"""

from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import box


def ask_path(prompt: str) -> Path:
    """ドラッグ&ドロップも想定したパス入力."""
    while True:
        s = input(prompt).strip()
        if not s:
            print("[ERROR] 空です。もう一度入力してください。")
            continue

        # ドラッグ&ドロップ時のクォート除去
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1]

        p = Path(s).expanduser()
        if not p.exists():
            print(f"[ERROR] パスが存在しません: {p}")
            continue
        return p


def ask_epsg(default: int | None = None) -> str:
    """EPSGコードを聞く."""
    while True:
        if default is None:
            s = input("出力する BBOX の座標系 EPSG コードを入力してください（例: 6673, 6674）: ").strip()
        else:
            s = input(f"出力する BBOX の座標系 EPSG コードを入力してください（空Enterで EPSG:{default}）: ").strip()
            if not s:
                return f"EPSG:{default}"

        if not s.isdigit():
            print("[ERROR] 数字で入力してください（例: 6673）")
            continue
        return f"EPSG:{int(s)}"


def ask_output_format() -> tuple[str, str]:
    """
    出力形式を選択させる。
    戻り値: (driver, default_ext)
    """
    print("\n出力形式を選択してください:")
    print("  1) GPKG (GeoPackage)")
    print("  2) GeoJSON")
    print("  3) Shapefile")
    while True:
        s = input("番号を入力 [1-3]（空Enterで 1 = GPKG）: ").strip()
        if not s:
            s = "1"
        if s == "1":
            return "GPKG", ".gpkg"
        elif s == "2":
            return "GeoJSON", ".geojson"
        elif s == "3":
            return "ESRI Shapefile", ".shp"
        else:
            print("[ERROR] 1〜3 を入力してください。")


def main():
    print("=== BBOX CSV → ポリゴン (GPKG / GeoJSON 他) 変換スクリプト ===\n")

    # 1) 入力CSV
    csv_path = ask_path('BBOX テキスト（CSV）のパスを入力してください（ドラッグ&ドロップ可）: ')
    print(f"[INFO] 入力CSV: {csv_path}")

    # 2) 座標系 EPSG
    crs = ask_epsg(default=6673)
    print(f"[INFO] 出力座標系: {crs}")

    # 3) 出力形式
    driver, ext = ask_output_format()
    print(f"[INFO] 出力形式: {driver}")

    # 4) 出力パス
    default_out = csv_path.with_suffix("")  # 拡張子なし
    default_out = default_out.with_name(default_out.name + "_bbox")  # xxx_bbox
    default_out = default_out.with_suffix(ext)
    out_str = input(f"\n出力ファイルパスを入力してください\n"
                    f"(空Enterで既定: {default_out}): ").strip()
    if not out_str:
        out_path = default_out
    else:
        if (out_str.startswith('"') and out_str.endswith('"')) or (
            out_str.startswith("'") and out_str.endswith("'")
        ):
            out_str = out_str[1:-1]
        out_path = Path(out_str).expanduser()

    print(f"[INFO] 出力ファイル: {out_path}\n")

    # 5) CSV 読み込み
    #   日本語ヘッダ/ BOM を考慮して encoding だけ少し丁寧に
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as e:
        print(f"[ERROR] CSV 読み込みに失敗しました: {e}")
        return

    required_cols = ["name", "Xmin", "Ymin", "Xmax", "Ymax"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[ERROR] 必須列 {col} が見つかりません。CSV 例:\n"
                  "  name,Xmin,Ymin,Xmax,Ymax\n  A,20400,-114850,21400,-114750")
            return

    # 数値列に変換（万一の文字混入に備えて）
    for col in ["Xmin", "Ymin", "Xmax", "Ymax"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[["Xmin", "Ymin", "Xmax", "Ymax"]].isnull().any().any():
        print("[WARN] 一部の行で座標が数値に変換できませんでした（NaN を含みます）。その行はスキップされます。")

    # 6) BBOX ポリゴン作成
    geometries = []
    valid_rows = []
    for idx, row in df.iterrows():
        xmin, ymin, xmax, ymax = row["Xmin"], row["Ymin"], row["Xmax"], row["Ymax"]
        if pd.isna(xmin) or pd.isna(ymin) or pd.isna(xmax) or pd.isna(ymax):
            continue
        # shapely.geometry.box(minx, miny, maxx, maxy)
        geom = box(xmin, ymin, xmax, ymax)
        geometries.append(geom)
        valid_rows.append(row)

    if not geometries:
        print("[ERROR] 有効な BBOX が 1 つも生成できませんでした。")
        return

    gdf = gpd.GeoDataFrame(valid_rows, geometry=geometries, crs=crs)

    print(f"[INFO] 作成された BBOX 数: {len(gdf)}")

    # 7) プレビュー（最初の数件）
    print("\n--- 先頭 5 件プレビュー ---")
    print(gdf[["name", "Xmin", "Ymin", "Xmax", "Ymax"]].head())

    # 8) 書き出し確認
    ans = input("\nこの内容で書き出してよろしいですか？ [y/N]: ").strip().lower()
    if ans != "y":
        print("キャンセルしました。")
        return

    # Shapefile の場合、出力先ディレクトリがなければ作成（.shp 以外も一応）
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        gdf.to_file(out_path, driver=driver)
    except Exception as e:
        print(f"[ERROR] 書き出しに失敗しました: {e}")
        return

    print("\n[OK] 書き出し完了:", out_path)


if __name__ == "__main__":
    main()
