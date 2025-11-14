#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rf_geomorph_interactive.py（学習用データ作成＋RandomForest分類・完全版）

【このスクリプトでできること】
  1) 学習用データ作成モード
     - 指定ディレクトリ内のラスター（GeoTIFF 等）とポリゴンGPKGをまとめて読み込み、
       - 範囲: 手入力の BBOX（xmin, ymin, xmax, ymax）
         または ポリゴンの外接矩形
       - 解像度: ユーザー指定グリッド間隔
       に基づいてグリッドポイントを生成し、
       各ポイントでラスター値・ポリゴン属性をサンプリングして
       学習用テーブル（CSV / Parquet / GPKG）を作成する。

  2) 学習モード（train）
     - CSV / Parquet / GPKG 形式の学習用テーブルを読み込み、
       目的変数（ラベル）列・特徴量列を対話的に選択して
       RandomForestClassifier を学習する。
     - SimpleImputer(median) → RandomForest の Pipeline 構成。
     - 実験タグ run_id を自動採番し、以下を出力:
         * 学習済みモデル（joblib）
         * メタ情報 JSON（使用特徴量・ターゲット列・CRS 等）
         * 特徴量重要度 CSV / PNG
         * 混同行列（正規化/非正規化）PNG
         * クラス別指標（precision / recall / F1）のテキスト

  3) 予測モード（predict）
     - 保存済みモデル＋メタ情報を読み込み、
       新しいテーブル（CSV / Parquet / GPKG）に対して予測を実行する。
     - 学習時と同じ特徴量名を自動照合（簡易エイリアス補完あり）し、
       y_pred / proba_max などを付与したテーブルを出力する。
     - 正解ラベル列が含まれていれば、自動で混同行列・指標を再計算する。

【典型的なワークフロー】
  1. 学習用データを作る
       python rf_geomorph_interactive.py
         → 「1) 学習用データ作成」を選択
         → ラスター/ポリゴンの入ったディレクトリを指定
         → 範囲・解像度・付与するポリゴン属性を指定
         → train_data.gpkg / .csv / .parquet などを出力

  2. モデルを学習する
         python rf_geomorph_interactive.py
           → 「2) 学習（train）」を選択
           → 1 で作成したテーブルを指定
           → 目的変数（例: LandClass）と特徴量列を対話的に選択
           → rf_model_xxxxxx.joblib / rf_meta_xxxxxx.json などを出力

  3. 別エリアや別DEMから作成した特徴量テーブルに予測をかける
         python rf_geomorph_interactive.py
           → 「3) 予測（predict）」を選択
           → 2 で保存したモデル＋メタ情報を指定
           → 予測対象テーブルを指定
           → y_pred 付きテーブル（CSV / Parquet / GPKG）を出力

【定義・前提】
  - 特徴量テーブル:
      * 各行 = 1 地点（通常はグリッドの中心点）
      * x, y 列があれば GPKG 出力時にポイントジオメトリを生成可能。
      * それ以外の数値列はすべて候補特徴量として扱える。
  - 目的変数（ターゲット）:
      * 列名は任意（例: LandClass, label など）
      * 文字列ラベルの場合は LabelEncoder で内部的に整数に変換される。
  - run_id:
      * 学習ごとに一意な ID（例: 20251113_163000_train001）
      * モデル・メタ情報・図表・CSV のファイル名に付与される。
  - 座標参照系（CRS）:
      * ラスター・ポリゴンは同一CRSであることが前提。
      * GPKG 出力時には EPSG コード等を対話的に指定（ask_crs）する。

依存: numpy, pandas, scikit-learn, joblib, matplotlib, (pyarrow: Parquet I/O),
     geopandas, shapely, fiona, (pyogrio: 推奨), rasterio（学習用データ作成で使用）
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import joblib

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import geopandas as gpd

try:
    from pyogrio import read_dataframe as _pg_read
    _HAS_PYOGRIO = True
except Exception:
    _HAS_PYOGRIO = False

import fiona

# =========================================================
# 共通ヘルパ
# =========================================================
def save_gpkg_with_points(df, out_path, x_col="x", y_col="y",
                          crs_epsg="EPSG:4326", layer_name="pred"):
    import geopandas as gpd
    from shapely.geometry import Point

    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"GPKGに書くには '{x_col}', '{y_col}' 列が必要です。")

    w = df.dropna(subset=[x_col, y_col]).copy()
    w[x_col] = w[x_col].astype(float)
    w[y_col] = w[y_col].astype(float)

    geom = [Point(xy) for xy in zip(w[x_col].values, w[y_col].values)]
    gdf = gpd.GeoDataFrame(w, geometry=geom, crs=crs_epsg)

    # pyogrio があれば高速・安定。無ければ Fiona/GDAL にフォールバック
    try:
        import pyogrio  # noqa: F401
        gdf.to_file(out_path, driver="GPKG", layer=layer_name, engine="pyogrio")
    except Exception:
        gdf.to_file(out_path, driver="GPKG", layer=layer_name)

def ask_crs(default_epsg="EPSG:4326"):
    """
    例: 'EPSG:6677', 'EPSG:4326', 'JGD2011 / Japan Plane Rectangular CS IX' など
    pyproj が解釈できればOK。空Enterで default を返す。
    """
    try:
        from pyproj import CRS
    except Exception:
        # 最低限、pyprojが無くても既定を返せば動く
        return default_epsg

    while True:
        s = input(f"保存する座標系（空={default_epsg}。例: EPSG:4326 / EPSG:6677）: ").strip()
        if not s:
            return default_epsg
        try:
            crs = CRS.from_user_input(s)  # なんでも判定
            # EPSG が取れればEPSG形式に正規化、無理ならWKT/OGC表記を返す
            return f"EPSG:{crs.to_epsg()}" if crs.to_epsg() else crs.to_wkt()
        except Exception as e:
            print(f"⚠ その指定は解釈できませんでした: {e}\nもう一度入力してください。")

def setup_matplotlib_japanese_font():
    import matplotlib
    import matplotlib.font_manager as fm

    # 環境別の“ありそうなフォント”候補（上から優先）
    candidates = [
        # Windows
        "Meiryo", "Yu Gothic", "MS Gothic",
        # macOS
        "Hiragino Sans", "Hiragino Kaku Gothic ProN", "YuGothic",
        # Linux 系
        "Noto Sans CJK JP", "IPAGothic", "TakaoGothic",
    ]

    installed = {f.name for f in fm.fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in installed:
            chosen = name
            break

    if chosen is None:
        # どうしても無ければデフォルトのまま（警告は出さない）
        return

    matplotlib.rcParams["font.family"] = chosen
    matplotlib.rcParams["axes.unicode_minus"] = False  # マイナス記号の豆腐回避

# =========================================================
# ユーティリティ
# =========================================================

MODEL_DEFAULT = "rf_model.joblib"
META_DEFAULT  = "rf_meta.json"
IMP_DEFAULT   = "rf_feature_importance.csv"


def strip_quotes(s: str) -> str:
    return s.strip().strip('"').strip("'")

def list_columns(df: pd.DataFrame, title="カラム一覧"):
    print(f"\n=== {title} ===")
    for i, c in enumerate(df.columns):
        print(f"[{i:03d}] {c}")

def input_indices(prompt: str, max_index: int, allow_empty=False):
    """
    カンマ区切りの整数インデックス入力を受け取り、整数リストを返す。
    allow_empty=False のとき、空Enterは再入力を促す。
    """
    while True:
        s = input(prompt).strip()
        if not s:
            if allow_empty:
                return []
            print("  空行は無効です。少なくとも1つは選んでください。")
            continue
        out = []
        ok = True
        for token in s.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                idx = int(token)
            except ValueError:
                print(f"  ⚠ 整数として解釈できません: {token}")
                ok = False
                break
            if not (0 <= idx <= max_index):
                print(f"  ⚠ 範囲外です: {idx}（0〜{max_index}）")
                ok = False
                break
            out.append(idx)
        if ok and out:
            return out

def ask_yes_no(prompt: str, default: bool | None = None) -> bool:
    """
    y/n を聞く簡易プロンプト。default=None のときは明示的な入力が必要。
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

def _safe_read_table(path: str | Path) -> pd.DataFrame:
    """
    CSV / Parquet / GPKG を自動判別して読み込む。
    GPKG の場合は geometry を取り除き、属性のみの DataFrame を返す。
    """
    path = str(path)
    low = path.lower()
    if low.endswith(".csv"):
        print(f"[INFO] CSV を読み込み中: {path}")
        return pd.read_csv(path)
    if low.endswith(".parquet") or low.endswith(".pq"):
        print(f"[INFO] Parquet を読み込み中: {path}")
        try:
            return pd.read_parquet(path)
        except Exception as e:
            print(f"⚠ Parquet 読み込み失敗: {e}")
            raise
    if low.endswith(".gpkg"):
        print(f"[INFO] GPKG（属性のみ）を読み込み中: {path}")
        if _HAS_PYOGRIO:
            gdf = _pg_read(path)
        else:
            gdf = gpd.read_file(path)
        # geometry を除いた属性のみ返す
        if "geometry" in gdf.columns:
            return pd.DataFrame(gdf.drop(columns=["geometry"]))
        return pd.DataFrame(gdf)
    raise ValueError(f"対応していない拡張子です: {path}")

def _read_gpkg_with_geom(path: str | Path, layer: str | None = None) -> gpd.GeoDataFrame:
    """
    GPKG を GeoDataFrame として読み込むヘルパ。pyogrio があれば優先使用。
    """
    if _HAS_PYOGRIO:
        return _pg_read(path, layer=layer)
    return gpd.read_file(path, layer=layer)


def auto_detect_xy_columns(df: pd.DataFrame):
    """
    x,y 座標の候補となる列を簡易検出する。
    """
    x_candidates = [c for c in df.columns if re.fullmatch(r"[Xx]|x_coord|X_COORD|lon|Lon|LON|longitude", c)]
    y_candidates = [c for c in df.columns if re.fullmatch(r"[Yy]|y_coord|Y_COORD|lat|Lat|LAT|latitude", c)]

    x_col = x_candidates[0] if x_candidates else None
    y_col = y_candidates[0] if y_candidates else None
    return x_col, y_col

def ensure_xy_columns(df: pd.DataFrame):
    """
    x,y 列が存在するか確認し、なければユーザーに聞く。
    """
    x_col, y_col = auto_detect_xy_columns(df)
    print("\n[座標列の確認]")
    if x_col and y_col:
        print(f"  検出された x,y 列: x={x_col}, y={y_col}")
        if ask_yes_no("この列を x,y として使ってよいですか？", default=True):
            return x_col, y_col

    list_columns(df)
    max_idx = len(df.columns) - 1
    print("x 列に使うカラム番号を指定してください。")
    x_idx = input_indices("x 列インデックス: ", max_idx)[0]
    print("y 列に使うカラム番号を指定してください。")
    y_idx = input_indices("y 列インデックス: ", max_idx)[0]
    return df.columns[x_idx], df.columns[y_idx]


# =========================================================
# 学習用データ作成モード（ラスター + ポリゴン属性）
# =========================================================

def make_training_data_mode():
    """
    対象ディレクトリ内のラスター（GeoTIFF 等）とポリゴンGPKGを使って
    グリッドポイント上に特徴量テーブルを作成するモード。
      - 範囲: 手動BBOX または ポリゴンGPKGの外接矩形
      - 解像度: ユーザー指定（座標系と同じ単位）
    出力: CSV / Parquet / GPKG
    （この出力をそのまま train_mode() の入力として利用できる）
    """
    print("\n=== 学習用データ作成モード（ラスター + ポリゴン属性） ===")
    root_dir = strip_quotes(input("特徴量ラスター / ポリゴンGPKG が入ったディレクトリ: ").strip())
    if not root_dir:
        print("ディレクトリが指定されていません。終了します。")
        return
    if not os.path.isdir(root_dir):
        print("ディレクトリが見つかりません。終了します。")
        return

    # --- ラスター/ポリゴンの探索 ---
    raster_exts = (".tif", ".tiff", ".img")
    rasters: list[Path] = []
    polygons: list[Path] = []
    for p in Path(root_dir).rglob("*"):
        if not p.is_file():
            continue
        low = p.suffix.lower()
        if low in raster_exts:
            rasters.append(p)
        elif low == ".gpkg":
            polygons.append(p)

    if not rasters and not polygons:
        print("ラスター(.tif 等) もポリゴン(.gpkg) も見つかりませんでした。終了します。")
        return

    if rasters:
        print("\n[INFO] 検出したラスター:")
        for i, r in enumerate(rasters):
            print(f"  [{i:02d}] {r}")
    else:
        print("\n[INFO] ラスターは検出されませんでした（ポリゴン属性だけでもテーブルは作成できます）。")

    if polygons:
        print("\n[INFO] 検出したポリゴンGPKG:")
        for i, g in enumerate(polygons):
            print(f"  [{i:02d}] {g}")
    else:
        print("\n[INFO] ポリゴンGPKGは検出されませんでした。")

    # どのラスターを使うか（空Enter=全て）
    if rasters:
        use_r_idx = input("\n特徴量として使用するラスターの番号（カンマ区切り、空=全て）: ").strip()
        if use_r_idx:
            idxs: list[int] = []
            for tok in use_r_idx.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    idxs.append(int(tok))
                except ValueError:
                    print(f"  ⚠ 無視します: '{tok}'")
            rasters = [rasters[i] for i in idxs if 0 <= i < len(rasters)]
            if not rasters:
                print("  ⚠ 有効なラスターが選択されませんでした。ラスターなしで続行します。")

    # 属性を付与するポリゴンGPKG（任意）
    poly_for_attr: Path | None = None
    if polygons:
        ans = input("ポリゴンGPKGの属性値も特徴量/ラベルとして付与しますか？ [y/N]: ").strip().lower() or "n"
        if ans.startswith("y"):
            p_idx = input("  使用するポリゴンGPKGの番号（空=0）: ").strip()
            try:
                p_i = int(p_idx) if p_idx else 0
            except ValueError:
                p_i = 0
            if 0 <= p_i < len(polygons):
                poly_for_attr = polygons[p_i]
            else:
                print("  ⚠ 範囲外の番号なのでポリゴン属性は使用しません。")

    # --- 範囲の決定 ---
    print("\n[範囲の決め方]")
    print("  1) 手動で xmin, ymin, xmax, ymax を入力する")
    print("  2) ポリゴンGPKGの外接矩形から決める")
    area_mode = input("番号を選んでください [1/2]（空=1）: ").strip() or "1"

    if area_mode == "1":
        try:
            xmin = float(input("xmin: ").strip())
            ymin = float(input("ymin: ").strip())
            xmax = float(input("xmax: ").strip())
            ymax = float(input("ymax: ").strip())
        except Exception:
            print("数値として解釈できませんでした。終了します。")
            return
        poly_for_extent = None
    else:
        if not polygons:
            print("ポリゴンGPKGが見つからないため、手動入力に切り替えます。")
            return make_training_data_mode()

        # 範囲取得用 GPKG
        if not poly_for_attr:
            print("\n[範囲用ポリゴンの選択]")
            for i, g in enumerate(polygons):
                print(f"  [{i:02d}] {g}")
            p_idx = input("範囲用ポリゴンGPKGの番号（空=0）: ").strip()
            try:
                p_i = int(p_idx) if p_idx else 0
            except ValueError:
                p_i = 0
            if not (0 <= p_i < len(polygons)):
                print("番号が不正です。終了します。")
                return
            poly_for_extent = polygons[p_i]
        else:
            poly_for_extent = poly_for_attr

        try:
            import fiona
            layers = fiona.listlayers(poly_for_extent)
            print(f"  範囲用GPKGのレイヤ一覧: {', '.join(layers)}")
        except Exception:
            layers = None
            print("  （レイヤ一覧の取得に失敗しました）")

        layer_name = input("  使用レイヤ名（空=最初のレイヤ）: ").strip()
        if not layer_name and layers:
            layer_name = layers[0]

        extent_gdf = gpd.read_file(poly_for_extent, layer=layer_name or None)
        xmin, ymin, xmax, ymax = extent_gdf.total_bounds
        print(f"  → ポリゴン外接矩形: xmin={xmin:.3f}, ymin={ymin:.3f}, xmax={xmax:.3f}, ymax={ymax:.3f}")

    # --- 解像度 ---
    res_in = input("\nグリッド解像度（サンプリング間隔、単位は座標系と同じ）[1.0]: ").strip()
    try:
        res = float(res_in) if res_in else 1.0
    except Exception:
        print("数値として解釈できませんでした（1.0 を使用）。")
        res = 1.0

    if res <= 0:
        print("解像度は正の値である必要があります。終了します。")
        return

    # --- グリッドポイント生成 ---
    xs = np.arange(xmin + res / 2.0, xmax, res)
    ys = np.arange(ymin + res / 2.0, ymax, res)
    if len(xs) == 0 or len(ys) == 0:
        print("グリッドが空になってしまいました。範囲や解像度を見直してください。")
        return
    XX, YY = np.meshgrid(xs, ys)
    df = pd.DataFrame({"x": XX.ravel(), "y": YY.ravel()})
    print(f"\n[INFO] グリッドポイント数: {len(df):,} 点")

    # --- ラスターサンプリング ---
    base_crs = None
    if rasters:
        try:
            import rasterio
        except Exception as e:
            print(f"⚠ rasterio がインポートできませんでした: {e}")
            print("   ラスターからのサンプリングはスキップします。")
            rasters = []

    for r_path in rasters:
        import rasterio
        with rasterio.open(r_path) as src:
            if base_crs is None:
                base_crs = src.crs
            else:
                if src.crs != base_crs:
                    raise RuntimeError(f"CRS が一致しません: {r_path} ({src.crs}) != {base_crs}")
            coords = list(zip(df["x"].values, df["y"].values))
            vals = list(src.sample(coords))
            arr = np.array(vals)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            stem = r_path.stem
            for b in range(arr.shape[1]):
                col = f"{stem}_b{b+1}"
                df[col] = arr[:, b]
            nodata = src.nodata
            if nodata is not None:
                df.replace(nodata, np.nan, inplace=True)
        print(f"  [OK] {r_path.name} をサンプリングして特徴量列を追加しました。")

    # --- ポリゴン属性の付与 ---
    if poly_for_attr is not None:
        from shapely.geometry import Point

        print(f"\n[INFO] ポリゴン属性の付与: {poly_for_attr}")
        poly_gdf = gpd.read_file(poly_for_attr)
        if base_crs is not None and poly_gdf.crs is not None:
            try:
                poly_gdf = poly_gdf.to_crs(base_crs)
            except Exception as e:
                print(f"⚠ CRSの変換に失敗しました: {e}\n   ポリゴン属性の付与をスキップします。")
                poly_gdf = None
        if poly_gdf is not None:
            attr_cols = [c for c in poly_gdf.columns if c != "geometry"]
            if attr_cols:
                print("  利用可能な属性列:")
                for i, c in enumerate(attr_cols):
                    print(f"    [{i:02d}] {c}")
                sel = input("  付与する属性列の番号（カンマ区切り、空=全て）: ").strip()
                if sel:
                    idxs: list[int] = []
                    for tok in sel.split(","):
                        tok = tok.strip()
                        if not tok:
                            continue
                        try:
                            idxs.append(int(tok))
                        except ValueError:
                            print(f"    ⚠ 無視します: '{tok}'")
                    attr_cols = [attr_cols[i] for i in idxs if 0 <= i < len(attr_cols)]
            else:
                print("  ⚠ 利用可能な属性列がありません（geometry のみ）。")

            pts_gdf = gpd.GeoDataFrame(
                df.copy(),
                geometry=[Point(xy) for xy in zip(df["x"].values, df["y"].values)],
                crs=poly_gdf.crs,
            )
            joined = gpd.sjoin(pts_gdf, poly_gdf[attr_cols + ["geometry"]] if attr_cols else poly_gdf,
                               how="left", predicate="intersects")
            drop_cols = [c for c in joined.columns if c in ("index_right",)]
            joined = joined.drop(columns=drop_cols)
            if "geometry" in joined.columns:
                joined = joined.drop(columns=["geometry"])
            df = pd.DataFrame(joined)
            print(f"  → ポリゴン属性を付与しました（列数: {len(attr_cols) if attr_cols else 0}）")

    # --- 出力 ---
    print("\n[出力]")
    out_path = strip_quotes(input("学習用テーブルの出力パス（.csv / .parquet / .gpkg）: ").strip())
    if not out_path:
        print("出力パスが指定されていません。終了します。")
        return

    out_suffix = Path(out_path).suffix.lower()
    if out_suffix in [".parquet", ".pq"]:
        try:
            df.to_parquet(out_path, index=False)
            print(f"✅ Parquet を書き出しました: {out_path}")
        except Exception as e:
            print(f"Parquet の書き出しに失敗しました: {e}")
    elif out_suffix == ".gpkg":
        # base_crs が取れていればそれを既定に
        if base_crs is not None:
            try:
                def_str = base_crs.to_string()
            except Exception:
                def_str = "EPSG:4326"
        else:
            def_str = "EPSG:4326"
        crs_epsg = ask_crs(default_epsg=def_str)
        save_gpkg_with_points(df, out_path, x_col="x", y_col="y",
                              crs_epsg=crs_epsg, layer_name="train")
        print(f"✅ GPKG を書き出しました: {out_path}")
    else:
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"✅ CSV を書き出しました: {out_path}")

    print("\n[完了] この出力テーブルを学習モードの入力として指定できます。")


# =========================================================
# 学習用テーブル読み込み・列選択まわり
# =========================================================

def load_table_interactive():
    """
    学習/予測に使うテーブルを対話的に読み込む。
    """
    path = strip_quotes(input("入力テーブルのパス（CSV/Parquet/GPKG）: ").strip())
    if not path:
        print("入力が空です。終了します。")
        sys.exit(1)
    if not os.path.exists(path):
        print(f"ファイルが見つかりません: {path}")
        sys.exit(1)

    df = _safe_read_table(path)
    print(f"[INFO] テーブル読み込み完了: {path}（{len(df):,} 行, {len(df.columns)} 列）")
    return df, path


def choose_target_and_features(df: pd.DataFrame):
    """
    目的変数（ターゲット）と特徴量列を対話的に選択する。
    """
    list_columns(df)
    max_idx = len(df.columns) - 1
    print("\n[ターゲット列の選択]")
    t_idx = input_indices("目的変数に使うカラム番号（1つ）: ", max_idx)[0]
    target_col = df.columns[t_idx]

    print("\n[特徴量列の選択]")
    print("※ 目的変数列、明らかなID列などは除外してください。")
    f_idxs = input_indices("特徴量に使うカラム番号（カンマ区切り）: ", max_idx)
    feature_cols = [df.columns[i] for i in f_idxs]

    print("\n[確認]")
    print(f"  目的変数: {target_col}")
    print(f"  特徴量: {feature_cols}")
    return target_col, feature_cols


def _alias_candidates(col: str):
    """
    特徴量名の "ゆらぎ" を吸収するための簡易エイリアス候補生成。
    例: 'slope_deg_r10m' → 'SlopeDeg_r10m' 等。
    """
    c = col
    yield c
    yield c.lower()
    yield c.upper()
    yield c.replace(" ", "")
    yield c.replace(" ", "_")
    yield re.sub(r"[^0-9A-Za-z_]+", "", c)


# =========================================================
# 学習（train）
# =========================================================

def train_mode():
    print("\n=== 学習モード（train） ===")
    df, path = load_table_interactive()

    # ターゲット＋特徴量列
    target_col, feature_cols = choose_target_and_features(df)

    # ターゲットが文字列なら LabelEncoder で整数化
    y_raw = df[target_col]
    if y_raw.dtype.kind in ("O", "U", "S"):
        print("[INFO] 文字列ラベルを LabelEncoder で整数化します。")
        le = LabelEncoder()
        y = le.fit_transform(y_raw.values)
        class_names = list(le.classes_)
        label_encoder_info = {
            "classes_": class_names,
            "target_col": target_col,
        }
    else:
        y = y_raw.values
        class_names = sorted(pd.unique(y))
        label_encoder_info = None

    X = df[feature_cols].values.astype(float)

    print("\n[学習データ分割設定]")
    test_size = input("テストデータ割合（0〜0.5 程度, 空=0.2）: ").strip()
    if test_size:
        try:
            test_size = float(test_size)
        except ValueError:
            print("  ⚠ 数値変換に失敗したので 0.2 を使います。")
            test_size = 0.2
    else:
        test_size = 0.2

    random_state = input("random_state（空=42）: ").strip()
    if random_state:
        try:
            random_state = int(random_state)
        except ValueError:
            print("  ⚠ 整数変換に失敗したので 42 を使います。")
            random_state = 42
    else:
        random_state = 42

    stratify = y if ask_yes_no("層化サンプリングを使いますか？", default=True) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=stratify
    )

    print("\n[RandomForest パラメータ設定]")
    n_estimators = input("n_estimators（木の本数, 空=200）: ").strip()
    if n_estimators:
        try:
            n_estimators = int(n_estimators)
        except ValueError:
            print("  ⚠ 整数変換に失敗したので 200 を使います。")
            n_estimators = 200
    else:
        n_estimators = 200

    max_depth = input("max_depth（空=None=制限なし）: ").strip()
    if max_depth:
        try:
            max_depth = int(max_depth)
        except ValueError:
            print("  ⚠ 整数変換に失敗したので None を使います。")
            max_depth = None
    else:
        max_depth = None

    class_weight = None
    if ask_yes_no("クラス不均衡対策として class_weight='balanced' を使いますか？", default=False):
        class_weight = "balanced"

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight=class_weight,
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", rf),
    ])

    print("\n[学習中...]")
    pipe.fit(X_train, y_train)
    print("学習完了。")

    # 評価
    print("\n[評価（テストデータ）]")
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("混同行列（行: 真, 列: 予測）:")
    print(cm)
    print("\nclassification_report:")
    if label_encoder_info:
        target_names = class_names
    else:
        target_names = [str(c) for c in class_names]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 実験ID (run_id)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{now}_train001"

    # 保存ディレクトリ
    base = Path(path)
    out_dir = base.parent
    model_path = out_dir / f"rf_model_{run_id}.joblib"
    meta_path  = out_dir / f"rf_meta_{run_id}.json"
    imp_path   = out_dir / f"rf_feature_importance_{run_id}.csv"

    # モデル保存
    joblib.dump(pipe, model_path)
    print(f"\n[保存] モデル: {model_path}")

    # メタ情報 JSON
    meta = {
        "run_id": run_id,
        "source_path": str(path),
        "target_col": target_col,
        "feature_cols": feature_cols,
        "class_names": target_names,
        "label_encoder": label_encoder_info,
        "train_params": {
            "test_size": test_size,
            "random_state": random_state,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "class_weight": class_weight,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[保存] メタ情報: {meta_path}")

    # 特徴量重要度
    rf_trained: RandomForestClassifier = pipe.named_steps["rf"]
    importances = rf_trained.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    imp_df.to_csv(imp_path, index=False, encoding="utf-8")
    print(f"[保存] 特徴量重要度 CSV: {imp_path}")

    # 特徴量重要度グラフ
    topn = min(25, len(imp_df))
    fig, ax = plt.subplots(figsize=(8, max(4, topn * 0.3)))
    ax.barh(np.arange(topn), imp_df["importance"].values[:topn][::-1])
    ax.set_yticks(np.arange(topn))
    ax.set_yticklabels(imp_df["feature"].values[:topn][::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Feature importance (top 25)")
    plt.tight_layout()
    fig_path = out_dir / f"rf_feature_importance_{run_id}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[保存] 特徴量重要度 図: {fig_path}")

    # 混同行列の図（絶対値 & 行正規化）
    def _plot_confusion_matrix(cm, normalize=False, title="Confusion matrix", cmap=None, save_path=None):
        if normalize:
            cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_plot = cmn
        else:
            cm_plot = cm

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_plot, interpolation="nearest", cmap=cmap or "Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set_title(title)
        tick_marks = np.arange(len(target_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(target_names, rotation=45, ha="right")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(target_names)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")

        fmt = ".2f" if normalize else "d"
        thresh = cm_plot.max() / 2.0
        for i in range(cm_plot.shape[0]):
            for j in range(cm_plot.shape[1]):
                ax.text(
                    j, i, format(cm_plot[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_plot[i, j] > thresh else "black",
                )
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=150)
            print(f"[保存] 混同行列 図: {save_path}")
        plt.close(fig)

    cm_abs_path = out_dir / f"rf_confusion_matrix_{run_id}.png"
    cm_norm_path = out_dir / f"rf_confusion_matrix_normalized_{run_id}.png"
    _plot_confusion_matrix(cm, normalize=False,
                           title="Confusion matrix (test)",
                           save_path=cm_abs_path)
    _plot_confusion_matrix(cm, normalize=True,
                           title="Confusion matrix (normalized, test)",
                           save_path=cm_norm_path)

    print("\n[完了] 学習モードが正常に終了しました。")


# =========================================================
# 予測（predict）
# =========================================================

def load_model_and_meta():
    """
    保存済みモデルとメタ情報を対話的に読み込む。
    """
    model_path = strip_quotes(input(f"モデルのパス（空={MODEL_DEFAULT}）: ").strip())
    if not model_path:
        model_path = MODEL_DEFAULT
    if not os.path.exists(model_path):
        print(f"モデルファイルが見つかりません: {model_path}")
        sys.exit(1)

    meta_path = strip_quotes(input(f"メタ情報 JSON のパス（空={META_DEFAULT}）: ").strip())
    if not meta_path:
        meta_path = META_DEFAULT
    if not os.path.exists(meta_path):
        print(f"メタ情報ファイルが見つかりません: {meta_path}")
        sys.exit(1)

    pipe = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"[INFO] モデルとメタ情報を読み込みました。run_id={meta.get('run_id')}")
    return pipe, meta, model_path, meta_path


def predict_mode():
    print("\n=== 予測モード（predict） ===")
    pipe, meta, model_path, meta_path = load_model_and_meta()

    feature_cols = meta["feature_cols"]
    target_col = meta["target_col"]
    class_names = meta.get("class_names")
    label_encoder_info = meta.get("label_encoder")

    df, in_path = load_table_interactive()

    # 特徴量列の整合性チェック
    print("\n[特徴量列の整合性チェック]")
    saved_feats = feature_cols
    alias_map = {}
    missing_feats = []

    for c in saved_feats:
        if c in df.columns:
            continue
        hit = None
        for a in _alias_candidates(c):
            if a in df.columns:
                hit = a
                break
        if hit is not None:
            df[c] = df[hit]         # 学習名で複製
            alias_map[c] = hit
        else:
            missing_feats.append(c)

    if alias_map:
        print("  以下の特徴量名はエイリアスで補正しました:")
        for k, v in alias_map.items():
            print(f"    {k} ← {v}")

    if missing_feats:
        print("  ⚠ 以下の特徴量は入力テーブルに見つかりませんでした:")
        for c in missing_feats:
            print(f"    {c}")
        if not ask_yes_no("このまま欠損列として続行しますか？（imputerで処理されます）", default=True):
            print("中断します。")
            return
        for c in missing_feats:
            df[c] = np.nan

    X = df[saved_feats].values.astype(float)

    print("\n[予測中...]")
    y_pred = pipe.predict(X)
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        proba_max = proba.max(axis=1)
    else:
        proba_max = None

    # ラベルを元に戻す
    if label_encoder_info and class_names:
        # 予測は整数 → クラス名
        y_pred_labels = [class_names[int(i)] for i in y_pred]
    else:
        y_pred_labels = y_pred

    out = df.copy()
    out["y_pred"] = y_pred_labels
    if proba_max is not None:
        out["proba_max"] = proba_max

    # 正解ラベルがあれば評価
    if target_col in df.columns:
        print("\n[評価（入力テーブルに正解ラベルが含まれていたため）]")
        y_true = df[target_col].values
        # 文字列 vs 整数の可能性があるので揃える
        if label_encoder_info and class_names:
            # y_true がクラス名であることを期待
            y_true_int = np.array([class_names.index(str(v)) if str(v) in class_names else -1 for v in y_true])
            valid_mask = y_true_int >= 0
            if not np.any(valid_mask):
                print("  ⚠ y_true がクラス名と一致しなかったため、評価をスキップします。")
            else:
                y_true_valid = y_true_int[valid_mask]
                y_pred_valid = y_pred[valid_mask]
                cm = confusion_matrix(y_true_valid, y_pred_valid)
                print("混同行列（行: 真, 列: 予測）:")
                print(cm)
                print("\nclassification_report:")
                print(classification_report(
                    y_true_valid, y_pred_valid,
                    target_names=class_names,
                ))
        else:
            cm = confusion_matrix(y_true, y_pred_labels)
            print("混同行列（行: 真, 列: 予測）:")
            print(cm)
            print("\nclassification_report:")
            print(classification_report(
                y_true, y_pred_labels,
                target_names=class_names if class_names else None,
            ))

    # 出力ファイル名
    base_in = str(Path(in_path).with_suffix(""))
    run_id = meta.get("run_id", "predict")
    out_path = input(f"\n予測結果の保存先（空なら {base_in}_pred_{run_id}.ext）: ").strip()
    if not out_path:
        # 入力の拡張子に応じて自動決定
        low = in_path.lower()
        if low.endswith(".csv"):
            out_path = f"{base_in}_pred_{run_id}.csv"
        elif low.endswith(".parquet") or low.endswith(".pq"):
            out_path = f"{base_in}_pred_{run_id}.parquet"
        elif low.endswith(".gpkg"):
            out_path = f"{base_in}_pred_{run_id}.gpkg"
        else:
            out_path = f"{base_in}_pred_{run_id}.csv"

    low = out_path.lower()
    if low.endswith(".csv"):
        out.to_csv(out_path, index=False, encoding="utf-8")
        print(f"\n✅ 予測結果（CSV）を書き出しました: {out_path}")

    elif low.endswith(".gpkg"):
        # 出力GPKGの場合、x,y列を使ってポイントにするかどうか確認
        if "geometry" in out.columns:
            use_geom = ask_yes_no("入力に geometry 列があるので、そのままGPKGに書き出しますか？", default=True)
        else:
            use_geom = False

        if use_geom:
            gtmp = gpd.GeoDataFrame(out, geometry="geometry")
            crs = ask_crs(default_epsg=str(gtmp.crs) if gtmp.crs else "EPSG:4326")
            gtmp.set_crs(crs, inplace=True)
            layer_name = input("GPKGのレイヤ名（空=pred）: ").strip() or "pred"
            gtmp.to_file(out_path, driver="GPKG", layer=layer_name)
            print(f"\n✅ 予測結果（GPKG）を書き出しました: {out_path}（レイヤ: {layer_name}, CRS={crs}）")
        else:
            # x,y 列があればそこから点を作る
            if {"x", "y"}.issubset(out.columns):
                crs = ask_crs()
                layer_name = input("GPKGのレイヤ名（空=pred）: ").strip() or "pred"
                save_gpkg_with_points(out, out_path, x_col="x", y_col="y",
                                      crs_epsg=crs, layer_name=layer_name)
                print(f"\n✅ 予測結果（GPKG）を書き出しました: {out_path}（レイヤ: {layer_name}, CRS={crs}）")
            else:
                print("geometry も x,y も見つからないため、CSV で保存します。")
                fallback = str(Path(out_path).with_suffix(".csv"))
                out.to_csv(fallback, index=False, encoding="utf-8")
                print(f"\n✅ 予測結果（CSV）を書き出しました: {fallback}")

    elif low.endswith(".parquet") or low.endswith(".pq"):
        try:
            out.to_parquet(out_path, index=False)
            print(f"\n✅ 予測結果（Parquet）を書き出しました: {out_path}")
        except Exception as e:
            print(f"Parquet 失敗 → CSVで再保存します: {e}")
            out_fallback = f"{base_in}_pred_{run_id}.csv"
            out.to_csv(out_fallback, index=False, encoding="utf-8")
            print(f"\n✅ 予測結果（CSV）を書き出しました: {out_fallback}")
    else:
        # よく分からない拡張子 → CSV にフォールバック
        fallback = f"{base_in}_pred_{run_id}.csv"
        out.to_csv(fallback, index=False, encoding="utf-8")
        print(f"\n✅ 予測結果（CSV）を書き出しました: {fallback}")

    print("\n[完了] 予測モードが正常に終了しました。")


# =========================================================
# メイン
# =========================================================

def main():
    print("\n=== ランダムフォレスト（地形分類＋学習データ作成） ===")
    print("  1) 学習用データ作成（ラスター/ポリゴン → テーブル）")
    print("  2) 学習（train）")
    print("  3) 予測（predict）")
    print("  0) 終了")
    choice = input("番号を選んでください [0-3]: ").strip() or "2"

    if choice == "1":
        make_training_data_mode()
    elif choice == "2":
        train_mode()
    elif choice == "3":
        predict_mode()
    else:
        print("終了します。")

# フォントセット（スクリプト開始時に一度でOK。未設定ならここで）
if __name__ == "__main__":
    try:
        setup_matplotlib_japanese_font()  # ← ここで一度だけ
        main()
    except KeyboardInterrupt:
        print("\n中断しました。")
