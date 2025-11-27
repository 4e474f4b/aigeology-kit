#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
point_class_to_grid_ABC.py

粗密のある「点の地形種別（クラス）」から、
ユーザー指定解像度のグリッド面データを作成するスクリプト。

3種類の補間・集約方式を選択可能:
  A) 最近傍（Nearest Neighbor）
  B) 半径内多数決（Kernel Majority）
  C) 半径内 距離減衰付き多数決（Weighted Majority）

想定用途
--------
- ランダムフォレスト等で予測された点の地形クラスを、
  1m / 5m / 10m など任意解像度のグリッド面データに変換したい。
- 元の予測点は HoldOut / k分割CV の都合で粗密があってもよい。
- 出力は「セルごとに1クラスを持つポリゴン（グリッド）」とし、
  必要に応じてクラスごとの dissolve（面マージ）を行う。

依存ライブラリ
---------------
- pandas
- geopandas
- shapely
- scikit-learn（最近傍・半径近傍探索に使用）
- pyarrow（parquet 入力時）

注意
----
- 最近傍方式（A）は常に「フルカバーのグリッド」を返す。
- 半径方式（B/C）は「その半径内に点がないセル」は NoData（クラス欠損）となる。
"""

import os
import sys
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

from sklearn.neighbors import NearestNeighbors as SklearnNearestNeighbors

# GPU (cuML) が使える環境では、POINTGRID_USE_GPU=1 のときに GPU バックエンドを利用する
try:
    from cuml.neighbors import NearestNeighbors as CuMLNearestNeighbors  # type: ignore
    _HAS_CUML = True
except Exception:
    CuMLNearestNeighbors = None  # type: ignore
    _HAS_CUML = False

# 環境変数 POINTGRID_USE_GPU が 1 / true / yes / y なら GPU バックエンドを優先
USE_GPU_NEIGHBORS = os.environ.get("POINTGRID_USE_GPU", "").lower() in ("1", "true", "yes", "y")

# メモリ安全対策用のしきい値
MAX_WARN_CELLS  = 1_000_000   # ここを超えたら強めの警告
MAX_ABORT_CELLS = 5_000_000   # ここを超えたら問答無用で中止
NN_BATCH_SIZE   = 200_000     # 近傍探索時に処理するセル中心のバッチサイズ


def make_nearest_neighbors(
    n_neighbors: Optional[int] = None,
    radius: Optional[float] = None,
):
    """
    近傍探索オブジェクトを生成するヘルパー。
    - GPU (cuML) が利用可能かつ USE_GPU_NEIGHBORS=True なら cuML を使用
    - それ以外は scikit-learn を使用
    """
    use_gpu = USE_GPU_NEIGHBORS and _HAS_CUML
    if use_gpu:
        nn_cls = CuMLNearestNeighbors
    else:
        nn_cls = SklearnNearestNeighbors

    if n_neighbors is not None:
        return nn_cls(n_neighbors=n_neighbors)
    else:
        return nn_cls(radius=radius)


# ============================================================
# 共通ユーティリティ
# ============================================================

def print_header():
    print("=" * 70)
    print("  点の地形クラス → 指定解像度グリッド面データ化ツール（A/B/C方式）")
    print("=" * 70)


def ask_yes_no(prompt: str, default: str = "y") -> bool:
    """
    Y/N を尋ねる簡易プロンプト。
    """
    default = default.lower()
    if default not in ("y", "n"):
        default = "y"

    while True:
        ans = input(prompt + " ").strip().lower()
        if ans == "" and default:
            ans = default
        if ans in ("y", "yes", "はい"):
            return True
        if ans in ("n", "no", "いいえ"):
            return False
        print("[WARN] y / n で入力してください。")


def detect_input_type(path: Path) -> str:
    """
    拡張子から入力種別を判定する。
    """
    ext = path.suffix.lower()
    if ext in (".gpkg", ".shp", ".geojson"):
        return "gpkg"
    if ext in (".parq", ".parquet"):
        return "parquet"
    if ext in (".csv", ".txt"):
        return "csv"
    return ""


def list_columns(df: pd.DataFrame):
    """
    DataFrame のカラムをインデックス付きで表示する。
    """
    print("\n[INFO] 利用可能なカラム一覧:")
    for i, col in enumerate(df.columns):
        print(f"  [{i:02d}] {col}")


def ask_column_name(df: pd.DataFrame, purpose: str) -> str:
    """
    X / Y / クラス列などを選択させる対話プロンプト。
    """
    list_columns(df)
    while True:
        val = input(f"\n{purpose} に使うカラム番号または名前を入力してください: ").strip()
        if val == "":
            print("[WARN] 空欄は指定できません。")
            continue

        # 数字（インデックス）指定
        if val.isdigit():
            idx = int(val)
            if 0 <= idx < len(df.columns):
                col = df.columns[idx]
                print(f"[INFO] 選択されたカラム: {col}")
                return col
            else:
                print("[WARN] インデックスが範囲外です。")
        else:
            # カラム名指定
            if val in df.columns:
                print(f"[INFO] 選択されたカラム: {val}")
                return val
            else:
                print("[WARN] 指定したカラム名が見つかりません。")


def build_fishnet(bounds: Tuple[float, float, float, float],
                  cell_size: float) -> gpd.GeoDataFrame:
    """
    外接矩形 bounds とセルサイズからフィッシュネットグリッドを作成する。
    セルサイズは X, Y 共通（座標系の単位に依存）。
    """
    minx, miny, maxx, maxy = bounds

    # ここでは簡易的に minx, miny を原点としてそのままグリッドを切る
    nx = int(math.ceil((maxx - minx) / cell_size))
    ny = int(math.ceil((maxy - miny) / cell_size))
    n_cells = nx * ny

    print(f"\n[INFO] グリッド数の見込み: {nx} 列 × {ny} 行 = {n_cells} セル")

    # セル数が大きすぎる場合の安全対策
    if n_cells > MAX_ABORT_CELLS:
        print(
            f"[ERROR] セル数が {MAX_ABORT_CELLS:,} を超えています（推定 {n_cells:,} セル）。\n"
            "       5百万セルを超えたので処理中止します。処理データの大きさを再検討して、\n"
            "       セルサイズを粗くするか、処理領域を分割してから再実行してください。"
        )
        sys.exit(1)
    elif n_cells > MAX_WARN_CELLS:
        print(
            f"[WARN] セル数が {MAX_WARN_CELLS:,} を超えています（推定 {n_cells:,} セル）。\n"
            "       処理時間・メモリ使用量がかなり大きくなる可能性があります。"
        )

    polygons: List[Polygon] = []
    centers_x: List[float] = []
    centers_y: List[float] = []
    grid_ids: List[int] = []

    print("[INFO] フィッシュネットを作成中...")
    gid = 0
    for iy in range(ny):
        y0 = miny + iy * cell_size
        y1 = y0 + cell_size
        cy = (y0 + y1) / 2.0
        for ix in range(nx):
            x0 = minx + ix * cell_size
            x1 = x0 + cell_size
            cx = (x0 + x1) / 2.0
            poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            polygons.append(poly)
            centers_x.append(cx)
            centers_y.append(cy)
            grid_ids.append(gid)
            gid += 1

    grid_gdf = gpd.GeoDataFrame(
        {"grid_id": grid_ids, "center_x": centers_x, "center_y": centers_y},
        geometry=polygons,
        crs=None,
    )
    print(f"[OK] グリッド作成完了（{len(grid_gdf)} セル）。")
    return grid_gdf


def save_geodataframe(gdf: gpd.GeoDataFrame, path: Path):
    """
    GeoDataFrame を拡張子に応じた形式で保存する。
    """
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    if ext == ".shp":
        driver = "ESRI Shapefile"
    else:
        if ext == "":
            path = path.with_suffix(".gpkg")
        driver = "GPKG"

    print(f"[INFO] ファイル出力中: {path}")
    gdf.to_file(path, driver=driver)
    print("[OK] 出力完了。")


# ============================================================
# クラス割り当てロジック（A/B/C）
# ============================================================

def assign_classes_A_nearest(
    grid_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    class_col: str,
) -> gpd.GeoDataFrame:
    """
    A方式: 最近傍（Nearest Neighbor）でクラスを割り当てる。
    - 各グリッドセル中心から最も近い点のクラスをコピー。
    - 少なくとも1点が存在すれば、すべてのセルにクラスが割り当てられる（フルカバー）。
    """
    print("\n[INFO] A方式：最近傍でクラスを割り当てます（Nearest Neighbor）。")

    # クラス欠損・geometry欠損を除外
    g = points_gdf.copy()
    g = g[g.geometry.notnull()].copy()
    g = g[~g[class_col].isna()].copy()

    # 点座標とクラス
    pts_xy = np.vstack([g.geometry.x.values, g.geometry.y.values]).T
    pts_class = g[class_col].values

    if len(pts_xy) == 0:
        print("[ERROR] 点が1つも存在しません。")
        sys.exit(1)

    # セル中心座標
    centers_xy = np.vstack(
        [grid_gdf["center_x"].values, grid_gdf["center_y"].values]
    ).T

    print("[INFO] 最近傍探索用のインデックスを構築中（NearestNeighbors バックエンド）...")
    nn = make_nearest_neighbors(n_neighbors=1)
    nn.fit(pts_xy)

    print("[INFO] 全セル中心に対して最近傍点を検索中（バッチ処理）...")

    n_cells = centers_xy.shape[0]
    batch_size = NN_BATCH_SIZE

    # 結果格納用（事前に配列を確保）
    nearest_idx = np.empty(n_cells, dtype=int)
    nearest_dist = np.empty(n_cells, dtype=float)

    start = 0
    while start < n_cells:
        end = min(start + batch_size, n_cells)
        centers_batch = centers_xy[start:end]

        distances, indices = nn.kneighbors(centers_batch, return_distance=True)
        distances = distances[:, 0]
        indices = indices[:, 0]

        nearest_dist[start:end] = distances
        nearest_idx[start:end] = indices

        print(f"[INFO] 最近傍探索: {end:,} / {n_cells:,} セルまで処理完了...", end="\r")
        start = end

    print(f"\n[INFO] 最近傍探索が完了しました（{n_cells:,} セル）。")

    # クラス割り当て
    grid_gdf = grid_gdf.copy()
    grid_gdf[class_col] = pts_class[nearest_idx]
    grid_gdf["dist_to_point"] = nearest_dist
    grid_gdf["n_points"] = 1  # 最近傍1点に基づくため（便宜上）

    print("[OK] A方式のクラス割り当てが完了しました。")
    return grid_gdf


def assign_classes_B_radius_majority(
    grid_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    class_col: str,
    radius: float,
) -> gpd.GeoDataFrame:
    """
    B方式: 半径内多数決（Kernel Majority）でクラスを割り当てる。
    - 各セル中心の周囲 radius 内にある点のクラスの多数決。
    - 該当点が無いセルはクラス欠損（NaN）となる。
    """
    print("\n[INFO] B方式：半径内多数決でクラスを割り当てます（Kernel Majority）。")
    print(f"[INFO] 使用する半径: {radius} (座標系の単位)")

    pts_xy = np.vstack(
        [points_gdf.geometry.x.values, points_gdf.geometry.y.values]
    ).T
    pts_class = points_gdf[class_col].values

    if len(pts_xy) == 0:
        print("[ERROR] 点が1つも存在しません。")
        sys.exit(1)

    centers_xy = np.vstack(
        [grid_gdf["center_x"].values, grid_gdf["center_y"].values]
    ).T
    n_cells = centers_xy.shape[0]

    print("[INFO] 半径近傍探索用のインデックスを構築中（NearestNeighbors バックエンド）...")
    nn = make_nearest_neighbors(radius=radius)
    nn.fit(pts_xy)

    # 結果格納用
    assigned_class: List[object] = []
    n_points_list: List[int] = []

    print("[INFO] 全セル中心に対して半径内の近傍点を検索中（バッチ処理）...")
    batch_size = NN_BATCH_SIZE
    start = 0
    while start < n_cells:
        end = min(start + batch_size, n_cells)
        centers_batch = centers_xy[start:end]

        distances_list, indices_list = nn.radius_neighbors(
            centers_batch, radius=radius, return_distance=True
        )

        for dists, idxs in zip(distances_list, indices_list):
            idxs = np.asarray(idxs, dtype=int)
            if idxs.size == 0:
                assigned_class.append(np.nan)
                n_points_list.append(0)
                continue

            cls_raw = pts_class[idxs]
            # None / NaN を除外
            cls_clean = [
                c
                for c in cls_raw
                if (c is not None) and not (isinstance(c, float) and np.isnan(c))
            ]

            if len(cls_clean) == 0:
                assigned_class.append(np.nan)
                n_points_list.append(0)
                continue

            # 最頻値（多数決）
            values, counts = np.unique(cls_clean, return_counts=True)
            majority_class = values[np.argmax(counts)]
            assigned_class.append(majority_class)
            n_points_list.append(len(cls_clean))

        print(f"[INFO] 半径内探索: {end:,} / {n_cells:,} セルまで処理完了...", end="\r")
        start = end

    print(f"\n[INFO] 半径内探索が完了しました（{n_cells:,} セル）。")

    grid_gdf = grid_gdf.copy()
    grid_gdf[class_col] = assigned_class
    grid_gdf["n_points"] = n_points_list

    n_valid = np.count_nonzero(~pd.isna(grid_gdf[class_col].values))
    print(f"[INFO] クラスが割り当てられたセル数: {n_valid} / {len(grid_gdf)}")
    print("[OK] B方式のクラス割り当てが完了しました。")
    return grid_gdf


def assign_classes_C_weighted_majority(
    grid_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    class_col: str,
    radius: float,
    power: float = 2.0,
    eps: float = 1e-6,
) -> gpd.GeoDataFrame:
    """
    C方式: 半径内 距離減衰付き多数決（Weighted Majority）でクラスを割り当てる。
    - 各セル中心について、周囲 radius 内の点を取得。
    - 距離 d に対し weight = 1 / (d^power + eps) を算出。
    - クラスごとの重み合計が最大のクラスを選ぶ。
    - 該当点が無いセルはクラス欠損（NaN）となる。
    """
    print("\n[INFO] C方式：半径内 距離減衰付き多数決でクラスを割り当てます（Weighted Majority）。")
    print(f"[INFO] 使用する半径: {radius} (座標系の単位)")
    print(f"[INFO] 距離減衰の指数 power: {power}, eps: {eps}")

    pts_xy = np.vstack(
        [points_gdf.geometry.x.values, points_gdf.geometry.y.values]
    ).T
    pts_class = points_gdf[class_col].values

    if len(pts_xy) == 0:
        print("[ERROR] 点が1つも存在しません。")
        sys.exit(1)

    centers_xy = np.vstack(
        [grid_gdf["center_x"].values, grid_gdf["center_y"].values]
    ).T
    n_cells = centers_xy.shape[0]

    print("[INFO] 半径近傍探索用のインデックスを構築中（NearestNeighbors バックエンド）...")
    nn = make_nearest_neighbors(radius=radius)
    nn.fit(pts_xy)

    assigned_class: List[object] = []
    n_points_list: List[int] = []
    class_conf_list: List[float] = []  # もっとも重みが大きいクラスの重み割合

    print("[INFO] 全セル中心に対して半径内の近傍点を検索中（バッチ処理＋距離取得）...")
    batch_size = NN_BATCH_SIZE
    start = 0
    while start < n_cells:
        end = min(start + batch_size, n_cells)
        centers_batch = centers_xy[start:end]

        distances_list, indices_list = nn.radius_neighbors(
            centers_batch, radius=radius, return_distance=True
        )

        for dists, idxs in zip(distances_list, indices_list):
            idxs = np.asarray(idxs, dtype=int)
            if idxs.size == 0:
                assigned_class.append(np.nan)
                n_points_list.append(0)
                class_conf_list.append(np.nan)
                continue

            dists = np.asarray(dists, dtype=float)

            # 重み計算（d=0 の場合もあるので eps を足しておく）
            weights = 1.0 / (np.power(dists, power) + eps)

            # クラスごとに重みを集計（None / NaN は除外）
            cls = pts_class[idxs]
            weight_by_class: Dict[object, float] = {}
            for c, w in zip(cls, weights):
                if c is None or (isinstance(c, float) and np.isnan(c)):
                    continue
                weight_by_class[c] = weight_by_class.get(c, 0.0) + w

            if not weight_by_class:
                assigned_class.append(np.nan)
                n_points_list.append(0)
                class_conf_list.append(np.nan)
                continue

            # 最大重みのクラス
            major_class, major_weight = max(
                weight_by_class.items(), key=lambda kv: kv[1]
            )
            total_weight = sum(weight_by_class.values())
            conf = major_weight / total_weight if total_weight > 0 else np.nan

            assigned_class.append(major_class)
            n_points_list.append(int(len(cls)))
            class_conf_list.append(float(conf))

        print(f"[INFO] 半径内探索（重み付き）: {end:,} / {n_cells:,} セルまで処理完了...", end="\r")
        start = end

    print(f"\n[INFO] 半径内探索（重み付き）が完了しました（{n_cells:,} セル）。")

    grid_gdf = grid_gdf.copy()
    grid_gdf[class_col] = assigned_class
    grid_gdf["n_points"] = n_points_list
    grid_gdf["class_conf"] = class_conf_list

    n_valid = np.count_nonzero(~pd.isna(grid_gdf[class_col].values))
    print(f"[INFO] クラスが割り当てられたセル数: {n_valid} / {len(grid_gdf)}")
    print("[OK] C方式のクラス割り当てが完了しました。")
    return grid_gdf

def dissolve_by_class(grid_gdf: gpd.GeoDataFrame, class_col: str) -> gpd.GeoDataFrame:
    """
    グリッドポリゴンを地形種別ごとに dissolve する。
    - クラスが NaN のセルは除外してから dissolve する。
    - GeoDataFrame.dissolve() / Shapely の union 系で TypeError が出る環境では、
      無理に union せず「dissolve レイヤは空のまま返す」ようにして処理継続を優先する。
    """
    print("\n[INFO] クラスごとに dissolve（面マージ）を実行します。")

    # クラスが NaN のセルは除外
    g = grid_gdf.dropna(subset=[class_col]).copy()
    if g.empty:
        print("[WARN] クラスが有効なセルが存在しないため、dissolve をスキップします。")
        return gpd.GeoDataFrame(columns=["geometry", class_col, "n_cells", "n_points"], crs=grid_gdf.crs)

    # 念のため geometry の NaN も除外
    g = g[g.geometry.notnull()].copy()

    if "n_points" not in g.columns:
        g["n_points"] = 1

    try:
        dissolved = g.dissolve(by=class_col, as_index=True, aggfunc={"n_points": "sum"})
        count_cells = g.groupby(class_col).size()
        dissolved["n_cells"] = count_cells
        dissolved[class_col] = dissolved.index
        dissolved = dissolved.reset_index(drop=True)
        print(f"[OK] dissolve 完了（{len(dissolved)} クラス）。")
        return dissolved
    except TypeError as e:
        print("[WARN] dissolve 実行中に TypeError が発生しました。dissolve レイヤは作成せずスキップします。")
        print(f"       詳細: {e}")
        # 空の GeoDataFrame を返して呼び出し側に処理を継続させる
        return gpd.GeoDataFrame(columns=["geometry", class_col, "n_cells", "n_points"], crs=grid_gdf.crs)


# ============================================================
# メイン処理
# ============================================================

def main():
    print_header()

    # 近傍探索のバックエンド情報を表示
    if USE_GPU_NEIGHBORS and _HAS_CUML:
        print("[INFO] 近傍探索バックエンド: GPU (cuML, POINTGRID_USE_GPU=1)")
    elif USE_GPU_NEIGHBORS and not _HAS_CUML:
        print("[WARN] POINTGRID_USE_GPU が指定されていますが、cuML が見つかりません。CPU (scikit-learn) を使用します。")
    else:
        print("[INFO] 近傍探索バックエンド: CPU (scikit-learn)")
    print()

    # ----------------------------------------
    # 1. 入力ファイル
    # ----------------------------------------
    in_path_str = input("入力ファイルパス（gpkg / parquet / csv）を入力してください: ").strip().strip('"')
    if not in_path_str:
        print("[ERROR] 入力ファイルパスが空です。")
        sys.exit(1)

    in_path = Path(in_path_str)
    if not in_path.exists():
        print(f"[ERROR] ファイルが見つかりません: {in_path}")
        sys.exit(1)

    in_type = detect_input_type(in_path)
    if in_type == "":
        print("[WARN] 拡張子から入力形式を判別できませんでした。")
        print("       .gpkg/.shp/.geojson/.parquet/.csv/.txt のいずれかを想定しています。")

    print("\n[INFO] 入力データを読み込み中...")

    # ----------------------------------------
    # 2. 入力読み込み＆GeoDataFrame化
    # ----------------------------------------
    if in_type == "gpkg":
        gdf = gpd.read_file(in_path)
        print(f"[INFO] GeoPackage 読み込み完了（{len(gdf)} レコード）。")
        if gdf.geometry.isna().all():
            print("[ERROR] GeoPackage 内に geometry が存在しないようです。")
            sys.exit(1)
        points_gdf = gdf
        df_for_cols = gdf.drop(columns=["geometry"], errors="ignore")
    else:
        # parquet / csv
        if in_type == "parquet":
            df = pd.read_parquet(in_path)
            print(f"[INFO] parquet 読み込み完了（{len(df)} レコード）。")
        elif in_type == "csv":
            df = pd.read_csv(in_path)
            print(f"[INFO] CSV 読み込み完了（{len(df)} レコード）。")
        else:
            # 念のため try
            try:
                df = pd.read_parquet(in_path)
                in_type = "parquet"
                print(f"[INFO] parquet として読み込みに成功（{len(df)} レコード）。")
            except Exception:
                df = pd.read_csv(in_path)
                in_type = "csv"
                print(f"[INFO] CSV として読み込みに成功（{len(df)} レコード）。")

        x_col = ask_column_name(df, "X 座標（東方向など）")
        y_col = ask_column_name(df, "Y 座標（北方向など）")
        class_col = ask_column_name(df, "地形クラス（ラベル）")

        geometry = [Point(xy) for xy in zip(df[x_col], df[y_col])]
        points_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=None)
        df_for_cols = df  # クラス列選択済み

    # gpkg 読み込み時はクラス列をここで選択
    if "class_col" not in locals():
        class_col = ask_column_name(df_for_cols, "地形クラス（ラベル）")

    # ----------------------------------------
    # 3. CRS 確認・設定
    # ----------------------------------------
    if points_gdf.crs is None:
        print("\n[INFO] 入力データに CRS（座標参照系）の情報がありません。")
        epsg_str = input("EPSG コードを入力してください（例: 6673, 6674 等）: ").strip()
        try:
            epsg_code = int(epsg_str)
        except Exception:
            print("[ERROR] EPSG コードが整数として解釈できません。")
            sys.exit(1)
        points_gdf.set_crs(epsg=epsg_code, inplace=True)
        print(f"[INFO] CRS を EPSG:{epsg_code} として設定しました。")
    else:
        print(f"\n[INFO] 入力データの CRS: {points_gdf.crs}")

    # ----------------------------------------
    # 4. グリッド解像度の指定と生成
    # ----------------------------------------
    bounds = points_gdf.total_bounds
    print("\n[INFO] データ範囲（外接矩形）:")
    print(f"  minx={bounds[0]:.3f}, miny={bounds[1]:.3f}, maxx={bounds[2]:.3f}, maxy={bounds[3]:.3f}")

    while True:
        cell_str = input("グリッドのセルサイズ（例: 1, 5, 10 など。座標系の単位）を入力してください: ").strip()
        try:
            cell_size = float(cell_str)
            if cell_size <= 0:
                raise ValueError
            break
        except Exception:
            print("[WARN] 正の数値を入力してください。")

    grid_gdf = build_fishnet(bounds, cell_size)
    grid_gdf.set_crs(points_gdf.crs, inplace=True)

    # ----------------------------------------
    # 5. クラス割り当て方式の選択（A/B/C）
    # ----------------------------------------
    print("\n[INFO] クラス割り当て方式を選択してください。")
    print("  1) A方式：最近傍（Nearest Neighbor）")
    print("  2) B方式：半径内多数決（Kernel Majority）")
    print("  3) C方式：半径内 距離減衰付き多数決（Weighted Majority）")

    method = None
    while method not in ("1", "2", "3"):
        method = input("方式番号を入力してください [1-3]: ").strip()

    if method == "1":
        # A方式
        grid_with_class = assign_classes_A_nearest(grid_gdf, points_gdf, class_col)

    elif method == "2":
        # B方式
        while True:
            r_str = input("半径（例: 3, 5, 10。座標系の単位）を入力してください: ").strip()
            try:
                radius = float(r_str)
                if radius <= 0:
                    raise ValueError
                break
            except Exception:
                print("[WARN] 正の数値を入力してください。")
        grid_with_class = assign_classes_B_radius_majority(
            grid_gdf, points_gdf, class_col, radius
        )

    else:
        # C方式
        while True:
            r_str = input("半径（例: 3, 5, 10。座標系の単位）を入力してください: ").strip()
            try:
                radius = float(r_str)
                if radius <= 0:
                    raise ValueError
                break
            except Exception:
                print("[WARN] 正の数値を入力してください。")

        # 減衰指数 power
        p_str = input("距離減衰の指数 power（空Enter=2.0）: ").strip()
        if p_str == "":
            power = 2.0
        else:
            try:
                power = float(p_str)
            except Exception:
                print("[WARN] 不正な値のため power=2.0 を使用します。")
                power = 2.0

        eps_str = input("eps（ゼロ距離回避用の微小値。空Enter=1e-6）: ").strip()
        if eps_str == "":
            eps = 1e-6
        else:
            try:
                eps = float(eps_str)
            except Exception:
                print("[WARN] 不正な値のため eps=1e-6 を使用します。")
                eps = 1e-6

        grid_with_class = assign_classes_C_weighted_majority(
            grid_gdf, points_gdf, class_col, radius, power, eps
        )

    # ----------------------------------------
    # 6. dissolve の有無
    # ----------------------------------------
    do_dissolve = ask_yes_no("\nクラスごとに dissolve（面マージ）したレイヤも作成しますか？ [Y/n]:", default="y")

    dissolved_gdf: Optional[gpd.GeoDataFrame] = None
    if do_dissolve:
        dissolved_gdf = dissolve_by_class(grid_with_class, class_col)

    # ----------------------------------------
    # 7. 出力ファイルパス
    # ----------------------------------------
    print("\n[INFO] 出力ファイルを指定してください。")

    mode_label = {"1": "A", "2": "B", "3": "C"}[method]
    base_name = in_path.with_suffix("").name + f"_grid_{mode_label}_{int(cell_size)}"
    default_dir = in_path.parent
    default_path = default_dir / (base_name + ".gpkg")

    out_path_str = input(f"出力ファイルパス（空= {default_path} ）: ").strip().strip('"')
    if out_path_str == "":
        out_path = default_path
    else:
        out_path = Path(out_path_str)

    if do_dissolve:
        b = out_path.with_suffix("")
        ext = out_path.suffix or ".gpkg"
        out_grid = b.with_name(b.name + "_grid").with_suffix(ext)
        out_diss = b.with_name(b.name + "_dissolve").with_suffix(ext)

        save_geodataframe(grid_with_class, out_grid)
        save_geodataframe(dissolved_gdf, out_diss)  # type: ignore[arg-type]

        print("\n[INFO] 出力ファイル:")
        print(f"  グリッド版:   {out_grid}")
        print(f"  dissolve版: {out_diss}")
    else:
        save_geodataframe(grid_with_class, out_path)
        print(f"\n[INFO] 出力ファイル: {out_path}")

    print("\n[完了] 点の地形クラスからグリッド面データを生成しました。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] ユーザーにより中断されました。")
