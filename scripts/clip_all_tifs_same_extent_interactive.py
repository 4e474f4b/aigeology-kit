#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ディレクトリ内の GeoTIFF(.tif) を、共通のクリップ範囲で一括トリムするスクリプト。

■ 想定する用途
  - DEM / 地形特徴量など多数の TIF を、
    ・同じ外接矩形（BBox）で揃えたい
    ・GPKG のポリゴン範囲だけ切り出したい
  といった場合の一括処理。

■ 入力
  - 対象ディレクトリ:
      任意のフォルダ（サブフォルダを再帰的に含めるか選択可）
      → 拡張子 .tif のファイルを全て対象とする。

  - クリップ範囲 (EPSG:XXXX 座標系で指定):
      1) 手動入力: xmin, ymin, xmax, ymax を数値入力
      2) ポリゴン GPKG:
         - GPKG パスを指定
         - レイヤを選択
         - (geopandas 利用可能な場合) 属性一覧を表示し、
           列と値（複数可）でフィーチャを絞り込んでから bbox/ポリゴンを生成
         - 属性フィルタしない場合はレイヤ全体を使用

■ 出力
  - デフォルト出力先:
      <入力ディレクトリ>/clipped_out/
  - ファイル名:
      Clipped-<元のファイル名>.tif
  - GeoTIFF (TILED=YES, COMPRESS=DEFLATE, PREDICTOR=3, BIGTIFF=IF_SAFER)

■ クリップ方法（対話で選択）
  - A) BBoxで切り出し（矩形トリム。高速）
      ・指定 BBox を各 TIF の CRS に transform し、
        交差する部分だけ window 読み出し
      ・完全に範囲外の TIF はスキップ

  - B) ポリゴン形状でマスク（外側を NoData。精度重視）
      ・まず BBox で概略の範囲を切り出し
      ・その後、ポリゴンでマスク（外側ピクセルを NoData）
      ・crop オプション ON の場合はマスクの外接矩形にトリム

■ 事前クリーン機能（任意）
  - gdal_translate が利用可能な場合、
    各 TIF を一度「きれいな GeoTIFF」に変換してからクリップ:
      - メタデータ / RAT / GCP を削除
      - TILED=YES / COMPRESS=DEFLATE / PREDICTOR=2 等で再出力
    → 元 TIF に難ありな場合の安定性向上を想定。

■ 必要ライブラリ
  - 必須: rasterio, shapely
  - 任意: geopandas, fiona
    ・インストールされていれば GPKG 属性によるフィルタが有効
  - 任意: gdal_translate, ogr2ogr, ogrinfo
    ・事前クリーンおよび、GPKG bbox / 形状取得で使用

■ 使い方（例）
  1) 対話式で起動:
       python clip_all_tifs_same_extent_interactive_v3_3.py

  2) 対話の流れ:
       - 入力ディレクトリを指定
       - 再帰処理の有無を選択
       - 出力フォルダを指定（空 Enter でデフォルト）
       - 事前クリーンを行うか選択
       - クリップ範囲 EPSG を指定（例: 6674）
       - クリップ範囲の取得方法 (1: 手入力 / 2: GPKG)
           * 2 の場合:
               - GPKG パスを指定
               - レイヤを選択
               - 属性一覧を確認し、必要なら列＆値でフィルタ
       - クリップ方法を選択 (A: BBox / B: ポリゴンマスク)
       - マスク時に crop するかどうかを選択
       → あとは全 TIF に対して自動処理

■ 制約・注意点
  - 入力 TIF の CRS が未設定の場合、そのファイルはスキップ。
  - GPKG 側レイヤの CRS が不明な場合、クリップ EPSG とみなして処理。
  - 非常に大きな GPKG を属性フィルタする場合、
    GeoPandas での読込に時間とメモリを要する可能性あり。
"""

import os, sys, json, tempfile, subprocess, warnings, contextlib
from pathlib import Path
from typing import Tuple, Optional, List

# --- Windows 日本語環境での GDAL ログによる UnicodeDecodeError 回避 ---
if os.name == "nt":
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("GDAL_FILENAME_IS_UTF8", "YES")
    os.environ.setdefault("CPL_LOG", "NUL")   # Windowsは NUL
    os.environ.setdefault("CPL_DEBUG", "OFF")
    os.environ.setdefault("CPL_ERROR_HANDLER", "CPLQuietErrorHandler")
# ----------------------------------------------------------------------

import rasterio
from rasterio.env import Env
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from rasterio.coords import BoundingBox
from rasterio.errors import RasterioIOError
from rasterio.mask import mask as rio_mask

from shapely.geometry import mapping

# optional
try:
    import geopandas as gpd
    import fiona
    HAVE_GPD = True
except Exception:
    gpd = None
    fiona = None
    HAVE_GPD = False

# ---------- utils ----------
def sanitize_path(s: str) -> str:
    if not s: return s
    s = s.strip().strip('"').strip("'")
    if len(s) % 2 == 0 and s[:len(s)//2] == s[len(s)//2:]:
        s = s[:len(s)//2]
    return os.path.normpath(os.path.expandvars(s))

def ask_path(prompt: str, must_exist=False, default="", allow_empty=False) -> str:
    while True:
        raw = input(f'{prompt}{" ["+default+"]" if default else ""}: ')
        s = sanitize_path(raw or default)
        if not s and allow_empty: return ""
        if must_exist and s and not os.path.exists(s):
            print(f"  ❌ パスが存在しません: {s}"); continue
        return s

def ask_yesno(prompt: str, default_yes=True) -> bool:
    d = "Y/n" if default_yes else "y/N"
    s = (input(f"{prompt} [{d}]: ").strip() or ("y" if default_yes else "n")).lower()
    return s.startswith("y")

def ask_int(prompt: str, default: Optional[int]=None) -> int:
    while True:
        s = input(f'{prompt}{(" ["+str(default)+"]") if default is not None else ""}: ').strip()
        if not s and default is not None: return default
        try: return int(s)
        except ValueError: print("  ❌ 整数で入力してください。")

def ask_float(prompt: str) -> float:
    while True:
        s = input(f"{prompt}: ").strip()
        try: return float(s)
        except ValueError: print("  ❌ 数値で入力してください。")

def collect_tifs(root: str, recursive: bool) -> List[str]:
    p = Path(root)
    return [str(fp) for fp in (p.rglob("*.tif") if recursive else p.glob("*.tif"))]

def have_cmd(cmd: str) -> bool:
    try:
        subprocess.run([cmd, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

# ---------- GPKG helpers ----------
def list_layers_gpkg(path: str) -> List[str]:
    if fiona is not None:
        try:
            return list(fiona.listlayers(path))
        except Exception:
            pass
    if not have_cmd("ogrinfo"):
        return []
    out = subprocess.run(["ogrinfo", path], capture_output=True, text=True)
    layers = []
    for ln in out.stdout.splitlines():
        ln = ln.strip()
        if ":" in ln and ln.split(":")[0].strip().isdigit():
            layers.append(ln.split(":")[1].strip().split(" ")[0])
    return layers

def get_layer_feature_count(path: str, layer: str) -> Optional[int]:
    if fiona is not None:
        try:
            with fiona.open(path, layer=layer) as src:
                return len(src)
        except Exception:
            pass
    if have_cmd("ogrinfo"):
        out = subprocess.run(["ogrinfo", "-so", path, layer], capture_output=True, text=True)
        for ln in out.stdout.splitlines():
            if "Feature Count:" in ln:
                try:
                    return int(ln.split("Feature Count:")[1].strip())
                except Exception:
                    return None
    return None

def choose_layer(path: str) -> Optional[str]:
    layers = list_layers_gpkg(path)
    if not layers:
        print("  ❌ GPKGにレイヤーが見つかりません。"); return None
    nonempty = []
    for L in layers:
        cnt = get_layer_feature_count(path, L)
        if cnt and cnt > 0:
            nonempty.append((L, cnt))
    if nonempty:
        print("[INFO] 非空レイヤー候補:", ", ".join([f"{L}({c})" for L,c in nonempty]))
        if ask_yesno(f"上記の最初のレイヤー '{nonempty[0][0]}' を使いますか？", True):
            return nonempty[0][0]
    print("利用可能レイヤー：")
    for i,L in enumerate(layers,1):
        print(f"  [{i}] {L}  (features={get_layer_feature_count(path, L)})")
    while True:
        s = input("番号を選んでください: ").strip()
        if s.isdigit() and 1 <= int(s) <= len(layers):
            return layers[int(s)-1]
        print("  ❌ 範囲外です。")

def bbox_from_layer(path: str, layer: str, epsg: int) -> Tuple[float,float,float,float]:
    if HAVE_GPD:
        try:
            gdf = gpd.read_file(path, layer=layer)
            if gdf.empty: raise RuntimeError("選択レイヤーが空です。")
            if gdf.crs is None: gdf = gdf.set_crs(epsg=epsg)
            if gdf.crs.to_epsg() != epsg: gdf = gdf.to_crs(epsg=epsg)
            minx, miny, maxx, maxy = gdf.total_bounds
            return float(minx), float(miny), float(maxx), float(maxy)
        except Exception:
            pass
    if not (have_cmd("ogr2ogr") and have_cmd("ogrinfo")):
        raise RuntimeError("GDALコマンドが見つからず、bbox取得に失敗。")
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "reproj.gpkg")
        subprocess.run(["ogr2ogr", "-t_srs", f"EPSG:{epsg}", "-f", "GPKG", tmp, path, layer], check=True)
        out = subprocess.run(["ogrinfo", "-al", "-so", tmp], capture_output=True, text=True, check=True)
        for ln in out.stdout.splitlines():
            if ln.strip().startswith("Extent:"):
                part = ln.split("Extent:")[1].strip()
                L, R = part.split(" - ")
                minx = float(L[L.find("(")+1:L.find(",")])
                miny = float(L[L.find(",")+1:L.find(")")])
                maxx = float(R[R.find("(")+1:R.find(",")])
                maxy = float(R[R.find(",")+1:R.find(")")])
                return (minx, miny, maxx, maxy)
    raise RuntimeError("Extent の抽出に失敗。")


def select_features_by_attribute(path: str, layer: str, epsg: int):
    """
    GPKGレイヤの属性でフィーチャを絞り込む（geopandas 必須）。
    - 戻り値: 絞り込み後の GeoDataFrame（CRS=EPSG:epsg） or None（失敗/未使用時）
    """
    if not HAVE_GPD:
        print("  ⚠️ geopandas が無いため属性での絞り込みはスキップします。")
        return None
    try:
        gdf = gpd.read_file(path, layer=layer)
    except Exception as e:
        print(f"  ⚠️ GPKG読み込みに失敗したため、属性絞り込みをスキップします: {e}")
        return None

    if gdf.empty:
        print("  ⚠️ レイヤが空です。属性絞り込みはスキップします。")
        return None

    # 座標系をEPSGに合わせる
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=epsg)
    if gdf.crs.to_epsg() != epsg:
        gdf = gdf.to_crs(epsg=epsg)

    geom_col = gdf.geometry.name
    attr_cols = [c for c in gdf.columns if c != geom_col]

    if not attr_cols:
        print("[INFO] 属性列が無いため、レイヤ内の全フィーチャを使用します。")
        return gdf

    print("\n[属性列一覧]")
    for i, c in enumerate(attr_cols, 1):
        try:
            nunique = gdf[c].nunique(dropna=True)
        except Exception:
            nunique = "?"
        print(f"  [{i}] {c}  (一意値 ≒ {nunique} 個)")

    # 列の選択（空Enterならフィルタ無し）
    while True:
        s = input("絞り込みに使う列番号を入力（空Enter=絞り込みなし）: ").strip()
        if not s:
            print("[INFO] 属性絞り込みは行わず、レイヤ全体を使用します。")
            return gdf
        if s.isdigit() and 1 <= int(s) <= len(attr_cols):
            col = attr_cols[int(s) - 1]
            break
        print("  ❌ 範囲外です。")


    # 一意値の一覧
    uniques = gdf[col].dropna().unique().tolist()
    try:
        uniques_sorted = sorted(uniques)
    except Exception:
        uniques_sorted = uniques

    print(f"\n列 '{col}' の一意な属性値（最大50件を表示）:")
    for v in uniques_sorted[:50]:
        print(f"  - {v}")
    if len(uniques_sorted) > 50:
        print(f"  ... ほか {len(uniques_sorted) - 50} 件")

    raw = input("絞り込みに使う値をカンマ区切りで入力（空Enter=その列の全フィーチャを使用）: ").strip()
    if not raw:
        print(f"[INFO] 値未指定のため、列 '{col}' での絞り込みは行わず、レイヤ全体を使用します。")
        return gdf

    wanted = [x.strip() for x in raw.split(",") if x.strip()]
    # 文字列比較に揃える（タイプ混在対策）
    series_str = gdf[col].astype(str)
    wanted_str = [str(v) for v in wanted]
    filtered = gdf[series_str.isin(wanted_str)].copy()

    if filtered.empty:
        print("  ⚠️ 絞り込み後のフィーチャが 0 件です。レイヤ全体を使用します。")
        return gdf

    print(f"[INFO] 絞り込み後フィーチャ数: {len(filtered)}")
    return filtered


def load_shapes_as_geojson_dicts(path: str, layer: str, epsg: int) -> List[dict]:
    if HAVE_GPD:
        try:
            gdf = gpd.read_file(path, layer=layer)
            if gdf.empty: raise RuntimeError("選択レイヤーが空です。")
            if gdf.crs is None: gdf = gdf.set_crs(epsg=epsg)
            if gdf.crs.to_epsg() != epsg: gdf = gdf.to_crs(epsg=epsg)
            geoms = []
            for geom in gdf.geometry:
                if geom is None or geom.is_empty: continue
                geoms.append(mapping(geom))
            if geoms: return geoms
        except Exception:
            pass
    if not have_cmd("ogr2ogr"):
        raise RuntimeError("ogr2ogr が見つからず、形状取得に失敗。")
    with tempfile.TemporaryDirectory() as td:
        out_json = os.path.join(td, "geom.json")
        subprocess.run(["ogr2ogr", "-t_srs", f"EPSG:{epsg}", "-f", "GeoJSON", out_json, path, layer], check=True)
        gj = json.load(open(out_json, "r", encoding="utf-8"))
        geoms = [f.get("geometry") for f in gj.get("features", []) if f.get("geometry")]
        if not geoms:
            raise RuntimeError("有効なジオメトリが得られませんでした。")
        return geoms

# ---------- geo helpers ----------
def clamp_intersection(a: BoundingBox, b: BoundingBox) -> Optional[BoundingBox]:
    xmin = max(a.left, b.left); ymin = max(a.bottom, b.bottom)
    xmax = min(a.right, b.right); ymax = min(a.top, b.top)
    if (xmax <= xmin) or (ymax <= ymin): return None
    return BoundingBox(left=xmin, bottom=ymin, right=xmax, top=ymax)


def choose_blocksize(width: int, height: int, max_block: int = 512) -> Tuple[Optional[int], Optional[int]]:
    """
    GTiff ドライバの制約:
      - タイルサイズは 16 の倍数 かつ 画像サイズ以下
    これを満たす blockxsize / blockysize を決める。
    どちらか一方でも 16 未満になってしまう場合は None を返し、
    （＝タイル化しない／strip に任せる方向）
    """
    if width <= 0 or height <= 0:
        return None, None

    # 16px 未満しかない極小画像はタイル化しない
    if width < 16 or height < 16:
        return None, None

    bx = min(max_block, width)
    by = min(max_block, height)

    # 16 の倍数に丸める（下方向）
    bx = (bx // 16) * 16
    by = (by // 16) * 16

    if bx < 16 or by < 16:
        return None, None

    return bx, by

# ---------- clean helpers ----------
def clean_tif_gdal(src: str, dst: str) -> None:
    """元TIFを 'きれいな' GTiff に再出力（RAT/GCPを除去）"""
    cmd = [
        "gdal_translate",
        "-of", "GTiff",
        "-co", "TILED=YES",
        "-co", "COMPRESS=DEFLATE",
        "-co", "PREDICTOR=2",
        "-co", "BIGTIFF=IF_SAFER",
        # "-nomd",   # ★ GDAL 3.10.3 では未サポートのため削除
        "-norat",  # RAT非コピー
        "-nogcp",  # GCP非コピー
        src, dst
    ]
    # 日本語環境でのログ抑制（ログは捨てる）
    env = os.environ.copy()
    env["CPL_DEBUG"] = "OFF"
    env["CPL_LOG"] = "NUL"
    env["CPL_ERROR_HANDLER"] = "CPLQuietErrorHandler"

    subprocess.run(
        cmd,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True
    )


# ---------- main ----------
def main():
    print("\n=== ディレクトリ内の TIF を同一範囲でクリップ（EPSG指定 / BBox or ポリゴンマスク / レイヤー選択対応 / 事前クリーン） ===")

    root_dir = ask_path("一括クリップする画像格納ディレクトリのパス", must_exist=True)
    recursive = ask_yesno("サブフォルダも再帰的に処理しますか？", True)

    out_dir_default = os.path.join(root_dir, "clipped_out")
    out_dir = ask_path("出力先フォルダ（空=既定）", default=out_dir_default, allow_empty=True) or out_dir_default
    os.makedirs(out_dir, exist_ok=True)

    # 事前クリーンを行うか？
    preclean = ask_yesno("処理前に入力TIFをクリーン再出力してからクリップしますか？（推奨）", True)
    if preclean and not have_cmd("gdal_translate"):
        print("  ⚠️ gdal_translate が見つからないため、事前クリーンはスキップします。")
        preclean = False

    epsg = ask_int("クリップ範囲の EPSG コード（例: 6674/6673/2447 など）", default=6674)
    print(f"[INFO] クリップ範囲の座標系: EPSG:{epsg}")

    print("\nクリップ範囲の取得方法：")
    print("  1) 手動で数値入力（xmin, ymin, xmax, ymax）")
    print("  2) ポリゴン GPKG を読み込み（レイヤー選択）")
    m_src = (input("番号を選んでください [1/2]: ").strip() or "1")
    if m_src not in ("1","2"):
        print("❌ 不正な選択。"); return

    if m_src == "1":
        print(f">> EPSG:{epsg} の座標で入力してください。")
        xmin = ask_float("xmin"); ymin = ask_float("ymin")
        xmax = ask_float("xmax"); ymax = ask_float("ymax")
        if xmax <= xmin or ymax <= ymin:
            print("❌ bbox が不正です。"); return
        clip_bbox_src = (xmin, ymin, xmax, ymax)
        shapes_geojson = [{
            "type":"Polygon",
            "coordinates":[[(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax),(xmin,ymin)]]
        }]

    else:
        gpkg_path = ask_path("ポリゴン GPKG のパス", must_exist=True)
        layer = choose_layer(gpkg_path)
        if not layer:
            print("❌ レイヤー選択に失敗。"); return
        print(f"[INFO] 使用レイヤー: {layer}")

        # 属性での絞り込み（geopandas がある場合のみ）
        filtered_gdf = None
        if HAVE_GPD:
            filtered_gdf = select_features_by_attribute(gpkg_path, layer, epsg)
        else:
            print("  ⚠️ geopandas が無いので、レイヤ全体を使用します。")

        if filtered_gdf is not None:
            # 絞り込まれたフィーチャだけで bbox / shapes を作成
            minx, miny, maxx, maxy = filtered_gdf.total_bounds
            clip_bbox_src = (float(minx), float(miny), float(maxx), float(maxy))
            shapes_geojson = [
                mapping(geom)
                for geom in filtered_gdf.geometry
                if geom is not None and not geom.is_empty
            ]
        else:
            # 従来どおりレイヤ全体を使用
            clip_bbox_src = bbox_from_layer(gpkg_path, layer, epsg)
            shapes_geojson = load_shapes_as_geojson_dicts(gpkg_path, layer, epsg)

        print(f"[INFO] BBox (EPSG:{epsg}): {clip_bbox_src}")

    print("\nクリップ方法：")
    print("  A) BBoxで切り出し（矩形トリム。早い）")
    print("  B) ポリゴン形状でマスク（外側をNoData。正確）")
    m_clip = (input("方法を選んでください [A/B]: ").strip() or "A").upper()
    if m_clip not in ("A","B"):
        print("❌ 不正な選択。"); return

    crop_when_mask = False
    if m_clip == "B":
        crop_when_mask = ask_yesno("マスク外接矩形にトリム（crop）しますか？", True)

    tifs = collect_tifs(root_dir, recursive)
    if not tifs:
        print("❌ TIFが見つかりません。"); return

    total, processed, skipped = len(tifs), 0, 0
    print(f"[INFO] 対象 TIF: {total} 枚")

    # 事前クリーンの出力先
    clean_cache = os.path.join(out_dir, "_clean_cache")
    if preclean:
        os.makedirs(clean_cache, exist_ok=True)

    # 標準エラーの表示も抑止（ログがうるさい環境対策）
    stderr_null = open(os.devnull, "w", encoding="utf-8", errors="ignore")
    with contextlib.redirect_stderr(stderr_null):

        for tif in tifs:
            src_path = tif

            # --- 事前クリーン（必要に応じて差分実行） ---
            if preclean:
                dst_clean = os.path.join(clean_cache, f"clean-{Path(tif).name}")
                need = (not os.path.exists(dst_clean)) or (os.path.getmtime(dst_clean) < os.path.getmtime(tif))
                try:
                    if need:
                        clean_tif_gdal(tif, dst_clean)
                    src_path = dst_clean
                except subprocess.CalledProcessError as e:
                    print(f"  ⚠️ クリーン化に失敗: {tif}（gdal_translate エラー）。通常処理を継続します。")

            with Env(CPL_DEBUG="OFF", CPL_LOG="NUL", CPL_ERROR_HANDLER="CPLQuietErrorHandler"):
                try:
                    with rasterio.open(src_path) as src:
                        if src.crs is None:
                            print(f"  ⚠️  CRS未設定スキップ: {src_path}")
                            skipped += 1
                            continue

                        out_path = os.path.join(out_dir, f"Clipped-{Path(tif).name}")

                        if m_clip == "A":
                            tb = transform_bounds(f"EPSG:{epsg}", src.crs, *clip_bbox_src, densify_pts=21)
                            bbox_target = BoundingBox(left=tb[0], bottom=tb[1], right=tb[2], top=tb[3])
                            inter = clamp_intersection(bbox_target, src.bounds)
                            if inter is None:
                                print(f"  ℹ️  範囲外スキップ: {tif}")
                                skipped += 1
                                continue
                            win = from_bounds(inter.left, inter.bottom, inter.right, inter.top, src.transform)
                            data = src.read(window=win, boundless=True, fill_value=src.nodata)
                            transform = src.window_transform(win)

                        else:
                            tb = transform_bounds(f"EPSG:{epsg}", src.crs, *clip_bbox_src, densify_pts=21)
                            bbox_target = BoundingBox(left=tb[0], bottom=tb[1], right=tb[2], top=tb[3])
                            inter = clamp_intersection(bbox_target, src.bounds)
                            if inter is None:
                                print(f"  ℹ️  マスク範囲外: {tif}")
                                skipped += 1
                                continue
                            win = from_bounds(inter.left, inter.bottom, inter.right, inter.top, src.transform)
                            tmp_data = src.read(window=win, boundless=True, fill_value=src.nodata)
                            tmp_transform = src.window_transform(win)
                            tmp_profile = src.profile.copy()
                            tmp_profile.update({"height": tmp_data.shape[1],
                                                "width": tmp_data.shape[2],
                                                "transform": tmp_transform})

                            with rasterio.io.MemoryFile() as memfile:
                                with memfile.open(**tmp_profile) as mem:
                                    mem.write(tmp_data)

                                    with tempfile.TemporaryDirectory() as td:
                                        in_json = os.path.join(td, "in.json")
                                        out_json = os.path.join(td, "out.json")
                                        gj = {
                                            "type": "FeatureCollection",
                                            "features": [{"type":"Feature","geometry":g,"properties":{}} for g in shapes_geojson]
                                        }
                                        with open(in_json, "w", encoding="utf-8") as f:
                                            json.dump(gj, f)

                                        dst_epsg = mem.crs.to_epsg()
                                        if not dst_epsg:
                                            data, transform = tmp_data, tmp_transform
                                        else:
                                            subprocess.run(
                                                ["ogr2ogr","-t_srs",f"EPSG:{dst_epsg}","-f","GeoJSON",out_json,in_json],
                                                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                                            )
                                            gj2 = json.load(open(out_json, "r", encoding="utf-8"))
                                            geoms_dst = [f.get("geometry") for f in gj2.get("features", []) if f.get("geometry")]
                                            data, transform = rio_mask(mem, shapes=geoms_dst, crop=crop_when_mask, nodata=mem.nodata)

                        profile = src.profile.copy()
                        # 出力サイズに合わせてタイルサイズを決定
                        out_height = data.shape[1]
                        out_width  = data.shape[2]
                        bx, by = choose_blocksize(out_width, out_height)

                        profile.update({
                            "height": out_height,
                            "width":  out_width,
                            "transform": transform,
                            "compress": "DEFLATE",
                            "predictor": 3,
                            "BIGTIFF": "IF_SAFER",
                        })

                        # 16 の倍数ブロックが取れるときだけタイル化する
                        if bx is not None and by is not None:
                            profile.update({
                                "tiled": True,
                                "blockxsize": bx,
                                "blockysize": by,
                            })
                        else:
                            # タイル指定があるとエラーになるので削除
                            profile.pop("tiled", None)
                            profile.pop("blockxsize", None)
                            profile.pop("blockysize", None)
                        with rasterio.open(out_path, "w", **profile) as dst:
                            dst.write(data)

                        print(f"  ✅ 出力: {out_path}")
                        processed += 1

                except RasterioIOError as e:
                    print(f"  ⚠️  読み込み失敗: {src_path} ({e})")
                    skipped += 1
                except subprocess.CalledProcessError as e:
                    print(f"  ⚠️  GDALコマンド失敗: {src_path} ({e})")
                    skipped += 1
                except Exception as e:
                    print(f"  ⚠️  予期せぬエラー: {src_path} ({e})")
                    skipped += 1

    print("\n=== 完了 ===")
    print(f"  クリップ成功: {processed}")
    print(f"  スキップ   : {skipped}")
    print(f"  出力先     : {out_dir}")

if __name__ == "__main__":
    warnings.filterwarnings("once", category=UserWarning)
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断しました。"); sys.exit(1)
