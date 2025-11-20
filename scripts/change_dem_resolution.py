#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEM / 地形特徴量 画像の解像度変更スクリプト（対話式）

【概要】
- 入力フォルダ内の GeoTIFF やワールドファイル付き画像をまとめて読み込み、
  指定したピクセル解像度（地図単位：m など）にアップ／ダウンサンプリングして出力します。
- 出力は、入力フォルダ直下に「res_XXXm」というサブディレクトリを作成して保存します。
  （例：1m → 5m に変換する場合、res_5m/ 以下に出力）
- 複数の解像度（例: 2,3,5,10）をカンマ区切りで指定すると、
  それぞれの解像度のサブディレクトリにまとめて出力します。

【前提・方針】
- DEM や傾斜量図、開度図など、連続量のラスタを想定しています。
- 解像度変更には rasterio + GDAL の Reproject 機能を利用します。
- CRS（座標系）は変更せず、元画像と同じ CRS のまま、ピクセルサイズのみ変更します。
- 「実際に存在しないデータ」を新たに作らないため、
    - 元ラスタの bounds（外接矩形）を基本的に維持
    - 元データの外側は nodata のまま（補間・外挿はしない）
  という方針で実装しています。
- フォルダ内のラスタは「ほぼ同じ解像度」であることを前提とし、
  最初の1枚の解像度とターゲット解像度の大小関係から、
  各ターゲット解像度ごとに「アップサンプリング」か「ダウンサンプリング」かを判定します。

【地形解析向けの典型的なリサンプリング手法】
- ダウンサンプリング（解像度を粗くする、例: 1m → 5m）
    1) average : ピクセル平均（標高や傾斜など連続量の代表値）
    2) max     : 最大値（ピーク検出や極大地形の強調）
    3) min     : 最小値（谷底の高さなどに着目する場合）
- アップサンプリング（解像度を細かくする、例: 5m → 1m）
    1) bilinear : 2次元線形補間（DEM など連続量で一般的）
    2) cubic    : 三次畳み込み（より滑らかな補間）
    3) nearest  : 最近傍（カテゴリラスタや、値を変えたくない場合用）

【圧縮（すべて可逆圧縮）】
- none    : 無圧縮
- deflate : 一般的な可逆圧縮（デフォルト）
- lzw     : LZW 可逆圧縮
- zstd    : Zstandard 可逆圧縮（GDAL の対応バージョンのみ）

【必要ライブラリ（例）】
  mamba activate terrain-env  # もしくは aigeology-env など
  mamba install -c conda-forge rasterio

【処理の流れ】
  1. 入力フォルダのパスを対話式で取得
  2. 対象とする拡張子（.tif, .tiff など）を指定（未指定ならデフォルト）
  3. 最初の 1 枚から EPSG / 単位 / 元解像度を確認して表示
  4. 出力するピクセル解像度を、CRS の「線単位」と同じ単位で1つ以上指定（例：2,3,5,10）
  5. ターゲット解像度群のうち、up / down / same のどれが含まれるかを判定し、
     up 用・down 用の代表的なリサンプリング手法をそれぞれ 1つ選択
  6. 可逆圧縮方式（none / deflate / lzw / zstd）を選択
  7. サマリを表示して実行確認
  8. 各解像度 × 各ラスタの組み合わせについてリサンプリングして出力
"""

import sys
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.transform import Affine


# =========================
# ユーティリティ関数群
# =========================

def ask_directory_path(prompt: str) -> Path:
    """ディレクトリパスを対話式で取得する。存在チェック付き。"""
    while True:
        path_str = input(prompt).strip().strip('"').strip("'")
        if not path_str:
            print("  ※ パスが空です。もう一度入力してください。")
            continue

        p = Path(path_str)
        if p.is_dir():
            return p
        elif p.is_file():
            # ファイルが指定された場合は、その親ディレクトリを処理対象とする
            print("  ※ ファイルパスが指定されたため、その親ディレクトリを処理対象とします。")
            return p.parent
        else:
            print(f"  ※ パスが存在しません: {p}")
            continue


def ask_extensions() -> list[str]:
    """対象とする拡張子を取得（カンマ区切り）。空ならデフォルトを返す。"""
    print("\n[対象ファイルの拡張子を指定してください]")
    print("例: tif,tiff,img    （大文字小文字は自動無視）")
    print("空 Enter の場合は、次の拡張子が対象になります: tif,tiff,img,png,jpg,jpeg")
    s = input("拡張子（カンマ区切り、ドットなし）: ").strip()
    if not s:
        return [".tif", ".tiff", ".img", ".png", ".jpg", ".jpeg"]

    exts = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.startswith("."):
            part = "." + part
        exts.append(part.lower())
    return exts


def find_raster_files(root: Path, exts: list[str]) -> list[Path]:
    """指定フォルダ直下のラスタファイル一覧を取得（再帰なし）。"""
    files = []
    for f in sorted(root.iterdir()):
        if f.is_file() and f.suffix.lower() in exts:
            files.append(f)
    return files


def show_crs_and_resolution(sample_path: Path) -> tuple[float, float, str, int | None]:
    """
    最初の 1 枚から CRS / EPSG / 単位 / 元ピクセルサイズを確認して表示する。

    戻り値: (px_size_x, px_size_y, unit_name, epsg)
    """
    with rasterio.open(sample_path) as src:
        crs = src.crs
        transform = src.transform

        px_size_x = transform.a
        px_size_y = -transform.e  # 通常は負なので符号反転しておく

        if crs is None:
            epsg = None
            unit_name = "(不明)"
        else:
            try:
                epsg = crs.to_epsg()
            except Exception:
                epsg = None

            try:
                unit_name = crs.linear_units
            except Exception:
                unit_name = "(不明)"

        print("\n[入力ラスタの CRS / 解像度情報（サンプル 1 枚目）]")
        print(f"  ファイル        : {sample_path.name}")
        print(f"  EPSG            : {epsg if epsg is not None else '不明'}")
        print(f"  線単位 (CRS)    : {unit_name}")
        print(f"  元ピクセルサイズ: {abs(px_size_x):.6f} × {abs(px_size_y):.6f} {unit_name}")

        if isinstance(unit_name, str) and "degree" in unit_name.lower():
            print("\n  !!! 注意 !!!")
            print("  このラスタは緯度経度（degree）単位の CRS のようです。")
            print("  出力解像度の指定も『度』単位になってしまうため、")
            print("  一般的には、事前にメートル単位の平面直角座標系（例: EPSG:6673 など）に")
            print("  再投影してからこのスクリプトを使うことを推奨します。")

        return abs(px_size_x), abs(px_size_y), unit_name, epsg


def ask_resolutions(px_ref: float, unit_name: str) -> list[float]:
    """
    出力ピクセル解像度を 1つ以上、カンマ区切りで取得する。
    例: 2,3,5,10
    """
    print("\n[出力ピクセル解像度の指定]")
    print("  ※ 単位は、上で表示した CRS の『線単位』と同じです。")
    print("     例：EPSG:6673 などの平面直角座標系であればメートル (metre)")
    print("         緯度経度 (degree) の場合は『度』単位になります。")
    print(f"  参考: 元ピクセルサイズ ≒ {px_ref:.6f} {unit_name}")
    print("  複数指定例: 2,3,5,10")

    while True:
        s = input("出力ピクセル解像度（カンマ区切り、例: 2,3,5,10）: ").strip()
        if not s:
            print("  ※ 少なくとも 1 つは指定してください。")
            continue

        parts = [p.strip() for p in s.split(",") if p.strip()]
        if not parts:
            print("  ※ 少なくとも 1 つは指定してください。")
            continue

        values: list[float] = []
        ok = True
        for p in parts:
            try:
                v = float(p)
            except ValueError:
                print(f"  ※ '{p}' を数値として解釈できません。もう一度入力してください。")
                ok = False
                break
            if v <= 0:
                print(f"  ※ 出力解像度は正の値のみ指定可能です（{p} は不正）。")
                ok = False
                break
            values.append(v)

        if not ok:
            continue

        # 重複を除去しつつ安定した順序に
        unique_values = []
        seen = set()
        for v in values:
            if v not in seen:
                seen.add(v)
                unique_values.append(v)

        print(f"  -> 指定された解像度: {', '.join(str(v) for v in unique_values)} {unit_name}")
        return unique_values


def ask_resampling_method_down() -> Resampling:
    """ダウンサンプリング向けの代表的な手法から選択する。"""
    options = {
        "1": ("average", Resampling.average),
        "2": ("max", Resampling.max),
        "3": ("min", Resampling.min),
    }
    print("\n[ダウンサンプリング用リサンプリング方法を選択してください]")
    print("  ※ 解像度を粗くする処理（例: 1m → 5m）")
    print("  1) average : ピクセル平均（標高・傾斜など連続量の代表値）")
    print("  2) max     : 最大値（ピーク・凸地形の強調など）")
    print("  3) min     : 最小値（谷底など凹地形に着目する場合）")

    while True:
        choice = input("番号を選んでください [1-3] (デフォルト=1): ").strip()
        if not choice:
            return options["1"][1]
        if choice in options:
            return options[choice][1]
        print("  ※ 1〜3 の番号で選択してください。")


def ask_resampling_method_up() -> Resampling:
    """アップサンプリング向けの代表的な手法から選択する。"""
    options = {
        "1": ("bilinear", Resampling.bilinear),
        "2": ("cubic", Resampling.cubic),
        "3": ("nearest", Resampling.nearest),
    }
    print("\n[アップサンプリング用リサンプリング方法を選択してください]")
    print("  ※ 解像度を細かくする処理（例: 5m → 1m）")
    print("  1) bilinear : 2次元線形補間（DEM など連続量で一般的）")
    print("  2) cubic    : 三次畳み込み（より滑らかに補間したい場合）")
    print("  3) nearest  : 最近傍（カテゴリラスタ・ラベルなど）")

    while True:
        choice = input("番号を選んでください [1-3] (デフォルト=1): ").strip()
        if not choice:
            return options["1"][1]
        if choice in options:
            return options[choice][1]
        print("  ※ 1〜3 の番号で選択してください。")


def ask_compression() -> str | None:
    """
    可逆圧縮方式を選択する。
    戻り値:
      - None  : 無圧縮
      - "deflate" / "lzw" / "zstd" など
    """
    options = {
        "0": ("none", None),
        "1": ("deflate", "deflate"),
        "2": ("lzw", "lzw"),
        "3": ("zstd", "zstd"),
    }
    print("\n[出力 GeoTIFF の圧縮方式（すべて可逆圧縮）]")
    print("  0) none    : 無圧縮")
    print("  1) deflate : 一般的な可逆圧縮（デフォルト）")
    print("  2) lzw     : LZW 可逆圧縮")
    print("  3) zstd    : Zstandard 可逆圧縮（GDAL が対応している環境のみ）")

    while True:
        choice = input("番号を選んでください [0-3] (デフォルト=1): ").strip()
        if not choice:
            return options["1"][1]
        if choice in options:
            return options[choice][1]
        print("  ※ 0〜3 の番号で選択してください。")


def build_output_dir(input_dir: Path, target_res: float) -> Path:
    """解像度から出力サブディレクトリ名を決定して作成する。"""
    if float(int(target_res)) == target_res:
        res_str = f"{int(target_res)}"
    else:
        res_str = f"{target_res}".rstrip("0").rstrip(".")

    out_dir = input_dir / f"res_{res_str}m"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_output_path(out_dir: Path, src_path: Path, target_res: float) -> Path:
    """入力ファイル名と解像度から出力ファイルパスを作成する。"""
    if float(int(target_res)) == target_res:
        res_str = f"{int(target_res)}"
    else:
        res_str = f"{target_res}".rstrip("0").rstrip(".")

    new_name = f"{src_path.stem}_res{res_str}m.tif"
    return out_dir / new_name


# =========================
# 解像度変更のメイン処理
# =========================

def resample_raster(
    src_path: Path,
    dst_path: Path,
    target_res: float,
    resampling: Resampling,
    compress: str | None,
) -> None:
    """
    1 枚のラスタについて、指定した解像度にリサンプリングして書き出す。

    - CRS は変更しない。
    - 元のラスタの外接矩形 (bounds) を維持しつつ、ピクセルサイズだけ変える。
    - 元データの外側は nodata のままにし、値を新たに作らない（外挿しない）。
    """
    with rasterio.open(src_path) as src:
        src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata
        left, bottom, right, top = src.bounds

        # 新しい幅・高さを計算
        width_new = int(max(1, round((right - left) / target_res)))
        height_new = int(max(1, round((top - bottom) / target_res)))

        # 新しいアフィン変換
        transform_new = Affine(
            target_res, 0.0, left,
            0.0, -target_res, top
        )

        profile = src.profile.copy()
        profile.update(
            {
                "width": width_new,
                "height": height_new,
                "transform": transform_new,
            }
        )

        # 元のブロック関連設定は一旦全部消す（ここが今回の肝）
        for k in ("tiled", "blockxsize", "blockysize", "blockxysize"):
            profile.pop(k, None)

        # 圧縮設定（すべて可逆）
        if compress is not None:
            profile["compress"] = compress
            # 浮動小数のときは predictor=3 がよく使われる
            try:
                if np.issubdtype(np.dtype(src.dtypes[0]), np.floating):
                    profile["predictor"] = 3
                else:
                    profile["predictor"] = 2
            except Exception:
                pass
        else:
            profile.pop("compress", None)
            profile.pop("predictor", None)

        # nodata は元と合わせる（None の場合はそのまま）
        if src_nodata is not None:
            profile["nodata"] = src_nodata

        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                dst_data = np.empty((height_new, width_new), dtype=src.dtypes[band_idx - 1])

                # 事前に nodata で初期化しておく（元データの外側は nodata のまま）
                if src_nodata is not None:
                    dst_data.fill(src_nodata)
                else:
                    dst_data.fill(0)

                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=dst_data,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    src_nodata=src_nodata,
                    dst_transform=transform_new,
                    dst_crs=src_crs,
                    dst_nodata=src_nodata,
                    resampling=resampling,
                )

                dst.write(dst_data, band_idx)


# =========================
# エントリポイント
# =========================

def main():
    print("=== DEM / 地形特徴量 解像度変更ツール（対話式） ===")

    # 1) 入力ディレクトリ
    input_dir = ask_directory_path("入力ディレクトリのパスを入力してください: ")

    # 2) 対象拡張子
    exts = ask_extensions()
    print(f"  -> 対象拡張子: {', '.join(exts)}")

    # 3) 対象ファイル一覧
    raster_files = find_raster_files(input_dir, exts)
    if not raster_files:
        print("※ 対象ファイルが見つかりませんでした。処理を終了します。")
        sys.exit(0)

    print(f"\n[INFO] 対象ファイル数: {len(raster_files)} 枚")
    for f in raster_files[:3]:
        print(f"  - {f.name}")
    if len(raster_files) > 3:
        print("  ...")

    # 4) CRS / 元解像度の確認（サンプル1枚目）
    px_x, px_y, unit_name, epsg = show_crs_and_resolution(raster_files[0])

    # 5) 出力ピクセル解像度（複数）を取得
    target_res_list = ask_resolutions(px_x, unit_name)

    # 6) 各解像度が up / down / same のどれかを判定
    downs = [r for r in target_res_list if r > px_x and not math.isclose(r, px_x, rel_tol=1e-6, abs_tol=1e-6)]
    ups = [r for r in target_res_list if r < px_x and not math.isclose(r, px_x, rel_tol=1e-6, abs_tol=1e-6)]
    sames = [r for r in target_res_list if math.isclose(r, px_x, rel_tol=1e-6, abs_tol=1e-6)]

    print("\n[INFO] サンプリング方向の内訳]")
    if downs:
        print(f"  ダウンサンプリング (元より粗い): {downs}")
    if ups:
        print(f"  アップサンプリング (元より細かい): {ups}")
    if sames:
        print(f"  ほぼ同一解像度 (same): {sames}")
    if not downs and not ups and not sames:
        print("  ※ 内訳判定に失敗しました。指定解像度を見直してください。")
        sys.exit(1)

    # 7) up/down 用のリサンプリング手法を選択
    resampling_down = None
    resampling_up = None

    if downs:
        print("\nこのフォルダには『ダウンサンプリング』（解像度を粗くする）対象の解像度が含まれます。")
        resampling_down = ask_resampling_method_down()

    if ups or sames:
        print("\nこのフォルダには『アップサンプリング or ほぼ同一解像度』の対象が含まれます。")
        resampling_up = ask_resampling_method_up()

    # 8) 圧縮方式（可逆）を選択
    compress = ask_compression()

    # 9) 設定の確認
    print("\n[設定の確認]")
    print(f"  入力ディレクトリ : {input_dir}")
    print(f"  対象ファイル数   : {len(raster_files)} 枚")
    print(f"  EPSG             : {epsg if epsg is not None else '不明'}")
    print(f"  CRS 線単位       : {unit_name}")
    print(f"  元ピクセルサイズ : {px_x:.6f} × {px_y:.6f} {unit_name}")
    print(f"  出力解像度一覧   : {', '.join(str(r) for r in target_res_list)} {unit_name}")
    if downs:
        print(f"    ダウン用       : {downs} → {resampling_down.name}")
    if ups:
        print(f"    アップ用       : {ups} → {resampling_up.name}")
    if sames:
        print(f"    same 用        : {sames} → {resampling_up.name}")
    print(f"  圧縮方式         : {compress if compress is not None else 'none (無圧縮)'}")
    print("  備考             : 元ラスタの bounds 内のみを使用し、外側は nodata のままとします。")
    print("                      出力は解像度ごとに res_XXXm/ サブフォルダに保存されます。")

    confirm = input("\nこの設定で処理を開始してよろしいですか？ [y/N]: ").strip().lower()
    if confirm not in ("y", "yes"):
        print("処理を中止しました。")
        sys.exit(0)

    # 10) メインループ
    print("\n=== 解像度変更を開始します ===")
    total_success = 0
    total_fail = 0

    for target_res in target_res_list:
        # 解像度ごとの方向とリサンプリングの決定
        if math.isclose(target_res, px_x, rel_tol=1e-6, abs_tol=1e-6) or target_res < px_x:
            direction = "up/同一"
            resampling = resampling_up
        else:
            direction = "down"
            resampling = resampling_down

        out_dir = build_output_dir(input_dir, target_res)
        print(f"\n--- 解像度 {target_res} {unit_name} 用の処理を開始 ---")
        print(f"  出力ディレクトリ : {out_dir}")
        print(f"  サンプリング方向 : {direction}")
        print(f"  リサンプリング   : {resampling.name}")

        success = 0
        fail = 0

        for idx, src_path in enumerate(raster_files, start=1):
            dst_path = build_output_path(out_dir, src_path, target_res)
            print(f"[{idx}/{len(raster_files)}] {src_path.name} → {dst_path.name}")

            try:
                resample_raster(src_path, dst_path, target_res, resampling, compress)
                success += 1
            except Exception as e:
                print(f"  !!! エラーが発生しました: {e}")
                fail += 1

        print(f"  >> 解像度 {target_res} の処理完了: 正常 {success} / 失敗 {fail}")
        total_success += success
        total_fail += fail

    print("\n=== 全解像度の処理完了 ===")
    print(f"  総正常終了 : {total_success} 枚")
    print(f"  総失敗     : {total_fail} 枚")
    print(f"  入力元     : {input_dir}")
    print("  出力先例   :")
    for target_res in target_res_list:
        print(f"    - {build_output_dir(input_dir, target_res)}")


if __name__ == "__main__":
    main()
