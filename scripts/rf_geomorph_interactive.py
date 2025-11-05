#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rf_geomorph_interactive.py（完全版）
 - CSV/Parquet 読み込み（pyarrow推奨）
 - 学習: Pipeline(SimpleImputer(median)→RandomForest) / 実験タグ(run_id) / クラス分布 & 混同行列PNG保存
 - 予測: メタの run_id を出力名に付与 / 正解ラベルがあれば自動評価（5桁→7桁マッピングCSV対応）

依存: numpy, pandas, scikit-learn, joblib, matplotlib, (pyarrow: Parquet I/O)
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
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

import matplotlib.pyplot as plt
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
        "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Osaka",
        # 共通/OSS
        "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "TakaoGothic"
    ]

    installed = {f.name: f.fname for f in fm.fontManager.ttflist}
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

def input_indices(prompt: str, ncols: int) -> list:
    """
    "0,1,2-5,7" のような入力に対応してインデックスのリストを返す
    """
    s = input(prompt).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    idx = []
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a = int(a); b = int(b)
            if a > b: a, b = b, a
            idx.extend(list(range(a, b+1)))
        else:
            idx.append(int(p))
    idx = [i for i in idx if 0 <= i < ncols]
    idx = sorted(set(idx))
    return idx

def ensure_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    指定列を数値化（エラーは NaN）。存在しない列は無視。
    """
    work = df.copy()
    for c in cols:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    return work

def read_table_auto(path: str) -> pd.DataFrame:
    """
    CSV / Parquet / GPKG を自動判別して DataFrame を返す。
    GPKG はレイヤー選択・部分読みに対応。
    """
    path = strip_quotes(path)
    low = path.lower()
    if low.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    if low.endswith(".gpkg"):
        # 最初は軽く試す：head=500 だけ読み、列確認してから本読み（ユーザー選択）
        print("[INFO] GPKG を軽くスキャンします（先頭500行）...")
        df_preview = _read_gpkg_smart(path, head=500)
        list_columns(df_preview, "プレビュー（先頭500行）のカラム")
        use_preview = input("この列構成で読み込みますか？ [Y/n]: ").strip().lower()
        if use_preview in ("", "y", "yes"):
            # 全量読み（列は数値列中心にしたいならここで columns を指定してもOK）
            return _read_gpkg_smart(path)
        # 絞り込みオプション
        use_bbox = input("bbox（minx,miny,maxx,maxy）で絞りますか？ [y/N]: ").strip().lower() == "y"
        bbox = None
        if use_bbox:
            vals = input("bbox をカンマ区切りで: ").strip().split(",")
            bbox = tuple(float(v) for v in vals)
        use_head = input("先頭N行だけ読みますか？（空=全量）: ").strip()
        head_n = int(use_head) if use_head else None
        # 列絞り（軽量化）
        pick_cols = input("読み込む列をカンマ区切りで（空=全部）: ").strip()
        columns = [c.strip() for c in pick_cols.split(",") if c.strip()] if pick_cols else None
        return _read_gpkg_smart(path, columns=columns, bbox=bbox, head=head_n)
    return pd.read_csv(path)

def _read_gpkg_smart(path: str, layer: str | None = None, columns: list | None = None,
                     bbox: tuple | None = None, head: int | None = None) -> pd.DataFrame:
    """
    GPKGを賢く読む:
      - レイヤー未指定なら選択プロンプト
      - pyogrio があれば高速経路
      - columns, bbox（minx, miny, maxx, maxy）, head で部分読み
      - Pointはx,y展開、その他はcentroid x,y 追加
    """
    layers = fiona.listlayers(path)
    if layer is None:
        if len(layers) == 0:
            raise RuntimeError("GPKG 内にレイヤーが見つかりません")
        if len(layers) == 1:
            layer = layers[0]
        else:
            print("\n=== GPKG レイヤー一覧 ===")
            for i, nm in enumerate(layers):
                print(f"[{i:02d}] {nm}")
            sel = input("読み込むレイヤー番号を選んでください: ").strip()
            layer = layers[int(sel)]

    # まずは列名だけ素早く知る（head 読みの前に）
    if _HAS_PYOGRIO:
        gdf = _pg_read(path, layer=layer, bbox=bbox, columns=columns)
    else:
        gdf = gpd.read_file(path, layer=layer, bbox=bbox)

    if head is not None:
        gdf = gdf.head(head)

    # geometry→x,y
    if "geometry" in gdf.columns and not gdf.geometry.isna().all():
        if gdf.geometry.geom_type.isin(["Point"]).all():
            df = gdf.copy()
            df["x"] = gdf.geometry.x
            df["y"] = gdf.geometry.y
            return pd.DataFrame(df.drop(columns="geometry"))
        else:
            cen = gdf.geometry.centroid
            df = gdf.copy()
            df["x"] = cen.x
            df["y"] = cen.y
            return pd.DataFrame(df.drop(columns="geometry"))
    return pd.DataFrame(gdf)

def save_meta(path: str, meta: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_meta(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_experiment_tag(test_size, rs, n_estimators, max_depth, class_weight, extra_note=""):
    md = "None" if max_depth is None else str(max_depth)
    cw = "none" if not class_weight else str(class_weight)
    base = f"ts{test_size}_rs{rs}_nest{n_estimators}_md{md}_cw{cw}"
    if extra_note:
        base += f"_{extra_note}"
    return base

def plot_class_distribution(y_all, classes_order=None, title="Class distribution", save_path=None):
    vc = pd.Series(y_all).value_counts().sort_index()
    if classes_order is not None:
        vc = vc.reindex(classes_order, fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    vc.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160)
    plt.close(fig)

def plot_confmat_heatmaps(y_true, y_pred, labels, prefix_path, tick_labels=None):
    """
    混同行列PNGを保存
      - labels: confusion_matrix に渡す「値（例: [0,1,2,...]）」
      - tick_labels: 画像軸に表示する文字列（None なら labels をそのまま表示）
    """
    cm_raw  = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    if tick_labels is None:
        tick_labels = labels

    def _plot(cm, title, out_png):
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(tick_labels, rotation=90)
        ax.set_yticklabels(tick_labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                txt = f"{val:.2f}" if cm.dtype.kind == "f" else f"{int(val)}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)

    _plot(cm_raw,  "Confusion Matrix (raw counts)",     f"{prefix_path}_cm_raw.png")
    _plot(cm_norm, "Confusion Matrix (row-normalized)", f"{prefix_path}_cm_norm.png")

# =========================================================
# 学習モード
# =========================================================

def train_mode():
    print("\n=== 学習モード ===")
    data_path = strip_quotes(input("学習用 CSV/Parquet/GPKG のパス: ").strip())
    if not os.path.exists(data_path):
        print("ファイルが見つかりません。終了します。")
        sys.exit(1)

    df = read_table_auto(data_path)
    list_columns(df, "学習データのカラム一覧")

    # 目的変数（ラベル）列
    t_in = input("目的変数（ラベル）列の番号を1つ指定してください（例: 3）: ").strip()
    try:
        t_idx = int(t_in)
        target_col = df.columns[t_idx]
    except Exception:
        print("番号指定が不正です。終了します。")
        sys.exit(1)
    print(f"ターゲット列: {target_col}")

    # 既定の特徴量候補（数値列、x/y/geometry/target除外）
    default_exclude = {"x", "y", "geometry", target_col}
    numeric_cols = [c for c in df.columns if c not in default_exclude and pd.api.types.is_numeric_dtype(df[c])]
    print("\n既定の特徴量候補（数値列）:")
    print(", ".join(numeric_cols[:20]) + (" ..." if len(numeric_cols) > 20 else ""))
    use_default = input("既定の候補を使いますか？ [Y/n]: ").strip().lower()
    if use_default in ("", "y", "yes"):
        feature_cols = numeric_cols
    else:
        f_idx = input_indices("特徴量列の番号をカンマ区切り（範囲OK: 0-5）で指定: ", len(df.columns))
        feature_cols = [df.columns[i] for i in f_idx]

    # 数値化（非数値は NaN）※特徴量のみ
    work = ensure_numeric(df, feature_cols + [target_col])
    X_all = work[feature_cols]
    y_raw = df[target_col]  # 元の型を保持（文字列ラベル対応）
    keep_mask = ~y_raw.isna()
    X_all = X_all[keep_mask]

    # 文字列ラベル対応：LabelEncoder で数値化
    enc = None
    if (y_raw.dtype == "object") or (not pd.api.types.is_numeric_dtype(y_raw)):
        enc = LabelEncoder()
        y_all = enc.fit_transform(y_raw[keep_mask].astype(str))
    else:
        y_all = pd.to_numeric(y_raw[keep_mask], errors="coerce").astype("int64")
    # ndarray を明示しておく（下流で .values を使わない）
    y_all = np.asarray(y_all)

    # パラメータ入力
    test_size = float(input("テストサイズ（0.0〜0.9、空=0.2）: ").strip() or 0.2)
    rs = int(input("random_state（空=42）: ").strip() or 42)
    n_estimators = int(input("n_estimators（空=300）: ").strip() or 300)
    max_depth_in = input("max_depth（空=None）: ").strip()
    max_depth = int(max_depth_in) if max_depth_in else None
    cw_in = input("class_weight（空=なし, 'balanced' 推奨）: ").strip()
    class_weight = cw_in if cw_in else None

    # 実験タグ
    note = input("実験メモ/タグ（例: stride10_balanced。空可）: ").strip()
    exp_tag = build_experiment_tag(test_size, rs, n_estimators, max_depth, class_weight, note)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{exp_tag}_{stamp}"
    print(f"[INFO] experiment tag: {run_id}")

    # 分割（層化）
    X_train, X_test, y_train, y_test = train_test_split(
        X_all.values, y_all, test_size=test_size, random_state=rs, stratify=y_all
    )

    # Pipeline
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=rs,
            n_jobs=-1,
            class_weight=class_weight,
            max_features="sqrt"
        ))
    ])

    print("\n学習中...")
    pipe.fit(X_train, y_train)

    # 評価
    y_pred = pipe.predict(X_test)
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    print("\n=== Classification Report (macro avg 推奨) ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # 特徴量重要度
    rf = pipe.named_steps["rf"]
    importances = rf.feature_importances_
    feat_importance = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    print("\n=== Feature Importance (Top 20) ===")
    print(feat_importance.head(20))

    # 可視化出力
    out_dir = Path("rf_runs"); out_dir.mkdir(exist_ok=True)
    plot_class_distribution(
        y_all,
        classes_order=sorted(pd.Series(y_all).unique()),
        title="Class distribution (full data)",
        save_path=str(out_dir / f"{run_id}_class_dist.png")
    )

    # 混同行列ラベル：エンコーダがあれば元のテキストに戻して可読化
    if enc is not None:
        labels_idx   = list(range(len(enc.classes_)))   # 計算用（整数）
        tick_labels  = enc.classes_.tolist()            # 表示用（文字列）
    else:
        labels_idx   = sorted(pd.Series(y_test).unique().tolist())
        tick_labels  = labels_idx

    plot_confmat_heatmaps(y_test, y_pred, labels_idx, str(out_dir / f"{run_id}"), tick_labels=tick_labels)

    # 保存（既定名に run_id を付与）
    default_model = f"rf_model_{run_id}.joblib"
    default_meta  = f"rf_meta_{run_id}.json"
    default_imp   = f"rf_feature_importance_{run_id}.csv"
    default_le    = f"label_encoder_{run_id}.json"

    model_out = strip_quotes(input(f"保存するモデル名（空={default_model}）: ").strip() or default_model)
    meta_out  = strip_quotes(input(f"保存するメタ名（空={default_meta}）: ").strip() or default_meta)
    imp_out   = strip_quotes(input(f"重要度CSV（空={default_imp}、保存しない=空白1文字）: ").strip() or default_imp)

    joblib.dump(pipe, model_out)
    meta = {
        "version": 3,
        "run_id": run_id,
        "experiment_tag": exp_tag,
        "timestamp": stamp,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "classes_": rf.classes_.tolist(),  # 学習に使われた内部クラス（数値）
        "label_encoder_json": default_le if enc is not None else None,
        "train_source": os.path.abspath(data_path),
        "rf_params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": rs,
            "class_weight": class_weight,
            "max_features": "sqrt",
            "test_size": test_size
        },
        "artifacts": {
            "class_dist_png": str(out_dir / f"{run_id}_class_dist.png"),
            "cm_raw_png": str(out_dir / f"{run_id}_cm_raw.png"),
            "cm_norm_png": str(out_dir / f"{run_id}_cm_norm.png")
        }
    }
    save_meta(meta_out, meta)
    print(f"\n✅ モデル保存: {model_out}")
    print(f"✅ メタ保存  : {meta_out}")

    if imp_out.strip():
        feat_importance.to_csv(imp_out, header=["importance"])
        print(f"✅ 重要度CSV : {imp_out}")

    # LabelEncoder を JSON で保存（日本語OK）
    if enc is not None:
        le_path = strip_quotes(input(f"ラベルエンコーダJSON（空={default_le}）: ").strip() or default_le)
        with open(le_path, "w", encoding="utf-8") as f:
            json.dump({"classes": enc.classes_.tolist()}, f, ensure_ascii=False, indent=2)
        print(f"✅ ラベルエンコーダ: {le_path}")

    # === 学習データ＋予測を GPKG 出力 ===
    choice = input("学習データ＋予測をGPKGに保存しますか？ [y/N]: ").strip().lower()
    if choice == "y":
        crs = ask_crs(default_epsg="EPSG:4326")

        base_in = os.path.splitext(os.path.basename(data_path))[0]
        default_gpkg = f"{base_in}_trainpred_{run_id}.gpkg"
        out_path = strip_quotes(input(f"保存先（空={default_gpkg}）: ").strip() or default_gpkg)

        # 全行の予測（keep_mask 行のみ予測可能）
        y_pred_all = np.full(len(df), np.nan, dtype=object)
        proba_max  = np.full(len(df), np.nan, dtype=float)

        X_ok = X_all.values  # keep_mask 済み
        y_pred_ok = pipe.predict(X_ok)
        if hasattr(pipe, "predict_proba"):
            proba_ok = pipe.predict_proba(X_ok).max(axis=1)
        else:
            proba_ok = np.full(len(y_pred_ok), np.nan, dtype=float)

        # 埋め戻し
        idx_ok = np.where(keep_mask.values)[0]
        y_pred_all[idx_ok] = y_pred_ok
        proba_max[idx_ok]  = proba_ok

        out_df = df.copy()
        out_df["y_true"]     = y_raw
        # エンコーダがあれば学習時の文字ラベルへ
        if enc is not None:
            inv = {i: lbl for i, lbl in enumerate(enc.classes_)}
            out_df["y_pred"] = [inv[int(v)] if isinstance(v, (int, np.integer)) else np.nan for v in y_pred_all]
        else:
            out_df["y_pred"] = y_pred_all
        out_df["proba_max"]  = proba_max

        layer_name = f"trainpred_{run_id}"
        save_gpkg_with_points(out_df, out_path, x_col="x", y_col="y",
                              crs_epsg=crs, layer_name=layer_name)
        print(f"✅ GPKG 保存: {out_path}（レイヤ: {layer_name}, CRS={crs}）")

    # 重要度バー図（おまけ）← if の外で常に出力
    fig, ax = plt.subplots(figsize=(9, 6))
    feat_importance.head(25).plot(kind="barh", ax=ax)
    ax.invert_yaxis(); ax.set_title("Feature importance (Top 25)")
    fig.tight_layout(); fig.savefig(Path(imp_out).with_suffix(".png"), dpi=160)
    plt.close(fig)

# =========================================================
# 予測モード（run_id 出力名 / 正解ラベル評価 / 5桁→7桁マップ対応）
# =========================================================

def predict_mode():
    print("\n=== 予測モード ===")
    model_path = strip_quotes(input(f"モデルファイル（空={MODEL_DEFAULT} でもOK。学習で保存したもの推奨）: ").strip() or MODEL_DEFAULT)
    meta_path  = strip_quotes(input(f"メタファイル  （空={META_DEFAULT} でもOK。学習で保存したもの推奨）: ").strip() or META_DEFAULT)
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        print("モデル/メタのどちらかが見つかりません。終了します。")
        sys.exit(1)

    clf  = joblib.load(model_path)  # Pipeline
    meta = load_meta(meta_path)
    saved_feats   = meta.get("feature_cols", [])
    saved_classes = meta.get("classes_", [])
    run_id        = meta.get("run_id", "noRunID")
    target_col_tr = meta.get("target_col", None)

    # 学習時の LabelEncoder を（あれば）ロード
    enc_classes = None
    le_hint = meta.get("label_encoder_json", None)
    cand = []
    if le_hint:
        cand.append(Path(meta_path).with_name(Path(le_hint).name))
    cand.append(Path(meta_path).with_name(f"label_encoder_{run_id}.json"))
    cand.append(Path(meta_path).with_name("label_encoder.json"))
    for p in cand:
        if Path(p).exists():
            try:
                enc_classes = np.array(json.load(open(p, "r", encoding="utf-8"))["classes"])
                break
            except Exception:
                pass

    print(f"\n学習時 run_id : {run_id}")
    print(f"学習時の特徴量: {saved_feats}")
    print(f"学習時クラス  : {saved_classes}")

    apply_path = strip_quotes(input("予測に用いる CSV/Parquet/GPKG のパス: ").strip())
    if not os.path.exists(apply_path):
        print("ファイルが見つかりません。終了します。")
        sys.exit(1)

    df = read_table_auto(apply_path)
    list_columns(df, "適用データのカラム一覧")

    def _alias_candidates(colname: str) -> list[str]:
        cands = set()
        pairs = [("ClipMeL", "ClipMeU"), ("ClipMeU", "ClipMeL"),
                 ("MeL", "MeU"), ("MeU", "MeL")]
        for src, dst in pairs:
            if src in colname:
                cands.add(colname.replace(src, dst))
        cands.discard(colname)
        return list(cands)

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
        print("\n[INFO] 列名エイリアス適用（学習名 ← 適用データ名）:")
        for k, v in alias_map.items():
            print(f"  {k} <- {v}")

    if missing_feats:
        print("\n⚠ 次の学習特徴量は見つかりませんでした（エイリアスでも不一致）:")
        print("   " + ", ".join(missing_feats))

    # 特徴量列（学習時と同じを推奨）
    use_saved = input("学習時と同じ特徴量列を使いますか？ [Y/n]: ").strip().lower()
    if use_saved in ("", "y", "yes"):
        feature_cols = saved_feats
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            print(f"\n⚠ 学習時の特徴量が適用データに存在しません: {missing}")
            print("→ 特徴量を選び直してください。")
            use_saved = "n"

    if use_saved in ("n", "no"):
        f_idx = input_indices("予測に使う特徴量列の番号をカンマ区切りで指定: ", len(df.columns))
        feature_cols = [df.columns[i] for i in f_idx]

    # 数値化のみ（欠測埋めは Pipeline 内の Imputer が学習時中央値で実施）
    work = ensure_numeric(df, feature_cols)
    X = work[feature_cols]

    print("\n予測中...")
    pred_idx = clf.predict(X.values)

    # 可能なら確率も出力
    proba = None
    class_labels = []
    if hasattr(clf, "predict_proba"):
        try:
            proba = clf.predict_proba(X.values)
            # 確率列名（クラス名）：エンコーダがあればテキスト名
            if enc_classes is not None:
                class_labels = [enc_classes[int(i)] for i in clf.classes_]
            else:
                class_labels = clf.classes_.tolist()
        except Exception:
            proba = None

    # 出力ファイル名（run_id を付与）
    base_in   = os.path.splitext(os.path.basename(apply_path))[0]
    default_out_csv = f"{base_in}_pred_{run_id}.csv"
    default_out_pq  = f"{base_in}_pred_{run_id}.parquet"
    default_out_gpkg_layer = f"{base_in}_pred_{run_id}"  # GPKG のレイヤー名に利用
    default_out_gpkg = f"{base_in}_pred_{run_id}.gpkg"

    print(f"出力形式を選択してください：")
    print(f"  1) CSV（既定: {default_out_csv}）")
    print(f"  2) Parquet（既定: {default_out_pq}）")
    print(f"  3) GPKG（既定: {default_out_gpkg}）")
    fmt_choice = input("番号 [1/2/3, 空=2]: ").strip() or "2"

    if fmt_choice == "1":
        default_out = default_out_csv
    elif fmt_choice == "3":
        default_out = default_out_gpkg
    else:
        default_out = default_out_pq

    out_path = strip_quotes(input(f"結果の保存先（空={default_out}）: ").strip() or default_out)

    # 予測列を付与
    out = df.copy()
    # 予測ラベル：エンコーダがあればテキストに復元
    if enc_classes is not None:
        idx2txt = {int(i): enc_classes[int(i)] for i in range(len(enc_classes))}
        out["pred_label"] = [idx2txt.get(int(i), str(i)) for i in pred_idx]
    else:
        out["pred_label"] = pred_idx
    if proba is not None:
        out["pred_proba_max"] = proba.max(axis=1)
        topk = 3
        ix = np.argsort(-proba, axis=1)[:, :topk]
        for k in range(topk):
            out[f"top{k+1}_label"] = [class_labels[i] for i in ix[:, k]]
            out[f"top{k+1}_proba"] = proba[np.arange(len(proba)), ix[:, k]]


    # -----------------------
    # 正解ラベルがあれば評価
    # -----------------------
    has_truth = False
    truth_col = None
    if "label" in df.columns:
        truth_col = "label"; has_truth = True
    elif target_col_tr and (target_col_tr in df.columns):
        truth_col = target_col_tr; has_truth = True
    else:
        ask = input("正解ラベル列がありますか？ [y/N]: ").strip().lower()
        if ask == "y":
            idx = input_indices("正解ラベル列の番号を1つ指定してください: ", len(df.columns))
            if len(idx) == 1:
                truth_col = df.columns[idx[0]]
                has_truth = True

    out_dir = Path("rf_runs"); out_dir.mkdir(exist_ok=True)

    if has_truth:
        # enc_classes があれば文字列 → 索引、なければ数値化
        if enc_classes is not None and (df[truth_col].dtype == "object" or not pd.api.types.is_numeric_dtype(df[truth_col])):
            txt2idx = {lbl: i for i, lbl in enumerate(enc_classes)}
            y_true_series = df[truth_col].map(txt2idx)
            raw_truth_for_coverage = None  # 5桁↔7桁カバレッジ判定は不要
        else:
            y_true_series = pd.to_numeric(df[truth_col], errors="coerce")
            raw_truth_for_coverage = y_true_series  # 数値コード系の場合は後で5→7判定

        need_mapping = False
        if raw_truth_for_coverage is not None:
            # --- 改良した「コード体系不一致」検知（数値コード系のときだけ） ---
            train_classes = set(int(x) for x in saved_classes)
            truth_vals    = raw_truth_for_coverage.dropna().astype("int64")
            train_wo0     = train_classes - {0}
            cover_rate_wo0 = truth_vals.isin(list(train_wo0)).mean() if len(truth_vals) else 0.0
            digits_truth   = pd.Series(truth_vals.astype(str)).str.len().median() if len(truth_vals) else 0
            digits_train   = pd.Series(list(train_wo0)).astype(str).str.len().median() if len(train_wo0) else 0
            if cover_rate_wo0 < 0.05 or (digits_truth and digits_train and abs(digits_truth - digits_train) >= 2):
                need_mapping = True

        # 必要なら 5桁→7桁マッピング
        if need_mapping:
            print("\n⚠ 正解ラベルのコード体系が学習時と異なる可能性があります")
            print(f"   - 学習側(0除外)カバレッジ: {cover_rate_wo0:.3f} / 真の桁数中央値: {digits_truth} / 学習の桁数中央値: {digits_train}")
            use_map = input("5桁→7桁のマッピングCSVを使いますか？ [y/N]: ").strip().lower() == "y"
            if use_map:
                map_path = strip_quotes(input("マッピングCSVのパス（列名: five, seven）: ").strip())
                if os.path.exists(map_path):
                    mp = pd.read_csv(map_path)
                    if {"five","seven"}.issubset(mp.columns):
                        m = dict(zip(pd.to_numeric(mp["five"], errors="coerce"),
                                     pd.to_numeric(mp["seven"], errors="coerce")))
                        y_true_series = raw_truth_for_coverage.map(m)
                        # カバレッジ再チェック（任意）
                    else:
                        print("⚠ CSVに five, seven 列がありません。評価をスキップします。")
                        has_truth = False
                else:
                    print("⚠ マッピングCSVが見つかりません。評価をスキップします。")
                    has_truth = False
            else:
                print("評価をスキップします（コード体系が一致する列を指定すると評価できます）。")
                has_truth = False

    # 実際の評価と保存
    if has_truth:
        mask   = ~pd.isna(y_true_series)
        y_true = y_true_series[mask].astype("int64").values
        idxpos = np.where(mask.values)[0]
        y_pred = pred_idx[idxpos]   # ← pred ではなく pred_idx を使う！

        print("\n=== Evaluation on provided ground-truth ===")
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))

        labels_sorted = sorted(pd.Series(y_true).unique().tolist())
        base_in = os.path.splitext(os.path.basename(apply_path))[0]
        prefix  = str(out_dir / f"{base_in}_{run_id}_apply")
        plot_confmat_heatmaps(y_true, y_pred, labels_sorted, prefix)

        acc = accuracy_score(y_true, y_pred)
        metrics_json = {
            "run_id": run_id,
            "apply_source": os.path.abspath(apply_path),
            "truth_col": truth_col,
            "n_eval": int(len(y_true)),
            "accuracy": float(acc),
            "labels_sorted": labels_sorted,
            "artifacts": {
                "cm_raw_png": f"{prefix}_cm_raw.png",
                "cm_norm_png": f"{prefix}_cm_norm.png"
            }
        }
        metrics_path = Path(out_dir) / f"{base_in}_{run_id}_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_json, f, ensure_ascii=False, indent=2)
        print(f"✅ メトリクスJSON: {metrics_path}")

        # is_correct フラグ
        eval_flag = np.full(len(out), "", dtype=object)
        eq = (pred_idx[idxpos] == y_true).astype(int)
        eval_flag[idxpos] = eq
        out["is_correct"] = eval_flag

    # 結果保存（CSV / Parquet / GPKG）
    low = out_path.lower()
    if low.endswith(".gpkg"):
        # GPKG: Point geometry で保存
        crs = ask_crs(default_epsg="EPSG:4326")
        layer_name = default_out_gpkg_layer  # ex: f"{base_in}_pred_{run_id}"
        # 念のため x,y が無ければ作る（GPKG入力なら read_table_auto が付与済み）
        if "x" not in out.columns or "y" not in out.columns:
            if "geometry" in df.columns:
                gtmp = gpd.GeoSeries.from_wkt(df["geometry"]) if df["geometry"].dtype == object else df.geometry
                out["x"] = gtmp.centroid.x
                out["y"] = gtmp.centroid.y
            else:
                raise ValueError("GPKG出力には x,y 列が必要です。入力に geometry がある場合は centroid から作れます。")
        save_gpkg_with_points(out, out_path, x_col="x", y_col="y",
                              crs_epsg=crs, layer_name=layer_name)
        print(f"\n✅ 予測結果（GPKG）を書き出しました: {out_path}（レイヤ: {layer_name}, CRS={crs}）")

    elif low.endswith(".parquet") or low.endswith(".pq"):
        try:
            out.to_parquet(out_path, index=False)
            print(f"\n✅ 予測結果（Parquet）を書き出しました: {out_path}")
        except Exception as e:
            print(f"Parquet 失敗 → CSVで再保存します: {e}")
            out_fallback = f"{base_in}_pred_{run_id}.csv"
            out.to_csv(out_fallback, index=False, encoding="utf-8")
            print(f"✅ 予測結果（CSV）を書き出しました: {out_fallback}")
    else:
        out.to_csv(out_path, index=False, encoding="utf-8")
        print(f"\n✅ 予測結果（CSV）を書き出しました: {out_path}")

# =========================================================
# メイン
# =========================================================

def main():
    print("\n=== ランダムフォレスト（地形分類） ===")
    print("  1) 学習（train）")
    print("  2) 予測（predict）")
    print("  0) 終了")
    choice = input("番号を選んでください [0-2]: ").strip() or "1"
    if choice == "1":
        train_mode()
    elif choice == "2":
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