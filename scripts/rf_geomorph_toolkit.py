#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rf_geomorph_toolkit.py

ランダムフォレストを用いた地形分類ワークフローを
「対話式 3 モード」で一通り回せるツールキット。

モード概要
----------
1) 学習用データ作成（ラスター / ポリゴン → テーブル）
    - DEM から作成した特徴量ラスター（GeoTIFF 等）と、地形分類ポリゴンGPKGを読み込み、
      指定範囲・解像度のグリッド点にサンプリングして「学習用テーブル」
      （CSV / Parquet / GPKG）を作成する。
    - 範囲は
        * 手動で xmin, ymin, xmax, ymax を入力
        * ポリゴンGPKGの外接矩形を自動利用
      のどちらかを選択。
    - GPKG 出力時は x, y から Point geometry を生成し、レイヤ名 "train" で書き出す。
      （この GPKG をそのまま学習モードの入力として使える）

2) 学習（train）
    - 目的変数（例: 地形分類コード）と特徴量列を選択し、
      RandomForestClassifier で分類モデルを作成する。
    - 学習に使うデータ行数の上限を指定可能（動作確認用の軽量サンプル）。
      空 Enter の場合は「全件」を対象とする。

    [検証方法の選択]
      1) ホールドアウト法（学習/テストに分割）
          - データを「学習用」と「テスト用」に 1 回だけ分割。
          - `テストデータ割合` でテスト側に回す比率を指定（例: 0.2 ⇒ 8:2）。
      2) Monte Carlo Cross-Validation法（モンテカルロCV）
          - 毎回ランダムに学習/テストに分割し、その評価を n 回平均する方式。
          - `テストデータ割合` と `n_splits` でテスト比率と繰り返し回数を指定。
      3) k-fold Cross-Validation法（k分割CV）
          - データ全体を k 個に分割し、k 回学習・検証を繰り返す。
          - `k-分割数` で k を指定（例: 5 ⇒ 5-fold CV）。      

    [主なパラメータの意味]
      - random_state:
          乱数の種。固定すると毎回同じ分割・同じ結果になり、再現性がとれる。
      - 層化サンプリング / 層化CV:
          クラス比率（地形分類コードの出現比）が各分割でも元データと
          できるだけ同じになるように分割する設定。
          クラス不均衡な分類問題では、基本的に「Y（使う）」が推奨。
      - n_estimators（木の本数）:
          ランダムフォレスト内の決定木の本数。増やすと精度は上がりやすいが、
          計算時間も増える。とりあえず 200〜500 程度が無難。
      - max_depth（木の深さの上限）:
          各決定木の深さの上限。None（空 Enter）の場合は制限なし。
          深さに制限をかけると、過学習をある程度抑えられることがある。
      - class_weight='balanced':
          クラスごとの出現頻度に応じて自動で重み付けする設定。
          少数派クラスの誤分類を相対的に重く扱うことで、
          不均衡データに対して感度を上げる。

3) 予測（predict）
    - 保存済みモデルとメタ情報を読み込み、別のテーブルに対して予測を行う。
    - 予測対象行数の上限を対話的に指定可能。
        * GPKG の場合:
            - SQLite 経由で SELECT * FROM "<layer>" LIMIT N を実行し、
              先頭から N 行だけ属性テーブルを高速に取得する（geometry は落とす）。
        * CSV / Parquet の場合:
            - いったん全件読み込んだ上で、必要なら DataFrame.sample(...)
              によりランダムサンプル（is_random_sample=True）を取る。
    - 予測結果を元テーブルに結合した CSV / Parquet / GPKG を出力する。
      出力先パスは最初に決定し、そのパスに合わせて評価ファイルも同じディレクトリへ保存する。

    [評価機能]
      - 入力テーブルに「正解ラベル（target_col）」が含まれている場合は、
        予測と突き合わせて自動的に評価を行う。
      - LabelEncoder 使用有無に応じて、ラベル対応関係を自動チェックし、
        必要に応じて対話的にマッピングを補正できる。
      - 評価結果は次の CSV として保存される（予測結果 out_path を `..._pred_xxx.*` とした場合）
          * `..._pred_xxx_eval_confusion_matrix.csv`
          * `..._pred_xxx_eval_classification_report.csv`
          * `..._pred_xxx_eval_pred_summary.csv`
        （pred_summary はクラスごとの件数と平均確信度 proba_max を出力）

[モデル出力]
  - 実験ごとの保存先:
      <出力ルート>/<元テーブル名>_<run_id>/
        ├ rf_model_<run_id>.joblib
        ├ rf_meta_<run_id>.json
        ├ rf_feature_importance_<run_id>.csv
        ├ rf_feature_importance_<run_id>.png
        ├ rf_confusion_matrix_<run_id>.png
        └ rf_confusion_matrix_normalized_<run_id>.png
  - 直近モデルへのショートカット:
      <出力ルート>/
        ├ rf_model.joblib
        └ rf_meta.json

この docstring を見れば、
  * 学習用テーブルの作り方（ラスター＋ポリゴン → グリッドサンプリング）
  * 検証方法（ホールドアウト / Monte Carlo CV / k-分割CV）の違い
  * random_state / 層化サンプリング / class_weight の意味
  * 予測モードでの高速ローダー（GPKG=SQLite LIMIT / CSV・Parquet=ランダムサンプル）の挙動
  * モデルファイルや評価ファイルがどこに・どのファイル名で出るか
を後から振り返れるようにしている。
"""

import os
import sys
import json
import sqlite3
use_label_encoder = False

try:
    import pyarrow.parquet as _pq  # Parquet のメタ情報取得用（あれば使う）
except Exception:
    _pq = None

import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    StratifiedShuffleSplit,
    KFold,
    ShuffleSplit,
    cross_val_score,
    cross_validate,
    cross_val_predict,
)
from sklearn.base import clone
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
import joblib

# XGBoost（GPU 学習用）はオプション依存
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

import matplotlib.pyplot as plt
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


def save_geoparquet_with_points(
    df: pd.DataFrame,
    out_path: Path,
    x_col: str = "x",
    y_col: str = "y",
    crs_epsg: str = "EPSG:4326",
) -> None:
    """
    x, y から Point geometry を生成して GeoParquet として保存するヘルパ。
    """
    from shapely.geometry import Point

    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(
            f"GeoParquet 出力には '{x_col}', '{y_col}' 列が必要です。"
        )

    w = df.dropna(subset=[x_col, y_col]).copy()
    w[x_col] = w[x_col].astype(float)
    w[y_col] = w[y_col].astype(float)

    geom = [Point(xy) for xy in zip(w[x_col].values, w[y_col].values)]
    gdf = gpd.GeoDataFrame(w, geometry=geom, crs=crs_epsg)
    gdf.to_parquet(out_path, index=False)

def is_geoparquet_file(path: str) -> bool:
    """
    与えられた Parquet ファイルが GeoParquet かどうかの簡易判定。

    - pyarrow.parquet が使える場合:
        メタデータに "geo" キーが含まれているかどうかで判定。
    - 失敗した場合は False を返す。
    """
    from pathlib import Path as _Path

    p = _Path(path)
    if p.suffix.lower() not in (".parquet", ".pq"):
        return False

    try:
        import pyarrow.parquet as pq  # type: ignore[import]

        meta = pq.read_metadata(path)
        md = meta.metadata or {}
        # キーは bytes なので b"geo" を確認
        if b"geo" in md:
            return True
    except Exception:
        return False

    return False

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

def print_output_format_guide_from_shape(n_rows, n_cols):
    """
    df 全体をまだ作っていない段階で、
    行数と列数の「見込み」から出力形式のガイドを出す版。

    Parameters
    ----------
    n_rows : int
        想定される行数
    n_cols : int
        想定される列数
    """
    import math

    # 「全部 float64 だったとしたら」くらいの雑な見積もり
    # 8 byte/セル × 安全係数 1.5
    est_gb = n_rows * n_cols * 8 * 1.5 / (1024 ** 3)

    print("\n[INFO] 学習用テーブルのサイズ目安")
    print(f"  行数: {n_rows:,}")
    print(f"  列数: {n_cols:,}")
    print(f"  メモリ使用の概算: 約 {est_gb:.2f} GB（float64 相当）")

    total_ram_gb = None
    try:
        import psutil
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        print(f"  このPCの実メモリ: 約 {total_ram_gb:.1f} GB")
    except Exception:
        # psutil が無い場合は RAM 情報なしで続行
        pass

    print("\n[ガイド] 推奨される出力形式の目安:")

    if total_ram_gb is not None:
        ratio = est_gb / max(total_ram_gb, 0.1)
        if ratio > 0.7:
            print("  - GPKG への直接出力は非推奨（ほぼ確実に重くなります）。")
            print("  - まず .parquet を推奨（必要に応じて後で一部を GPKG 化）。")
        elif ratio > 0.4:
            print("  - GPKG 出力はかなり重くなる可能性があります。")
            print("  - .parquet / .csv を基本とし、確認用だけ GPKG にするのを推奨。")
        else:
            print("  - どの形式でも扱えるサイズですが、性能面では .parquet 推奨。")
    else:
        # RAM 不明な場合は絶対値でざっくり
        if est_gb > 8:
            print("  - GPKG への直接出力は非推奨（概算で 8GB 超）。")
            print("  - まず .parquet を推奨。")
        elif est_gb > 4:
            print("  - GPKG 出力はかなり重くなる可能性があります。")
            print("  - .parquet / .csv を基本とした方が安全です。")
        else:
            print("  - どの形式でも扱えますが、.parquet が最も実務的です。")

    print("  - 学習や予測だけが目的なら geometry 不要 → .parquet / .csv で十分です。")
    print("  - QGIS で確認したい場合だけ、絞り込んだ点群を GPKG に変換してください。\n")


def print_output_format_guide(df):
    """
    DataFrame の行数・列数からガイドを出すラッパー（後方互換用）。
    """
    n_rows = len(df)
    n_cols = len(df.columns)
    print_output_format_guide_from_shape(n_rows, n_cols)


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

def _plot_confusion_matrix(cm, labels, normalize=False,
                           title="Confusion matrix", cmap=None, save_path=None):
    """
    混同行列を画像として保存するヘルパー。
    labels: クラスラベルのリスト
             - 原則 len(labels) == cm.shape[0] を想定
             - ずれている場合は行列サイズに合わせて自動調整する
    """
    if normalize:
        with np.errstate(all="ignore"):
            cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_plot = cmn
    else:
        cm_plot = cm

    # === ラベル数と行列サイズの整合性をとる =========================
    n_classes = cm_plot.shape[0]

    if labels is None:
        # ラベルが渡されていない場合は 0,1,2,... の文字列にする
        labels_for_plot = [str(i) for i in range(n_classes)]
    else:
        labels_for_plot = list(labels)

    if len(labels_for_plot) != n_classes:
        # sklearn.metrics.confusion_matrix は、labels を指定しないと
        # 「テストデータに実際に出現したクラスのみ」を採用するため、
        # classification_report ベースのラベル一覧とズレることがある。
        # （例: support=0 のクラスが classification_report には出るが、
        #       confusion_matrix には含まれない）
        print(
            f"[WARN] labels の数 {len(labels_for_plot)} と混同行列のサイズ "
            f"{n_classes} が一致しません。行列サイズに合わせてラベルを調整します。"
        )
        if len(labels_for_plot) > n_classes:
            # 余っているラベルは末尾を切り捨て
            labels_for_plot = labels_for_plot[:n_classes]
        else:
            # 足りない場合はダミーラベルで埋める（通常は起こりにくい）
            labels_for_plot = labels_for_plot + [
                f"class_{i}" for i in range(len(labels_for_plot), n_classes)
            ]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap=cmap or "Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_title(title)

    tick_marks = np.arange(len(labels_for_plot))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels_for_plot, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels_for_plot)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    fmt = ".2f" if normalize else "d"
    thresh = cm_plot.max() / 2.0 if cm_plot.size > 0 else 0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            ax.text(
                j, i, format(cm_plot[i, j], fmt),
                ha="center", va="center",
                color="white" if cm_plot[i, j] > thresh else "black",
            )

    fig.tight_layout()

    if save_path:
        # 図を PNG として保存
        fig.savefig(save_path, dpi=150)

        # 追加: 図の元データ（cm_plot）を CSV でも保存
        try:
            import pandas as pd
            from pathlib import Path

            csv_path = Path(save_path).with_suffix(".csv")
            df_cm = pd.DataFrame(
                cm_plot,
                index=labels_for_plot,
                columns=labels_for_plot,
            )
            df_cm.to_csv(csv_path, encoding="utf-8-sig")
            print(f"[保存] 混同行列データ CSV: {csv_path}")
        except Exception as e:
            print(f"[WARN] 混同行列データ CSV の保存に失敗しました: {e}")

        plt.close(fig)

        print(f"[INFO] 混同行列を保存しました: {save_path}")


def _plot_learning_curve(
    estimator,
    X,
    y,
    cv,
    out_dir,
    run_id_prefix,
    run_id,
    eval_mode,
):
    """
    n_estimators を横軸にした学習曲線を描画する。

    - 上段: 学習 / 検証 log loss（メイン指標）
    - 下段: 検証 Accuracy / Macro-F1
    """
    if cv is None:
        print("[INFO] 学習曲線用の CV が指定されていないためスキップします。")
        return

    import numpy as np
    import matplotlib.pyplot as plt

    # X, y を numpy 配列にそろえる
    X = np.asarray(X)
    y = np.asarray(y)

    # estimator は Pipeline を想定（'imputer' + 'rf' or 'xgb'）
    if not hasattr(estimator, "named_steps"):
        print("[WARN] estimator が Pipeline ではないため学習曲線をスキップします。")
        return

    steps = estimator.named_steps
    imputer = steps.get("imputer", None)

    clf_name = None
    for name in ("rf", "xgb", "clf"):
        if name in steps:
            clf_name = name
            break

    if clf_name is None:
        print("[WARN] Pipeline に分類器ステップが見つからないため学習曲線をスキップします。")
        return

    base_clf = steps[clf_name]
    if not hasattr(base_clf, "n_estimators"):
        print("[WARN] 分類器が n_estimators を持っていないため学習曲線をスキップします。")
        return

    base_n = int(getattr(base_clf, "n_estimators", 0))
    if base_n <= 1:
        print("[INFO] n_estimators が 1 以下のため学習曲線をスキップします。")
        return

    # 評価する n_estimators のリスト（例: 200 → 20,40,...,200）
    num_points = min(10, base_n)
    est_list = np.linspace(
        max(1, base_n // num_points),
        base_n,
        num=num_points,
        dtype=int,
    )
    est_list = sorted(set(int(n) for n in est_list))

    train_logloss_mean = []
    val_logloss_mean = []
    val_acc_mean = []
    val_f1_mean = []

    print("[INFO] 学習曲線（n_estimators vs log loss / accuracy / macro-F1）を計算します...")
    print(f"[INFO] 評価する n_estimators: {est_list}")

    for n_estimators in est_list:
        train_ll_scores = []
        val_ll_scores = []
        acc_scores = []
        f1_scores_list = []

        for train_idx, valid_idx in cv.split(X, y):
            X_tr, X_val = X[train_idx], X[valid_idx]
            y_tr, y_val = y[train_idx], y[valid_idx]

            # 分類器をクローンして n_estimators を上書き
            clf = clone(base_clf)
            if hasattr(clf, "set_params"):
                clf.set_params(n_estimators=int(n_estimators))

            # imputer も都度クローンして Pipeline を作り直す
            if imputer is not None:
                imp = clone(imputer)
                from sklearn.pipeline import Pipeline as SkPipeline

                model = SkPipeline(
                    [("imputer", imp), (clf_name, clf)]
                )
            else:
                model = clf

            model.fit(X_tr, y_tr)

            # 予測確率（log loss 用: 学習 / 検証の両方）
            if hasattr(model, "predict_proba"):
                y_proba_tr = model.predict_proba(X_tr)
                y_proba_val = model.predict_proba(X_val)
            elif hasattr(model, "decision_function"):
                # decision_function から softmax で擬似確率化（学習 / 検証）
                scores_tr = model.decision_function(X_tr)
                scores_tr = np.atleast_2d(scores_tr)
                scores_tr = scores_tr - scores_tr.max(axis=1, keepdims=True)
                exp_scores_tr = np.exp(scores_tr)
                y_proba_tr = exp_scores_tr / exp_scores_tr.sum(axis=1, keepdims=True)

                scores_val = model.decision_function(X_val)
                scores_val = np.atleast_2d(scores_val)
                scores_val = scores_val - scores_val.max(axis=1, keepdims=True)
                exp_scores_val = np.exp(scores_val)
                y_proba_val = exp_scores_val / exp_scores_val.sum(axis=1, keepdims=True)
            else:
                y_proba_tr = None
                y_proba_val = None

            y_pred = model.predict(X_val)

            # log loss（確率が取れない場合は NaN）
            if y_proba_tr is not None:
                try:
                    ll_tr = log_loss(y_tr, y_proba_tr, labels=np.unique(y))
                except Exception:
                    ll_tr = np.nan
            else:
                ll_tr = np.nan

            if y_proba_val is not None:
                try:
                    ll_val = log_loss(y_val, y_proba_val, labels=np.unique(y))
                except Exception:
                    ll_val = np.nan
            else:
                ll_val = np.nan

            train_ll_scores.append(ll_tr)
            val_ll_scores.append(ll_val)
            acc_scores.append(accuracy_score(y_val, y_pred))
            f1_scores_list.append(f1_score(y_val, y_pred, average="macro"))

        train_logloss_mean.append(float(np.nanmean(train_ll_scores)))
        val_logloss_mean.append(float(np.nanmean(val_ll_scores)))
        val_acc_mean.append(float(np.mean(acc_scores)))
        val_f1_mean.append(float(np.mean(f1_scores_list)))

    # === プロット（2 段） ===
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8, 8),
        sharex=True,
        constrained_layout=True,
    )

    # 上段: log loss（学習 / 検証）
    ax1.plot(est_list, train_logloss_mean, marker="o", label="Train log loss")
    ax1.plot(est_list, val_logloss_mean, marker="s", label="Validation log loss")
    ax1.set_ylabel("log loss")
    ax1.set_title(
        f"Learning curve (eval={eval_mode})\nmain metric: validation log loss"
    )
    ax1.grid(True)
    ax1.legend(loc="best")

    # 下段: Accuracy / Macro-F1
    ax2.plot(est_list, val_acc_mean, marker="o", label="Accuracy")
    ax2.plot(est_list, val_f1_mean, marker="s", label="Macro F1")
    ax2.set_xlabel("n_estimators")
    ax2.set_ylabel("Accuracy / Macro-F1")
    ax2.grid(True)
    ax2.legend(loc="best")

    save_path = out_dir / f"{run_id_prefix}_learning_curve_{eval_mode}_{run_id}.png"

    # 追加: 学習曲線の元データ（評価指標の推移）も CSV で保存
    try:
        import pandas as pd
        csv_path = out_dir / f"{run_id_prefix}_learning_curve_{eval_mode}_{run_id}.csv"

        # n_estimators はループで使ったスカラーではなく、評価に使った一覧 est_list を保存する
        df_lc = pd.DataFrame({
            "n_estimators": est_list,
            "train_logloss_mean": train_logloss_mean,
            "val_logloss_mean": val_logloss_mean,
            "val_acc_mean": val_acc_mean,
            "val_f1_mean": val_f1_mean,
        })
        df_lc.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[保存] 学習曲線 元データ: {csv_path}")
    except Exception as e:
        print(f"[WARN] 学習曲線 元データの保存に失敗しました: {e}")

    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"[INFO] 学習曲線を保存しました: {save_path}")

# =========================================================
# ユーティリティ
# =========================================================

MODEL_DEFAULT = "rf_model.joblib"
META_DEFAULT  = "rf_meta.json"
# IMP_DEFAULT   = "rf_feature_importance.csv"


def strip_quotes(s: str) -> str:
    return s.strip().strip('"').strip("'")

def list_columns(df: pd.DataFrame, title="カラム一覧"):
    print(f"\n=== {title} ===")
    for i, c in enumerate(df.columns):
        print(f"[{i:03d}] {c}")

def input_indices(prompt: str, max_index: int, allow_empty=False):
    """
    カンマ区切りのインデックス入力を受け取り、整数リストを返す。

    サポートする書式:
      - 単一インデックス: 2, 5, 010 など
      - 範囲指定: 2-10, 0-41 など（両端を含む）
      - 混在: 0, 2-5, 10

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

            # 範囲指定（例: 2-10）
            if "-" in token:
                parts = token.split("-")
                if len(parts) != 2:
                    print(f"  ⚠ 範囲指定の形式が不正です: {token}（例: 2-10）")
                    ok = False
                    break
                start_str, end_str = parts[0].strip(), parts[1].strip()
                try:
                    start = int(start_str)
                    end = int(end_str)
                except ValueError:
                    print(f"  ⚠ 範囲指定を整数として解釈できません: {token}")
                    ok = False
                    break
                if start > end:
                    print(f"  ⚠ 範囲の順序が逆です: {token}（start ≤ end にしてください）")
                    ok = False
                    break
                if start < 0 or end > max_index:
                    print(f"  ⚠ 範囲外です: {token}（0〜{max_index} の間で指定してください）")
                    ok = False
                    break
                out.extend(range(start, end + 1))
            else:
                # 単一インデックス
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
            # 重複は消してソートしておくと気持ちいい
            out = sorted(set(out))
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

def ask_thinning_factor(prompt: str, default: int = 1) -> int:
    """
    GPKG 書き出し時の間引き率を尋ねるヘルパ。
      - 1   : 間引きなし（全件）
      - 10  : 1/10（10 行ごとに 1 行）
      - 100 : 1/100 ...
    """
    while True:
        s = input(f"{prompt}（1=間引きなし）: ").strip()
        if not s:
            return max(1, int(default))
        try:
            v = int(s)
        except ValueError:
            print("  ⚠ 整数として解釈できません。もう一度入力してください。")
            continue
        if v < 1:
            print("  ⚠ 1 以上の整数を指定してください。")
            continue
        return v

def ask_vector_output_format(default: str = "both") -> tuple[bool, bool]:
    """
    評価結果ベクタの出力形式（GPKG / GeoParquet / 両方）を選ぶ。
    戻り値: (save_gpkg, save_parquet)
    """
    default = default.lower()
    default_map = {"gpkg": "1", "geoparquet": "2", "both": "3"}
    default_choice = default_map.get(default, "3")

    while True:
        print("\n[ベクタ出力形式の選択]")
        print("  1) GPKG のみ")
        print("  2) GeoParquet のみ（.parquet）")
        print("  3) GPKG + GeoParquet の両方")
        s = input(f"番号を選択してください [1-3]（空={default_choice}）: ").strip()
        if not s:
            s = default_choice
        if s == "1":
            return True, False
        if s == "2":
            return False, True
        if s == "3":
            return True, True
        print("  ⚠ 1〜3 の番号で指定してください。")

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


def inspect_table(path: str, layer_name: str | None = None) -> tuple[list[str], int]:
    """
    入力テーブルの「列名リスト」と「総行数」を、可能な限り軽量に取得する。
    predict モードで、フルロード前に「ざっくりサイズ感」を知るために使う。

    - CSV :
        * ヘッダーだけ read_csv で取得し、
        * 行数は生テキストとして行カウント（ヘッダー1行を引く）
    - GPKG: SQLite を直接叩いて PRAGMA table_info / COUNT(*) で取得
    - Parquet: pyarrow があればメタデータから取得。なければ fallback として read_parquet
    - それ以外: read_csv(nrows=100) のような簡易ヘッダー取得に fallback

    戻り値:
        (columns: list[str], n_rows: int)
        取得に失敗した場合は ([], 0) を返す。
    """
    path = str(path)
    suffix = Path(path).suffix.lower()

    # -----------------------------
    # CSV
    # -----------------------------
    if suffix == ".csv":
        # 列名
        try:
            cols = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception as e:
            print(f"[WARN] inspect_table(csv) で列名取得に失敗しました: {e}")
            cols = []

        # 行数（ヘッダー行を除外）
        n_rows = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                # ヘッダーも含めた総行数
                n_lines = sum(1 for _ in f)
            n_rows = max(0, n_lines - 1)
        except Exception as e:
            print(f"[WARN] inspect_table(csv) で行数取得に失敗しました: {e}")
        return cols, int(n_rows)

    # -----------------------------
    # GPKG（GeoPackage = SQLite）
    # -----------------------------
    if suffix == ".gpkg":
        cols: list[str] = []
        n_rows: int = 0
        try:
            conn = sqlite3.connect(path)
            cur = conn.cursor()

            # レイヤ名が指定されていなければ gpkg_contents から 1つ拾う
            layer = layer_name
            if layer is None:
                try:
                    cur.execute("SELECT table_name FROM gpkg_contents LIMIT 1;")
                    row = cur.fetchone()
                    if row:
                        layer = row[0]
                except Exception:
                    layer = None

            if layer:
                # 列名
                cur.execute(f"PRAGMA table_info('{layer}')")
                cols = [r[1] for r in cur.fetchall()]  # 2列目がカラム名

                # 行数
                cur.execute(f"SELECT COUNT(*) FROM '{layer}';")
                row2 = cur.fetchone()
                if row2:
                    n_rows = int(row2[0])

            conn.close()
        except Exception as e:
            print(f"[WARN] inspect_table(gpkg) でメタ情報取得に失敗しました: {e}")
        return cols, int(n_rows)

    # -----------------------------
    # Parquet
    # -----------------------------
    if suffix in (".parquet", ".pq"):
        # pyarrow があればメタデータから高速取得
        if _pq is not None:
            try:
                pf = _pq.ParquetFile(path)
                cols = pf.schema.names
                n_rows = int(pf.metadata.num_rows)
                return list(cols), n_rows
            except Exception as e:
                print(f"[WARN] inspect_table(parquet/pyarrow) でメタ情報取得に失敗しました: {e}")

        # fallback: pandas で一旦読み込む（重いが最悪動く）
        try:
            df_head = pd.read_parquet(path)
            return list(df_head.columns), int(len(df_head))
        except Exception as e:
            print(f"[WARN] inspect_table(parquet/pandas) でメタ情報取得に失敗しました: {e}")
            return [], 0

    # -----------------------------
    # その他のフォーマット
    # -----------------------------
    # 「とりあえずヘッダーだけ読んでみる」レベルにとどめる
    try:
        df_head = pd.read_csv(path, nrows=100)
        return list(df_head.columns), int(len(df_head))
    except Exception as e:
        print(f"[WARN] inspect_table(その他) でメタ情報取得に失敗しました: {e}")
        return [], 0

def _read_gpkg_attributes_with_limit(
    path: str,
    layer_name: str | None = None,
    limit: int | None = None,
    random_sample_n: int | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    GPKG から「属性テーブルだけ」を読み込むための内部ヘルパ。
    GeoPandas を使わず、GPKG の属性テーブルだけを pandas.read_sql で読む
    Parameters
    ----------
    path : str
        GPKG ファイルパス
    layer_name : str | None
        読み込むレイヤ名。None のときは gpkg_contents から 1 つ拾う。
    limit: 先頭から何行読むか（None の場合は全件）
           ※ random_sample_n が指定されている場合は無視される。
    where: 追加の WHERE 句
    # random_sample_n を使う場合は SQLite の RANDOM() を使ってランダム抽出する。
    # 「再現性のあるランダムサンプリング」が必要なケースでは、
    # GPKG ではなく CSV/Parquet で読み込み、DataFrame.sample(..., random_state=...) を利用する想定。
    random_state : int | None
        乱数シード用パラメータ（現状 SQLite の RANDOM() には直接は効かないため未使用）。

    備考
    ----
    - geometry 列（geom / geometry）は読み込んだあとにドロップする。
      → 予測処理では属性だけを使い、
         出力時にあらためて x,y から geometry を作り直すか、
         元 GPKG から geometry を使う。
    """
    path = str(path)
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()

        layer = layer_name
        if layer is None:
            try:
                cur.execute("SELECT table_name FROM gpkg_contents LIMIT 1;")
                row = cur.fetchone()
                if row:
                    layer = row[0]
            except Exception:
                layer = None

        if not layer:
            raise RuntimeError("GPKG のレイヤ名を特定できませんでした。gpkg_contents が空かもしれません。")

        # 読み込み方法:
        # - random_sample_n が指定されていれば ORDER BY RANDOM() LIMIT ... でランダムサンプリング
        # - それ以外は従来どおり LIMIT 付き（または全件）で先頭から読み込む
        if random_sample_n is not None and random_sample_n > 0:
            n = int(random_sample_n)
            sql = f'SELECT * FROM "{layer}" ORDER BY RANDOM() LIMIT {n}'
            print(
                f"[INFO] GPKG（属性のみ, random_sample_n={n}）をランダムサンプリングして読み込み中: {path}"
            )
        else:
            sql = f'SELECT * FROM "{layer}"'
            if limit is not None and limit > 0:
                sql += f" LIMIT {int(limit)}"
            print(
                f"[INFO] GPKG（属性のみ, limit={limit if limit is not None else 'ALL'}）を読み込み中: {path}"
            )

        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()

    # geometry 列があれば一旦落とす（予測処理では不要。出力時に改めて geometry を付与する）
    for gc in ("geom", "geometry"):
        if gc in df.columns:
            df = df.drop(columns=[gc])
    return df


def load_table_for_predict(
    path: str,
    max_rows: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, bool, bool]:
    """
    予測用にテーブルを読み込むユーティリティ。

    戻り値: (df, is_random_sample, is_geoparquet)

    - max_rows が指定されている場合:
        * GPKG の場合は、SQLite 側で RANDOM() を用いて
          random_sample_n = max_rows 件をランダムサンプリングして取得する。
        * CSV/Parquet の場合は、一旦全件または max_rows 件を読み込んだうえで、
          DataFrame.sample(n=max_rows, random_state=random_state) によりランダム抽出する。

    戻り値: (df, is_random_sample)
        is_random_sample: max_rows を指定してランダムサンプリングした場合 True
    """
    path = str(path)
    suffix = Path(path).suffix.lower()
    is_geopq = False    

    # GPKG は、max_rows が指定されている場合はランダムサンプリング、
    # 指定されていない場合は全件読み込み（従来どおり先頭から）とする。
    if suffix == ".gpkg":
        if max_rows is not None and max_rows > 0:
            df = _read_gpkg_attributes_with_limit(
                path,
                layer_name=None,
                limit=None,
                random_sample_n=max_rows,
                random_state=random_state,
            )
            return df, True, False  # ランダムサンプル, GeoParquet ではない
        else:
            df = _read_gpkg_attributes_with_limit(path, layer_name=None, limit=None)
            return df, False, False  # 全件・先頭から

    # CSV
    if suffix == ".csv":
        df = pd.read_csv(path)
        if max_rows is not None and 0 < max_rows < len(df):
            df = df.sample(n=max_rows, random_state=random_state)
            return df, True
        return df, False

    # Parquet
    if suffix in (".parquet", ".pq"):
        # GeoParquet かどうか判定
        is_geopq = is_geoparquet_file(path)
        if is_geopq:
            try:
                # geometry 付きで読み込み、geometry 列だけ落として属性テーブルとして扱う
                gdf = gpd.read_parquet(path)
                geom_name = gdf.geometry.name if hasattr(gdf, "geometry") else "geometry"
                if geom_name in gdf.columns:
                    df = pd.DataFrame(gdf.drop(columns=[geom_name]))
                else:
                    df = pd.DataFrame(gdf)
                print("[INFO] GeoParquet として検出しました（geometry 列は学習・予測から除外）。")
            except Exception as e:
                print(
                    f"[WARN] GeoParquet としての読み込みに失敗したため、"
                    f"通常の Parquet として読み込みます: {e}"
                )
                df = pd.read_parquet(path)
                is_geopq = False
        else:
            df = pd.read_parquet(path)

        if max_rows is not None and 0 < max_rows < len(df):
            df = df.sample(n=max_rows, random_state=random_state)
            return df, True, is_geopq
        return df, False, is_geopq

    # その他（とりあえず CSV として読む）
    df = pd.read_csv(path)
    if max_rows is not None and 0 < max_rows < len(df):
        df = df.sample(n=max_rows, random_state=random_state)
        return df, True, False
    return df, False, False


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

    # --- 出力先の指定（ベースパスのみ先に聞いておく） ---
    out_base_input = strip_quotes(
        input(
            "学習用テーブルの出力ベースパス（拡張子なし。例: E:\\AiGeology\\train\\A-train / 相対パス=特徴量ディレクトリ基準）: "
        ).strip()
    )

    if not out_base_input:
        # 未指定なら特徴量ディレクトリ直下に自動生成
        out_base = os.path.join(root_dir, "training_data")
        print(f"[INFO] 出力ベースパス未指定のため、自動で設定しました: {out_base}")
    else:
        # 絶対パスかどうかで挙動を分ける
        if os.path.isabs(out_base_input):
            out_base = out_base_input
        else:
            # 相対パスの場合は特徴量ディレクトリ(root_dir) 基準で解釈
            out_base = os.path.join(root_dir, out_base_input)

    # Path 化（ここまでで out_base は絶対 or root_dir 基準になっている想定）
    base_out_path = Path(out_base)

    # 既存ディレクトリが指定された場合は、その中に同名ファイルを作成
    if base_out_path.is_dir():
        print(f"[INFO] 既存フォルダが指定されたため、その中に同名ファイルを作成します（{base_out_path.name}.*）。")
        base_out_path = base_out_path / base_out_path.name

    print(f"[INFO] 学習用テーブルの出力ベースパス: {base_out_path}")
    if base_out_path.suffix:
        print(f"[INFO] 拡張子 {base_out_path.suffix} は無視し、ベースパスとして扱います。")
        base_out_path = base_out_path.with_suffix("")

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
            print(f"  [R{i:02d}] {r}")
    else:
        print("\n[INFO] ラスターは検出されませんでした（ポリゴン属性だけでもテーブルは作成できます）。")

    if polygons:
        print("\n[INFO] 検出したポリゴンGPKG:")
        for i, g in enumerate(polygons):
            print(f"  [P{i:02d}] {g}")
    else:
        print("\n[INFO] ポリゴンGPKGは検出されませんでした。")

    # どのラスターを使うか（空Enter=全て）
    if rasters:
        use_r_idx = input(
            "\n特徴量として使用するラスターの番号 "
            "（例: 0,1,3 または R0,R1,R3。空=全て）: "
        ).strip()
        if use_r_idx:
            idxs: list[int] = []
            for tok in use_r_idx.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                # R0 / r0 形式 → 0 に変換
                m = re.fullmatch(r"[Rr](\d+)", tok)
                if m:
                    tok = m.group(1)
                try:
                    i = int(tok)
                except ValueError:
                    print(f"  ⚠ 無視します: '{tok}'")
                    continue
                if 0 <= i < len(rasters):
                    idxs.append(i)
                else:
                    print(f"  ⚠ 範囲外の番号なので無視します: {i}")
            rasters = [rasters[i] for i in idxs]
            if not rasters:
                print("  ⚠ 有効なラスターが選択されませんでした。ラスターなしで続行します。")

    # 属性を付与するポリゴンGPKG（任意）
    poly_for_attr: Path | None = None
    poly_attr_cols: list[str] | None = None
    selected_poly_attr_cols: list[str] | None = None
    if polygons:
        ans = input("ポリゴンGPKGの属性値も特徴量/ラベルとして付与しますか？ [y/N]: ").strip().lower() or "n"
        if ans.startswith("y"):
            p_idx = input(
                "  使用するポリゴンGPKGの番号 "
                "（例: 0 または P0。null/空=使用しない）: "
            ).strip()

            # null / none / 空文字 → 使用しない
            if not p_idx or p_idx.lower() in {"null", "none"}:
                print("  ポリゴン属性は使用しません。")
            else:
                m = re.fullmatch(r"[Pp](\d+)", p_idx)
                if m:
                    p_idx = m.group(1)
                try:
                    p_i = int(p_idx)
                except ValueError:
                    print("  ⚠ 番号が不正なのでポリゴン属性は使用しません。")
                else:
                    if 0 <= p_i < len(polygons):
                        poly_for_attr = polygons[p_i]
                        # ここで属性列の一覧表示と選択も行う
                        try:
                            import geopandas as gpd
                        except ImportError:
                            print("  ⚠ geopandas がインポートできないため、ポリゴン属性は付与しません。")
                            poly_for_attr = None
                        else:
                            try:
                                poly_gdf_preview = gpd.read_file(poly_for_attr)
                            except Exception as e:
                                print(f"  ⚠ ポリゴン属性の列情報を取得できませんでした: {e}")
                                poly_for_attr = None
                            else:
                                attr_cols = [c for c in poly_gdf_preview.columns if c.lower() not in {"geometry"}]
                                if not attr_cols:
                                    print("  ⚠ 利用可能な属性列がありません（geometry のみ）。")
                                    poly_attr_cols = None
                                else:
                                    print(f"\\n[INFO] ポリゴン属性の付与: {poly_for_attr}")
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
                                        selected_poly_attr_cols = [
                                            attr_cols[i] for i in idxs if 0 <= i < len(attr_cols)
                                        ]
                                        if not selected_poly_attr_cols:
                                            print("  ⚠ 有効な番号が指定されなかったため、全ての属性列を使用します。")
                                            selected_poly_attr_cols = attr_cols
                                    else:
                                        selected_poly_attr_cols = attr_cols
                    else:
                        print("  ⚠ 範囲外の番号なのでポリゴン属性は使用しません。")
        else:
            print("  ポリゴン属性は使用しません。")

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
        poly_for_extent = None

        # まず範囲用GPKGをどうするか決める
        print("\n[範囲用ポリゴンGPKGの指定]")
        if poly_for_attr:
            print(f"  現在のポリゴン属性用GPKG: {poly_for_attr}")
        if polygons:
            print("  検出されたポリゴンGPKG一覧:")
            for i, g in enumerate(polygons):
                print(f"    [P{i:02d}] {g}")
        ext_path_in = input("  範囲用GPKGのフルパス（空=上の一覧/属性用から選択）: ").strip()

        if ext_path_in:
            # 任意パスを直接指定
            poly_for_extent = strip_quotes(ext_path_in)
        else:
            # 何も指定されなかった場合は、既存情報から決める
            if not polygons and not poly_for_attr:
                print("ポリゴンGPKGが見つからないため、手動入力に切り替えます。")
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
                # 範囲取得用 GPKG を既存から選ぶ
                if not poly_for_attr:
                    print("\n[範囲用ポリゴンの選択]")
                    for i, g in enumerate(polygons):
                        print(f"  [P{i:02d}] {g}")
                    p_idx = input("範囲用ポリゴンGPKGの番号（例: 0 または P0）: ").strip()

                    if not p_idx:
                        print("番号が空です。範囲 2) を選んだ場合は必ずどれかを指定してください。")
                        print("必要なければ、範囲 1) 手動入力 を選んでください。")
                        return

                    m = re.fullmatch(r"[Pp](\d+)", p_idx)
                    if m:
                        p_idx = m.group(1)
                    try:
                        p_i = int(p_idx)
                    except ValueError:
                        print("番号が不正です。終了します。")
                        return
                    if not (0 <= p_i < len(polygons)):
                        print("番号が不正です。終了します。")
                        return
                    poly_for_extent = polygons[p_i]
                else:
                    # すでにポリゴン属性用GPKGが選択されていれば、それを使う
                    poly_for_extent = poly_for_attr

        # ここまでで poly_for_extent が決まっていれば、その GPKG から bbox を取る
        if poly_for_extent:
            try:
                layers = fiona.listlayers(poly_for_extent)
                if not layers:
                    print("  範囲用GPKGにレイヤが見つかりません。終了します。")
                    return

                print("  範囲用GPKGのレイヤ一覧:")
                for i, lname in enumerate(layers):
                    print(f"    [{i:02d}] {lname}")

                sel_layer = input("  使用レイヤの番号（空=0）: ").strip()
                if sel_layer == "":
                    # 何も指定されなければ 0 番目
                    layer_name = layers[0]
                else:
                    try:
                        idx = int(sel_layer)
                    except ValueError:
                        # 数字じゃなければ「名前で直接指定された」とみなす
                        if sel_layer in layers:
                            layer_name = sel_layer
                        else:
                            print("  ⚠ 指定されたレイヤ名/番号が正しくありません。最初のレイヤを使用します。")
                            layer_name = layers[0]
                    else:
                        if 0 <= idx < len(layers):
                            layer_name = layers[idx]
                        else:
                            print("  ⚠ 指定された番号が範囲外です。最初のレイヤを使用します。")
                            layer_name = layers[0]

            except Exception:
                print("  （レイヤ一覧の取得に失敗しました。レイヤ未指定で読み込みを試みます）")
                layer_name = None

            extent_gdf = gpd.read_file(poly_for_extent, layer=layer_name)

            # --- 属性列・属性値で範囲ポリゴンを絞り込む（任意） ---
            attr_cols = [c for c in extent_gdf.columns if c.lower() != "geometry"]
            used_attr = None
            used_value = None

            if attr_cols:
                print("\n  [範囲ポリゴンの絞り込み（任意）]")
                print("  このレイヤに含まれる属性列:")
                for i, c in enumerate(attr_cols):
                    print(f"    [{i:02d}] {c}")
                sel_attr = input("  範囲選択に使う属性列の番号（空=絞り込みなし）: ").strip()

                if sel_attr:
                    try:
                        attr_idx = int(sel_attr)
                    except ValueError:
                        print("  ⚠ 番号が不正のため、絞り込みは行わず全ポリゴンを対象とします。")
                    else:
                        if 0 <= attr_idx < len(attr_cols):
                            used_attr = attr_cols[attr_idx]
                            # ユニーク値ではなく、「行ごと」に候補として扱う
                            rows = extent_gdf[[used_attr]].dropna(subset=[used_attr])
                            if rows.empty:
                                print("  ⚠ 選択された属性列に有効な値が無いため、絞り込みは行いません。")
                                used_attr = None
                            else:
                                # 元のインデックスと値を保持
                                idx_list = list(rows.index)
                                vals = list(rows[used_attr])

                                print(f"\n  属性列 '{used_attr}' の値一覧（行ごと）:")
                                for i, (idx, v) in enumerate(zip(idx_list, vals)):
                                    print(f"    [{i:03d}] fid={idx}  {v}")

                                sel_val = input("  外接矩形を取りたい行の番号（空=絞り込みなし）: ").strip()
                                if sel_val:
                                    try:
                                        val_idx = int(sel_val)
                                    except ValueError:
                                        print("  ⚠ 番号が不正のため、絞り込みは行わず全ポリゴンを対象とします。")
                                        used_attr = None
                                    else:
                                        if 0 <= val_idx < len(idx_list):
                                            # 選ばれた 1 行だけを対象にする
                                            chosen_idx = idx_list[val_idx]
                                            used_value = vals[val_idx]
                                            subset = extent_gdf.loc[[chosen_idx]]
                                            if subset.empty:
                                                print("  ⚠ 絞り込み後のポリゴンが 0 件のため、全ポリゴンを対象とします。")
                                                used_attr = None
                                                used_value = None
                                            else:
                                                extent_gdf = subset
                                        else:
                                            print("  ⚠ 番号が範囲外のため、絞り込みは行わず全ポリゴンを対象とします。")
                                            used_attr = None

            # --- 最終的な外接矩形 ---
            xmin, ymin, xmax, ymax = extent_gdf.total_bounds
            if used_attr is not None and used_value is not None:
                print(
                    f"  → 属性 '{used_attr}' = '{used_value}' のポリゴン外接矩形: "
                    f"xmin={xmin:.3f}, ymin={ymin:.3f}, xmax={xmax:.3f}, ymax={ymax:.3f}"
                )
            else:
                print(
                    f"  → ポリゴン全体の外接矩形: "
                    f"xmin={xmin:.3f}, ymin={ymin:.3f}, xmax={xmax:.3f}, ymax={ymax:.3f}"
                )

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

    # --- 出力形式のガイド（概算） & 選択（サンプリング前） ---
    # 「x, y + ラスター列」を想定した列数でざっくり見積もり
    est_cols = 2 + (len(rasters) if rasters else 0)
    print_output_format_guide_from_shape(len(df), est_cols)

    print("\n[出力形式の選択]")
    print("  1) CSV（.csv）")
    print("  2) Parquet（.parquet）")
    print("  3) GeoPackage（.gpkg）")
    fmt = input("保存形式を選んでください [1-3]（空=Parquet推奨）: ").strip()

    # --- CRS の基準値（ラスター群があればその CRS を使用） ---
    # 関数の早い段階で初期化しておくことで、
    # 任意の分岐パスでも base_crs 未定義エラーを防ぐ
    base_crs = None

    # --- 出力用の座標系（CSV 以外） ---
    out_crs_epsg = None
    if fmt in ("2", "3"):  # Parquet / GPKG の場合のみ CRS を取得
        # 学習用テーブルの座標系として素直に EPSG:6673 を既定値にする
        out_crs_epsg = ask_crs(default_epsg="EPSG:6673")

    if fmt == "1":
        out_suffix = ".csv"
    elif fmt == "3":
        out_suffix = ".gpkg"
    else:
        out_suffix = ".parquet"

    out_path = str(base_out_path.with_suffix(out_suffix))

    # --- ラスター読み込み準備 ---
    if rasters:
        try:
            import rasterio
            from rasterio.transform import rowcol
        except Exception as e:
            print(f"⚠ rasterio がインポートできませんでした: {e}")
            print("   ラスターからのサンプリングはスキップします。")
            rasters = []

    # すべてのラスターで共通に使うグリッド座標（ndarray）
    # → rasterio.sample(coords) は Python ループ＋I/O が重いので使わず、
    #    x,y → row,col を一括で求めて配列から直接取り出す方式にする
    x_arr = df["x"].to_numpy()
    y_arr = df["y"].to_numpy()

    # ここに「列名 → 値配列」を貯めて、最後に一括で df に結合する
    cols_dict = {}

    for r_path in rasters:
        import rasterio

        # ファイルが途中で消えている / パスがおかしい場合をスキップ
        if not r_path.exists():
            print(f"  [WARN] ラスターが見つかりませんでした（スキップ）: {r_path}")
            continue

        try:
            with rasterio.open(r_path) as src:
                if base_crs is None:
                    base_crs = src.crs
                else:
                    if src.crs != base_crs:
                        raise RuntimeError(
                            f"CRS が一致しません: {r_path} ({src.crs}) != {base_crs}"
                        )

                # -----------------------------
                # 高速サンプリング：
                #   1) ラスターを一括で配列に読み込み
                #   2) x,y → row,col をベクトルで算出
                #   3) data[:, row, col] からまとめて抽出
                # -----------------------------
                data = src.read()  # shape: (bands, height, width)

                # rasterio.src.index に配列を渡せない環境でも動作するよう、
                # アフィン変換パラメータから行・列をベクトル計算する
                tf = src.transform
                # 環境によって transform のイテレータ長が 6 超になる場合があるので、
                # Affine の属性 or 先頭 6 要素だけを使うようにする
                try:
                    # Affine オブジェクトを想定
                    a, b, c, d, e, f = tf.a, tf.b, tf.c, tf.d, tf.e, tf.f
                except AttributeError:
                    # tuple / list 等のときは先頭 6 要素のみを使用
                    a, b, c, d, e, f = tf[:6]
                if b == 0 and d == 0:
                    # 一般的な north-up ラスター（回転なし）の場合は式で一括計算
                    cols = ((x_arr - c) / a).astype("int64")
                    rows = ((y_arr - f) / e).astype("int64")
                else:
                    # 回転成分を含む特殊なケースではフォールバックとして 1 点ずつ index を呼ぶ
                    rows_list = []
                    cols_list = []
                    for xx, yy in zip(x_arr, y_arr):
                        r, cidx = src.index(float(xx), float(yy))
                        rows_list.append(r)
                        cols_list.append(cidx)
                    rows = np.array(rows_list, dtype="int64")
                    cols = np.array(cols_list, dtype="int64")

                # 範囲外 (out-of-bounds) は NaN にしたいのでまずマスクを作る
                mask_oob = (
                    (rows < 0)
                    | (rows >= src.height)
                    | (cols < 0)
                    | (cols >= src.width)
                )

                # index エラー防止のため、範囲外は一旦 (0,0) に退避
                rows_safe = rows.copy()
                cols_safe = cols.copy()
                rows_safe[mask_oob] = 0
                cols_safe[mask_oob] = 0

                # data[:, rows, cols] → shape: (bands, N) なので転置して (N, bands) に
                vals = data[:, rows_safe, cols_safe].transpose(1, 0)

                # 範囲外は NaN 扱い
                if mask_oob.any():
                    vals[mask_oob, :] = np.nan

                # nodata → NaN に置き換え（この時点で配列側に反映しておく）
                nodata = src.nodata
                if nodata is not None:
                    vals = np.where(vals == nodata, np.nan, vals)

                stem = r_path.stem
                for b in range(vals.shape[1]):
                    col = f"{stem}_b{b+1}"
                    cols_dict[col] = vals[:, b]

        except rasterio.errors.RasterioIOError as e:
            print(
                f"  [WARN] ラスターを開けませんでした（スキップ）: {r_path}\n"
                f"         → {e}"
            )
            continue

        print(f"  [OK] {r_path.name} をサンプリングして特徴量列を追加しました。")

    # まとめて DataFrame に結合（断片化を避ける）
    if cols_dict:
        feat_df = pd.DataFrame(cols_dict, index=df.index)
        df = pd.concat([df, feat_df], axis=1)
        df = df.copy()  # fragmentation を解消

    # ポリゴン属性の付与（任意）
    if poly_for_attr is not None:
        print("\n[INFO] ポリゴン属性の付与:", poly_for_attr)
        import geopandas as gpd
        from shapely.geometry import Point as ShapelyPoint

        poly_gdf = gpd.read_file(poly_for_attr)

        # --- CRS の合わせ込み ---
        # ラスターが 1 枚以上あれば base_crs に揃える。
        # ラスターが無い場合は、ポリゴン側の CRS を基準にする。
        if base_crs is None:
            base_crs = poly_gdf.crs
        else:
            if poly_gdf.crs is None:
                poly_gdf.set_crs(base_crs, inplace=True)
            elif poly_gdf.crs != base_crs:
                poly_gdf = poly_gdf.to_crs(base_crs)

        # 利用する属性列（事前に選択された列を優先）
        attr_candidates = [c for c in poly_gdf.columns if c.lower() not in {"geometry"}]
        if not attr_candidates:
            print("  ⚠ 利用可能な属性列がありません（geometry のみ）。ポリゴン属性の付与はスキップします。")
        else:
            if selected_poly_attr_cols:
                attr_cols = [c for c in selected_poly_attr_cols if c in attr_candidates]
                if not attr_cols:
                    print("  ⚠ 事前に選択された属性列がポリゴンに存在しません。全ての属性列を使用します。")
                    attr_cols = attr_candidates
            else:
                attr_cols = attr_candidates

            print("  使用する属性列:", ", ".join(attr_cols))

            # サンプル点を GeoDataFrame に変換
            df_points = df[["x", "y"]].copy()
            df_points = df_points.dropna(subset=["x", "y"])
            pts_geom = [ShapelyPoint(xy) for xy in zip(df_points["x"], df_points["y"])]
            pts_gdf = gpd.GeoDataFrame(df_points, geometry=pts_geom, crs=base_crs)

            # sjoin でポリゴン属性を付与
            joined = gpd.sjoin(
                pts_gdf,
                poly_gdf[attr_cols + ["geometry"]],
                how="left",
                predicate="intersects",
            )
            drop_cols = [c for c in joined.columns if c in ("index_right",)]
            joined = joined.drop(columns=drop_cols)
            if "geometry" in joined.columns:
                joined = joined.drop(columns=["geometry"])
        # 元の特徴量テーブル df（x, y + ラスター特徴量）に
        # ポリゴン属性をマージする（x, y はグリッドで一意）
        df = df.merge(joined, on=["x", "y"], how="left")
        print(f"  → ポリゴン属性を付与しました（列数: {len(attr_cols)}）")

    # --- 出力 ---
    print("\n[出力]")
    if out_suffix in [".parquet", ".pq"]:
        # x, y 列から Point を生成して GeoParquet として保存
        try:
            crs_epsg = out_crs_epsg or "EPSG:6673"
            save_geoparquet_with_points(
                df,
                Path(out_path),
                x_col="x",
                y_col="y",
                crs_epsg=crs_epsg,
            )
            print(f"✅ GeoParquet を書き出しました: {out_path}")
        except Exception as e:
            print(f"GeoParquet の書き出しに失敗しました: {e}")
    elif out_suffix == ".gpkg":
        # CRS はすでに out_crs_epsg として取得済み（なければ EPSG:4326）
        if out_crs_epsg:
            crs_epsg = out_crs_epsg
        elif base_crs is not None:
            try:
                crs_epsg = base_crs.to_string()
            except Exception:
                crs_epsg = "EPSG:4326"
        else:
            crs_epsg = "EPSG:4326"

        save_gpkg_with_points(
            df,
            out_path,
            x_col="x",
            y_col="y",
            crs_epsg=crs_epsg,
            layer_name="train",
        )
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

def train_mode(backend: str = "rf"):
    """
    backend:
      "rf"  -> scikit-learn RandomForest（CPU）
      "xgb" -> XGBoost（GPU が利用可能なら GPU）
    """
    if backend == "xgb":
        print("\n=== 学習モード（XGBoost / GPU） ===")
    else:
        backend = "rf"
        print("\n=== 学習モード（RandomForest / CPU） ===")
    df, path = load_table_interactive()

    # デバッグ・軽量実験用のサンプル制限（任意）
    total_rows = len(df)
    print(f"[INFO] テーブル行数: {total_rows:,}")
    print("  ※ このあと行うホールドアウト / クロスバリデーションも、")
    print("     ここで絞った行だけを対象にします。")
    print("     本番でしっかり精度を評価したい場合は、空Enterで『全件』を使うことを推奨します。")
    print("     まずは動作確認だけしたい場合に、上限行数を指定してください。")
    max_rows_in = input(
        "学習＋検証に使う行数の上限（空=全件, 例: 50000 や 100000）: "
    ).strip()
    if max_rows_in:
        try:
            max_rows = int(max_rows_in)
            if 0 < max_rows < total_rows:
                df = df.sample(n=max_rows, random_state=42)
                print(f"  → {max_rows:,} 行をランダムサンプリングして学習＋検証に使用します。")
            else:
                print("  → 上限行数が全件以上のため、全件を使用します。")
        except ValueError:
            print("  ⚠ 整数として解釈できなかったため、全件を使用します。")

    # モデル出力先（ルートフォルダ）を先に決めておく
    print("\n[モデル出力設定]")
    base = Path(path)
    base_name = base.stem
    default_root = base.with_suffix("").parent / "rf_models"
    out_root_in = strip_quotes(
        input(f"モデル保存ルートフォルダ（空={default_root}）: ").strip()
    )
    if out_root_in:
        model_root = Path(out_root_in)
    else:
        model_root = default_root
    model_root.mkdir(parents=True, exist_ok=True)

    # ターゲット＋特徴量列（サンプリング後の df に対して実施）
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

    # オプション: 評価結果を地図上で確認するための座標列（x/y）＋間引き率＋EPSG
    xy_cols: tuple[str, str] | None = None
    eval_gpkg_thinning: int | None = None
    eval_crs_epsg: str | None = None
    # ベクタ出力形式フラグ
    save_gpkg: bool = False
    save_parquet: bool = False

    if ask_yes_no(
        "\n評価結果をベクタ（ポイント）として出力しますか？\n"
        "  → GPKG / GeoParquet。テーブル内に x/y 座標列がある場合のみ有効です。",
        default=False,
    ):
        # どの形式を出すか
        save_gpkg, save_parquet = ask_vector_output_format(default="both")

        # ★ ここで間引き率を聞く（LabelEncoder メッセージと [座標列の確認] の間）
        eval_gpkg_thinning = ask_thinning_factor(
            "  → y の場合、間引き率を指定してください"
            "（例: 1=全件, 10=1/10, 1000=1/1000。空=1）: ",
            default=1,
        )
        try:
            # ここから [座標列の確認]
            x_col, y_col = ensure_xy_columns(df)
            xy_cols = (x_col, y_col)
            # x/y 確定の直後で EPSG も聞いて、変数に保持しておく
            eval_crs_epsg = ask_crs(default_epsg=None)
        except Exception as e:
            xy_cols = None
            eval_gpkg_thinning = None
            eval_crs_epsg = None
            save_gpkg = False
            save_parquet = False
            print(
                "  ⚠ x/y 座標列の指定に失敗したため、このセッションでは"
                f"評価用 GPKG の出力をスキップします: {e}"
            )

    X = df[feature_cols].values.astype(float)

    # -------------------------------
    # 検証方法の選択（ホールドアウト / k分割CV）
    # -------------------------------
    print("\n[検証方法の選択]")
    print("  1) ホールドアウト法（学習/テストに分割）")
    print("  2) Monte Carlo Cross-Validation法（モンテカルロCV）")
    print("  3) k-fold Cross-Validation法（k分割CV）")
    val_mode = input("番号を選択してください [1-3]（空=1）: ").strip() or "1"

    # 共通: random_state
    random_state = input("random_state（空=42）: ").strip()
    if random_state:
        try:
            random_state = int(random_state)
        except ValueError:
            print("  ⚠ 整数変換に失敗したので 42 を使います。")
            random_state = 42
    else:
        random_state = 42

    # 共通: 層化を使うかどうか
    use_stratify = ask_yes_no("層化サンプリング / 層化CV を使いますか？", default=True)

    # 共通: 学習曲線を出力するかどうか
    print("\n[オプション] 学習曲線（Learning Curve）の出力設定")
    enable_learning_curve = ask_yes_no(
        "[オプション] 学習曲線の PNG（Accuracy / F1-macro vs 木の本数 n_estimators）を出力しますか？\n"
        "  ※ n_estimators の値に応じて学習に少し時間がかかります。",
        default=False,
    )
    # -------------------------------
    # モデル別パラメータ設定
    # -------------------------------
    # class_weight は現状 RF / XGBoost とも None 固定。
    # （クラス不均衡への対応は、サンプリング設計などで統一的に扱う想定）
    # ※ 将来、クラス重みを対話的に指定したくなったらここを書き換える。
    class_weight = None

    if backend == "xgb":
        print("\n[XGBoost パラメータ設定]")
    else:
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

    if backend == "xgb":
        # XGBoost 用の追加パラメータ
        lr_in = input("learning_rate（学習率, 空=0.1）: ").strip()
        if lr_in:
            try:
                learning_rate = float(lr_in)
            except ValueError:
                print("  ⚠ float 変換に失敗したので 0.1 を使います。")
                learning_rate = 0.1
        else:
            learning_rate = 0.1

        subsample_in = input("subsample（サンプル行の割合, 空=0.8）: ").strip()
        if subsample_in:
            try:
                subsample = float(subsample_in)
            except ValueError:
                print("  ⚠ float 変換に失敗したので 0.8 を使います。")
                subsample = 0.8
        else:
            subsample = 0.8

        colsample_in = input("colsample_bytree（サンプル特徴量の割合, 空=0.8）: ").strip()
        if colsample_in:
            try:
                colsample_bytree = float(colsample_in)
            except ValueError:
                print("  ⚠ float 変換に失敗したので 0.8 を使います。")
                colsample_bytree = 0.8
        else:
            colsample_bytree = 0.8

    if backend == "xgb":
        if XGBClassifier is None:
            print("\n⚠ XGBoost（xgboost）がインポートできません。terrain-env で xgboost をインストールしてください。")
            print("   例: mamba install -c conda-forge xgboost")
            return

        xgb_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth if max_depth is not None else 6,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
        }
        try:
            # xgboost >= 2 系（device 指定スタイル）
            clf = XGBClassifier(
                **xgb_params,
                tree_method="hist",
                device="cuda",
            )
        except TypeError:
            # 旧バージョン向けフォールバック
            clf = XGBClassifier(
                **xgb_params,
                tree_method="gpu_hist",
                predictor="gpu_predictor",
            )

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("xgb", clf),
        ])
    else:
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

    # -------------------------------
    # 実験ID (run_id) と出力ディレクトリ
    # -------------------------------
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id_prefix = "xgb" if backend == "xgb" else "rf"
    run_id = f"{run_id_prefix}_{now}"
    base = Path(path)
    out_dir = model_root / f"{base.stem}_{run_id}"
    model_path = out_dir / f"{run_id_prefix}_model_{run_id}.joblib"
    meta_path  = out_dir / f"{run_id_prefix}_meta_{run_id}.json"
    imp_path   = out_dir / f"{run_id_prefix}_feature_importance_{run_id}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # 検証ロジック
    # -------------------------------
    # 後続のメタ情報作成で必ず参照するので、まずは安全側に初期化しておく
    eval_mode = None
    eval_info = {}

    # ===== Monte Carlo Cross-Validation =====
    if val_mode == "2":
        # ===== Monte Carlo Cross-Validation =====
        print("\n[Monte Carlo Cross-Validation 設定]")
        n_splits_in = input("繰り返し回数 n_splits（空=10）: ").strip()
        if n_splits_in:
            try:
                n_splits = int(n_splits_in)
            except ValueError:
                print("  ⚠ 整数変換に失敗したので 10 を使います。")
                n_splits = 10
        else:
            n_splits = 10

        if n_splits < 2:
            print("  ⚠ n_splits は 2 以上が必要なので 10 を使います。")
            n_splits = 10

        # Monte Carlo でもテスト比率は指定したいので、ここで入力
        test_size_in = input("テストデータ割合 test_size（0〜0.5 程度, 空=0.2）: ").strip()
        if test_size_in:
            try:
                test_size = float(test_size_in)
            except ValueError:
                print("  ⚠ 数値変換に失敗したので 0.2 を使います。")
                test_size = 0.2
        else:
            test_size = 0.2

        # 0 < test_size <= 0.5 の範囲に統一（ホールドアウトと揃える）
        if not (0.0 < test_size <= 0.5):
            print("  ⚠ 0〜0.5 の範囲外なので 0.2 を使います。")
            test_size = 0.2

        # 層化サンプリングの可否をチェック（クラス数や最小サンプル数が少なすぎるとエラーになる）
        if use_stratify:
            unique_mc, counts_mc = np.unique(y, return_counts=True)
            if len(unique_mc) > 1 and np.all(counts_mc >= 2):
                cv = StratifiedShuffleSplit(
                    n_splits=n_splits,
                    test_size=test_size,
                    random_state=random_state,
                )
            else:
                print(
                    "  ⚠ クラス数が 1 つ、または一部クラスのサンプル数が少なすぎるため "
                    "Monte Carlo では層化サンプリングを無効化します。"
                )
                cv = ShuffleSplit(
                    n_splits=n_splits,
                    test_size=test_size,
                    random_state=random_state,
                )
        else:
            cv = ShuffleSplit(
                n_splits=n_splits,
                test_size=test_size,
                random_state=random_state,
            )

        # Monte Carlo と同じ分割設定を用いた学習曲線（任意）
        if enable_learning_curve:
            print("\n[Learning Curve] Monte Carlo CV 設定で学習曲線を計算中...")
            _plot_learning_curve(
                pipe,
                X,
                y,
                cv=cv,
                out_dir=out_dir,
                run_id=run_id,
                run_id_prefix=run_id_prefix,
                eval_mode="montecarlo",
            )

        scoring = {
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
            "f1_weighted": "f1_weighted",
        }
        cv_result = cross_validate(
            pipe, X, y, cv=cv, scoring=scoring,
            n_jobs=-1, return_train_score=False,
        )

        print("\n[評価（Monte Carlo Cross-Validation）]")
        print("  ※ ランダム分割を複数回行い、その平均スコアを表示します。")
        for key, label in [
            ("test_accuracy", "Accuracy"),
            ("test_f1_macro", "F1-macro"),
            ("test_f1_weighted", "F1-weighted"),
        ]:
            vals = cv_result[key]
            print(f"  {label}: {vals.mean():.4f} ± {vals.std():.4f} (n={len(vals)})")

        # Monte Carlo CV では、各サンプルがテストに出る回数が一定でないため、
        # cross_val_predict による OOF 混同行列は作らず、スコアの平均のみ保存。
        # その代わりに「最後の分割」のみを代表として可視化＆必要なら GPKG 出力する。

        last_split = None
        for idx_train_cv, idx_test_cv in cv.split(X, y):
            last_split = (idx_train_cv, idx_test_cv)

        train_gpkg_path = None  # Monte Carlo 用の学習 GPKG パス

        if last_split is not None:
            idx_train_cv, idx_test_cv = last_split
            X_train_cv, X_test_cv = X[idx_train_cv], X[idx_test_cv]
            y_train_cv, y_test_cv = y[idx_train_cv], y[idx_test_cv]

            pipe.fit(X_train_cv, y_train_cv)
            y_pred_cv = pipe.predict(X_test_cv)
            # 学習データ側の予測（train GPKG 用）
            y_pred_train_cv = pipe.predict(X_train_cv)            

            # 混同行列・レポート（最後の分割）
            cm = confusion_matrix(y_test_cv, y_pred_cv)
            print("\n[評価（Monte Carlo: 最終分割）]")
            print("混同行列（行: 真, 列: 予測）:")
            print(cm)
            print("\nclassification_report:")

            if label_encoder_info and class_names:
                all_labels = np.arange(len(class_names))
                print(
                    classification_report(
                        y_test_cv,
                        y_pred_cv,
                        labels=all_labels,
                        target_names=[str(c) for c in class_names],
                        zero_division=0,
                    )
                )
                # Monte Carlo 最終分割の classification_report を CSV でも保存
                try:
                    rep_dict_mc = classification_report(
                        y_test_cv,
                        y_pred_cv,
                        labels=all_labels,
                        target_names=[str(c) for c in class_names],
                        output_dict=True,
                        zero_division=0,
                    )
                    rep_df_mc = pd.DataFrame(rep_dict_mc).T
                    rep_path_mc = out_dir / f"rf_classification_report_mc_last_{run_id}.csv"
                    rep_df_mc.to_csv(rep_path_mc, encoding="utf-8-sig")
                    print(f"  → Monte Carlo 最終分割の classification_report を保存しました: {rep_path_mc}")
                except Exception as e:
                    print(f"  ⚠ Monte Carlo 最終分割の classification_report 保存に失敗しました: {e}")
            else:
                print(classification_report(y_test_cv, y_pred_cv))
                # ラベルエンコーダ未使用の場合も classification_report を CSV で保存
                try:
                    rep_dict_mc = classification_report(
                        y_test_cv,
                        y_pred_cv,
                        output_dict=True,
                        zero_division=0,
                    )
                    rep_df_mc = pd.DataFrame(rep_dict_mc).T
                    rep_path_mc = out_dir / f"rf_classification_report_mc_last_{run_id}.csv"
                    rep_df_mc.to_csv(rep_path_mc, encoding="utf-8-sig")
                    print(f"  → Monte Carlo 最終分割の classification_report を保存しました: {rep_path_mc}")
                except Exception as e:
                    print(f"  ⚠ Monte Carlo 最終分割の classification_report 保存に失敗しました: {e}")

            if class_names is not None:
                cm_labels = [str(c) for c in class_names]
            else:
                cm_labels = [str(i) for i in range(cm.shape[0])]

            # ★ Monte Carlo 用とわかるファイル名に変更
            cm_abs_path = out_dir / f"rf_confusion_matrix_mc_last_{run_id}.png"
            cm_norm_path = out_dir / f"rf_confusion_matrix_mc_last_normalized_{run_id}.png"
            _plot_confusion_matrix(
                cm,
                cm_labels,
                normalize=False,
                title="Confusion matrix (Monte Carlo: last split)",
                save_path=cm_abs_path,
            )
            _plot_confusion_matrix(
                cm,
                cm_labels,
                normalize=True,
                title="Confusion matrix (Monte Carlo: last split, normalized)",
                save_path=cm_norm_path,
            )

            # Monte Carlo: 最終分割の評価結果を GPKG / Parquet で出力（指定間引き対応）
            if xy_cols is not None and eval_gpkg_thinning is not None and (save_gpkg or save_parquet):
                x_col, y_col = xy_cols
                if (x_col in df.columns) and (y_col in df.columns):
                    eval_df = df.iloc[idx_test_cv].copy()
                    # 元のラベルをそのまま true_label として保存
                    eval_df["true_label"] = df[target_col].iloc[idx_test_cv].values

                    # 予測ラベル（LabelEncoder 使用時は decode）
                    if label_encoder_info is not None:
                        try:
                            le_local = LabelEncoder()
                            le_local.classes_ = np.array(label_encoder_info["classes_"])
                            pred_labels = le_local.inverse_transform(y_pred_cv)
                        except Exception:
                            pred_labels = y_pred_cv
                    else:
                        pred_labels = y_pred_cv
                    eval_df["pred_label"] = pred_labels

                    eval_df["is_correct"] = (
                        eval_df["true_label"].astype(str)
                        == eval_df["pred_label"].astype(str)
                    )

                    thinning = eval_gpkg_thinning if eval_gpkg_thinning and eval_gpkg_thinning > 1 else 1
                    if thinning > 1:
                        before_n = len(eval_df)
                        eval_df = eval_df.iloc[::thinning].copy()
                        print(
                            "  → Monte Carlo 最終分割の評価 GPKG を "
                            f"{thinning} 行ごとに 1 行に間引き "
                            f"{before_n} 件 → {len(eval_df)} 件を書き出します。"
                        )

                    # 事前に取得した EPSG を利用（念のため未設定ならここで聞く）
                    crs_epsg = eval_crs_epsg or ask_crs(default_epsg=None)

                    if save_gpkg:
                        eval_gpkg_path = out_dir / f"{base.stem}_eval_mc_last_{run_id}.gpkg"
                        save_gpkg_with_points(
                            eval_df,
                            eval_gpkg_path,
                            x_col=x_col,
                            y_col=y_col,
                            crs_epsg=crs_epsg,
                            layer_name="eval_mc_last",
                        )
                        print(f"[保存] Monte Carlo 最終分割 評価 GPKG: {eval_gpkg_path}")

                # Monte Carlo 最終分割の評価用 GeoParquet も保存（オプション）
                if save_parquet:
                    eval_parquet_path = out_dir / f"{base.stem}_eval_mc_last_{run_id}.parquet"
                    try:
                        save_geoparquet_with_points(
                            eval_df,
                            eval_parquet_path,
                            x_col=x_col,
                            y_col=y_col,
                            crs_epsg=crs_epsg,
                        )
                        print(
                            f"[保存] Monte Carlo 最終分割 評価用 GeoParquet: {eval_parquet_path}"
                        )
                    except Exception as e:
                        print(
                            f"[WARN] Monte Carlo 最終分割 評価用 GeoParquet の書き出しに失敗しました: {e}"
                        )

                # --- 学習データ側（train）の GPKG / Parquet も出力 ---
                    train_df = df.iloc[idx_train_cv].copy()
                    train_df["true_label"] = df[target_col].iloc[idx_train_cv].values

                    if label_encoder_info is not None:
                        try:
                            le_local = LabelEncoder()
                            le_local.classes_ = np.array(label_encoder_info["classes_"])
                            train_pred_labels = le_local.inverse_transform(y_pred_train_cv)
                        except Exception:
                            train_pred_labels = y_pred_train_cv
                    else:
                        train_pred_labels = y_pred_train_cv
                    train_df["pred_label"] = train_pred_labels

                    train_df["is_correct"] = (
                        train_df["true_label"].astype(str)
                        == train_df["pred_label"].astype(str)
                    )

                    thinning_train = eval_gpkg_thinning if eval_gpkg_thinning and eval_gpkg_thinning > 1 else 1
                    if thinning_train > 1:
                        before_n_train = len(train_df)
                        train_df = train_df.iloc[::thinning_train].copy()
                        print(
                            "  → Monte Carlo 最終分割の学習用 GPKG を "
                            f"{thinning_train} 行ごとに 1 行に間引き "
                            f"{before_n_train} 件 → {len(train_df)} 件を書き出します。"
                        )

                    if save_gpkg:
                        train_gpkg_path = out_dir / f"{base.stem}_train_mc_last_{run_id}.gpkg"
                        save_gpkg_with_points(
                            train_df,
                            train_gpkg_path,
                            x_col=x_col,
                            y_col=y_col,
                            crs_epsg=crs_epsg,
                            layer_name="train_mc_last",
                        )
                        print(f"[保存] Monte Carlo 最終分割 学習用 GPKG: {train_gpkg_path}")

                    if save_parquet:
                        # Monte Carlo 最終分割の学習用 Parquet も保存
                        train_parquet_path = out_dir / f"{base.stem}_train_mc_last_{run_id}.parquet"
                        try:
                            save_geoparquet_with_points(
                                train_df,
                                train_parquet_path,
                                x_col=x_col,
                                y_col=y_col,
                                crs_epsg=crs_epsg,
                            )
                            print(f"[保存] Monte Carlo 最終分割 学習用 GeoParquet: {train_parquet_path}")
                        except Exception as e:
                            print(f"[WARN] Monte Carlo 最終分割 学習用 GeoParquet の書き出しに失敗しました: {e}")
            else:
                print(
                    "  ⚠ 評価用 GPKG / Parquet 出力は無効です（座標列が無いか、GPKG / Parquet 出力オプションが OFF です）。"
                )

        # Monte Carlo CV では OOF 混同行列は作らず、
        # スコア平均＋最終分割（上で可視化済み）のみとする
        print("\n[最終モデルの学習（全データで再学習）...]")
        pipe.fit(X, y)
        print("学習完了。")

        eval_mode = "montecarlo"
        eval_info = {
            "test_size": test_size,
            "use_stratify": use_stratify,
            "n_splits": n_splits,
            "scores": {
                "accuracy_mean": float(cv_result["test_accuracy"].mean()),
                "accuracy_std": float(cv_result["test_accuracy"].std()),
                "f1_macro_mean": float(cv_result["test_f1_macro"].mean()),
                "f1_macro_std": float(cv_result["test_f1_macro"].std()),
                "f1_weighted_mean": float(cv_result["test_f1_weighted"].mean()),
                "f1_weighted_std": float(cv_result["test_f1_weighted"].std()),
            },
        }

        # Monte Carlo でも評価 GPKG / 学習用 GPKG を出力した場合は、そのパスをメタ情報に含める
        if eval_gpkg_path is not None:
            eval_info["eval_gpkg_path"] = str(eval_gpkg_path)
        if 'train_gpkg_path' in locals() and train_gpkg_path is not None:
            eval_info["train_gpkg_path"] = str(train_gpkg_path)

    # ===== k-分割クロスバリデーション =====
    elif val_mode == "3":
        print("\n[クロスバリデーション設定]")
        k_in = input("k-分割数（空=5）: ").strip()
        if k_in:
            try:
                k_splits = int(k_in)
            except ValueError:
                print("  ⚠ 整数変換に失敗したので 5 を使います。")
                k_splits = 5
        else:
            k_splits = 5

        if k_splits < 2:
            print("  ⚠ k は 2 以上が必要なので 5 を使います。")
            k_splits = 5

        if use_stratify:
            # 層化k-foldが成立するかチェック
            #  - クラスが2種類以上
            #  - 全クラスでサンプル数 >= k_splits
            unique_cv, counts_cv = np.unique(y, return_counts=True)
            if len(unique_cv) > 1 and np.all(counts_cv >= k_splits):
                cv = StratifiedKFold(
                    n_splits=k_splits, shuffle=True, random_state=random_state
                )
            else:
                print(
                    "  ⚠ クラス数が 1 つ、または一部クラスのサンプル数が "
                    f"k={k_splits} 未満のため、層化 k-fold を無効化します。"
                )
                cv = KFold(
                    n_splits=k_splits, shuffle=True, random_state=random_state
                )
        else:
            cv = KFold(
                n_splits=k_splits, shuffle=True, random_state=random_state
            )

        # k-fold と同じ分割設定を用いた学習曲線（任意）
        if enable_learning_curve:
            print("\n[Learning Curve] k-fold CV 設定で学習曲線を計算中...")
            _plot_learning_curve(
                pipe,
                X,
                y,
                cv=cv,
                out_dir=out_dir,
                run_id=run_id,
                run_id_prefix=run_id_prefix,
                eval_mode="kfold",
            )

        print("\n[クロスバリデーション中...]")
        scoring = {
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
            "f1_weighted": "f1_weighted",
        }
        cv_result = cross_validate(
            pipe, X, y, cv=cv, scoring=scoring,
            n_jobs=-1, return_train_score=False,
        )

        print("\n[評価（OOF = out-of-fold 予測）]")
        print("  ※ データ全体を対象に、『そのサンプルを学習に使っていないモデル』の予測だけで評価しています。")
        print("     → 全サンプルが一度ずつテストに回る、厳しめの精度評価です。")
        for key, label in [
            ("test_accuracy", "Accuracy"),
            ("test_f1_macro", "F1-macro"),
            ("test_f1_weighted", "F1-weighted"),
        ]:
            vals = cv_result[key]
            print(f"  {label}: {vals.mean():.4f} ± {vals.std():.4f} (n={len(vals)})")

        # OOF 予測を使った評価（混同行列・classification_report）
        print("\n[評価（クロスバリデーションの out-of-fold 予測）]")
        y_oof = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
        cm = confusion_matrix(y, y_oof)
        print("混同行列（行: 真, 列: 予測）:")
        print(cm)
        print("\nclassification_report:")

        # target_names を安全に決定
        target_names = None
        if label_encoder_info and class_names:
            target_names = [str(c) for c in class_names]
        elif class_names is not None:
            target_names = [str(c) for c in class_names]

        if target_names is not None:
            print(classification_report(y, y_oof, target_names=target_names))
            # CV OOF の classification_report を CSV でも保存
            try:
                rep_dict_cv = classification_report(
                    y,
                    y_oof,
                    target_names=target_names,
                    output_dict=True,
                    zero_division=0,
                )
                rep_df_cv = pd.DataFrame(rep_dict_cv).T
                rep_path_cv = out_dir / f"rf_classification_report_cv_oof_{run_id}.csv"
                rep_df_cv.to_csv(rep_path_cv, encoding="utf-8-sig")
                print(f"  → CV OOF の classification_report を保存しました: {rep_path_cv}")
            except Exception as e:
                print(f"  ⚠ CV OOF の classification_report 保存に失敗しました: {e}")
        else:
            print(classification_report(y, y_oof))
            # target_names 無しの場合も classification_report を CSV で保存
            try:
                rep_dict_cv = classification_report(
                    y,
                    y_oof,
                    output_dict=True,
                    zero_division=0,
                )
                rep_df_cv = pd.DataFrame(rep_dict_cv).T
                rep_path_cv = out_dir / f"rf_classification_report_cv_oof_{run_id}.csv"
                rep_df_cv.to_csv(rep_path_cv, encoding="utf-8-sig")
                print(f"  → CV OOF の classification_report を保存しました: {rep_path_cv}")
            except Exception as e:
                print(f"  ⚠ CV OOF の classification_report 保存に失敗しました: {e}")

        # 混同行列の図も保存（OOF ベース）
        if class_names is not None:
            cm_labels = [str(c) for c in class_names]
        else:
            cm_labels = [str(i) for i in range(cm.shape[0])]

        cm_abs_path = out_dir / f"rf_confusion_matrix_cv_oof_{run_id}.png"
        cm_norm_path = (
            out_dir / f"rf_confusion_matrix_cv_oof_normalized_{run_id}.png"
        )
        _plot_confusion_matrix(
            cm,
            cm_labels,
            normalize=False,
            title="Confusion matrix (CV oof)",
            save_path=cm_abs_path,
        )
        _plot_confusion_matrix(
            cm,
            cm_labels,
            normalize=True,
            title="Confusion matrix (normalized, CV oof)",
            save_path=cm_norm_path,
        )

        # オプション: クロスバリデーション OOF 予測を GPKG / GeoParquet（ポイント）として保存
        if xy_cols is not None and (save_gpkg or save_parquet):
            x_col, y_col = xy_cols
            if (x_col in df.columns) and (y_col in df.columns):
                eval_df = df.copy()
                eval_df["true_label"] = df[target_col].values
                if label_encoder_info is not None:
                    # y_oof はエンコード後の整数ラベルなので decode して保存
                    pred_labels = le.inverse_transform(y_oof)
                else:
                    pred_labels = y_oof
                eval_df["pred_label"] = pred_labels
                eval_df["is_correct"] = (
                    eval_df["true_label"] == eval_df["pred_label"]
                )

                # 事前に指定した eval_gpkg_thinning を適用（1 なら全件）
                if eval_gpkg_thinning and eval_gpkg_thinning > 1:
                    before_n = len(eval_df)
                    eval_df = eval_df.iloc[::eval_gpkg_thinning].copy()
                    print(
                        "  → CV OOF 評価 GPKG を "
                        f"{eval_gpkg_thinning} 行ごとに 1 行に間引き "
                        f"{before_n} 件 → {len(eval_df)} 件を書き出します。"
                    )

                print(
                    "\n[オプション] CV OOF 評価結果を GPKG / GeoParquet として保存します。"
                )

                # 事前に取得した EPSG を利用（念のため未設定ならここで聞く）
                crs_epsg = eval_crs_epsg or ask_crs(default_epsg=None)

                if save_gpkg:
                    eval_cv_gpkg_path = (
                        out_dir / f"{base.stem}_eval_cv_oof_{run_id}.gpkg"
                    )
                    save_gpkg_with_points(
                        eval_df,
                        eval_cv_gpkg_path,
                        x_col=x_col,
                        y_col=y_col,
                        crs_epsg=crs_epsg,
                        layer_name="eval_cv_oof",
                    )
                    print(f"[保存] CV OOF 評価 GPKG: {eval_cv_gpkg_path}")

                if save_parquet:
                    # CV OOF 評価用 GeoParquet も保存
                    eval_parquet_path = out_dir / f"{base.stem}_eval_cv_oof_{run_id}.parquet"
                    try:
                        save_geoparquet_with_points(
                            eval_df,
                            eval_parquet_path,
                            x_col=x_col,
                            y_col=y_col,
                            crs_epsg=crs_epsg,
                        )
                        print(
                            f"[保存] CV OOF 評価用 GeoParquet: {eval_parquet_path}"
                        )
                    except Exception as e:
                        print(
                            f"[WARN] CV OOF 評価用 GeoParquet の書き出しに失敗しました: {e}"
                        )
            else:
                eval_cv_gpkg_path = None
                print(
                    "  ⚠ 指定された座標列が df に存在しないため、CV 用 GPKG / Parquet の出力をスキップします。"
                )
        else:
            eval_cv_gpkg_path = None
            print(
                "  ⚠ 評価用 GPKG / Parquet 出力は無効です（座標列が無いか、GPKG / Parquet 出力オプションが OFF です）。"
            )

        # 最終モデルは全データでフィット
        print("\n[最終モデルの学習（全データで再学習）...]")
        pipe.fit(X, y)
        print("学習完了。")

        eval_mode = "kfold"
        eval_info = {
            "k_splits": k_splits,
            "use_stratify": use_stratify,
        }
        if eval_cv_gpkg_path is not None:
            eval_info["eval_gpkg_path"] = str(eval_cv_gpkg_path)

    # ===== ホールドアウト法 =====
    else:
        print("\n[学習データ分割設定（ホールドアウト）]")
        test_size = input("テストデータ割合（0〜0.5 程度, 空=0.2）: ").strip()
        if test_size:
            try:
                test_size = float(test_size)
            except ValueError:
                print("  ⚠ 数値変換に失敗したので 0.2 を使います。")
                test_size = 0.2
        else:
            test_size = 0.2

        # 0 < test_size <= 0.5 の範囲に統一（Monte Carlo と揃える）
        if not (0.0 < test_size <= 0.5):
            print("  ⚠ 0〜0.5 の範囲外なので 0.2 を使います。")
            test_size = 0.2

        if use_stratify:
            # 層化サンプリングが有効な場合のみ stratify を指定
            # （クラスが1種類だけ、あるいは極端に少ないとエラーになるため）
            unique, counts = np.unique(y, return_counts=True)
            if len(unique) > 1 and np.all(counts >= 2):
                stratify_arr = y
            else:
                print("  ⚠ クラス数が1つ、または極端に少ないため層化サンプリングは無効化します。")
                stratify_arr = None
        else:
            stratify_arr = None

        # --- 学習曲線（ホールドアウト設定に基づく ShuffleSplit） ---
        if enable_learning_curve:
            print("\n[Learning Curve] ホールドアウト設定に基づく学習曲線を計算中...")
            if use_stratify and stratify_arr is not None:
                cv_lc = StratifiedShuffleSplit(
                    n_splits=5,
                    test_size=test_size,
                    random_state=random_state,
                )
            else:
                cv_lc = ShuffleSplit(
                    n_splits=5,
                    test_size=test_size,
                    random_state=random_state,
                )

            _plot_learning_curve(
                pipe,
                X,
                y,
                cv=cv_lc,
                out_dir=out_dir,
                run_id=run_id,
                run_id_prefix=run_id_prefix,
                eval_mode="holdout",
            )

        indices = np.arange(len(df))
        idx_train, idx_test = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arr,
        )

        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        print("\n[学習中...]")
        pipe.fit(X_train, y_train)
        print("学習完了。")

        # 予測
        y_pred = pipe.predict(X_test)
        # 学習データ側の予測（train GPKG 用）
        y_pred_train = pipe.predict(X_train)

        # 混同行列・レポート
        cm = confusion_matrix(y_test, y_pred)
        print("混同行列（行: 真, 列: 予測）:")
        print(cm)
        print("\nclassification_report:")

        if label_encoder_info and class_names:
            all_labels = np.arange(len(class_names))
            print(
                classification_report(
                    y_test,
                    y_pred,
                    labels=all_labels,
                    target_names=[str(c) for c in class_names],
                    zero_division=0,
                )
            )
            # ホールドアウト検証の classification_report を CSV でも保存
            try:
                rep_dict_ho = classification_report(
                    y_test,
                    y_pred,
                    labels=all_labels,
                    target_names=[str(c) for c in class_names],
                    output_dict=True,
                    zero_division=0,
                )
                rep_df_ho = pd.DataFrame(rep_dict_ho).T
                rep_path_ho = out_dir / f"rf_classification_report_ho_{run_id}.csv"
                rep_df_ho.to_csv(rep_path_ho, encoding="utf-8-sig")
                print(f"  → ホールドアウト検証の classification_report を保存しました: {rep_path_ho}")
            except Exception as e:
                print(f"  ⚠ ホールドアウト検証の classification_report 保存に失敗しました: {e}")
        else:
            print(classification_report(y_test, y_pred))
            # ラベルエンコーダ未使用の場合も classification_report を CSV で保存
            try:
                rep_dict_ho = classification_report(
                    y_test,
                    y_pred,
                    output_dict=True,
                    zero_division=0,
                )
                rep_df_ho = pd.DataFrame(rep_dict_ho).T
                rep_path_ho = out_dir / f"rf_classification_report_ho_{run_id}.csv"
                rep_df_ho.to_csv(rep_path_ho, encoding="utf-8-sig")
                print(f"  → ホールドアウト検証の classification_report を保存しました: {rep_path_ho}")
            except Exception as e:
                print(f"  ⚠ ホールドアウト検証の classification_report 保存に失敗しました: {e}")

        # 混同行列の図も保存（ホールドアウト）
        if class_names is not None:
            cm_labels = [str(c) for c in class_names]
        else:
            cm_labels = [str(i) for i in range(cm.shape[0])]

        suffix = "holdout"
        cm_abs_path = out_dir / f"rf_confusion_matrix_{suffix}_{run_id}.png"
        cm_norm_path = out_dir / f"rf_confusion_matrix_{suffix}_normalized_{run_id}.png"
        _plot_confusion_matrix(
            cm,
            cm_labels,
            normalize=False,
            title=f"Confusion matrix ({suffix})",
            save_path=cm_abs_path,
        )
        _plot_confusion_matrix(
            cm,
            cm_labels,
            normalize=True,
            title=f"Confusion matrix (normalized, {suffix})",
            save_path=cm_norm_path,
        )

        # オプション: ホールドアウト法のテストデータを評価用 GPKG として出力（指定間引き対応）
        eval_gpkg_path = None
        train_gpkg_path = None
        if xy_cols is not None and eval_gpkg_thinning is not None:
            # x,y 列名の取り出し
            if isinstance(xy_cols, tuple):
                x_col, y_col = xy_cols
            else:
                x_col, y_col = xy_cols[0], xy_cols[1]

            if (x_col in df.columns) and (y_col in df.columns):
                # テストデータ部分のみ抽出
                eval_df = df.iloc[idx_test].copy()
                # 元のラベルをそのまま true_label として保存
                eval_df["true_label"] = df[target_col].iloc[idx_test].values

                # 予測ラベル（LabelEncoder 使用時は decode）
                if label_encoder_info is not None:
                    try:
                        le_local = LabelEncoder()
                        le_local.classes_ = np.array(label_encoder_info["classes_"])
                        pred_labels = le_local.inverse_transform(y_pred)
                    except Exception:
                        pred_labels = y_pred
                else:
                    pred_labels = y_pred
                eval_df["pred_label"] = pred_labels

                # 正誤フラグ
                eval_df["is_correct"] = (
                    eval_df["true_label"].astype(str)
                    == eval_df["pred_label"].astype(str)
                )

                thinning = (
                    eval_gpkg_thinning if eval_gpkg_thinning and eval_gpkg_thinning > 1 else 1
                )
                if thinning > 1:
                    before_n = len(eval_df)
                    eval_df = eval_df.iloc[::thinning].copy()
                    print(
                        "  → ホールドアウト評価 GPKG を "
                        f"{thinning} 行ごとに 1 行に間引き "
                        f"{before_n} 件 → {len(eval_df)} 件を書き出します。"
                    )

                # 事前に取得した EPSG を利用（念のため未設定ならここで聞く）
                crs_epsg = eval_crs_epsg or ask_crs(default_epsg=None)

                if save_gpkg:
                    eval_gpkg_path = out_dir / f"{base.stem}_eval_holdout_{run_id}.gpkg"
                    save_gpkg_with_points(
                        eval_df,
                        eval_gpkg_path,
                        x_col=x_col,
                        y_col=y_col,
                        crs_epsg=crs_epsg,
                        layer_name="eval_holdout",
                    )
                    print(f"[保存] ホールドアウト評価用 GPKG: {eval_gpkg_path}")

                if save_parquet:
                    # 評価用 GeoParquet（geometry 付き）も保存
                    eval_parquet_path = out_dir / f"{base.stem}_eval_holdout_{run_id}.parquet"
                    try:
                        save_geoparquet_with_points(
                            eval_df,
                            eval_parquet_path,
                            x_col=x_col,
                            y_col=y_col,
                            crs_epsg=crs_epsg,
                        )
                        print(f"[保存] ホールドアウト評価用 GeoParquet: {eval_parquet_path}")
                    except Exception as e:
                        print(f"[WARN] ホールドアウト評価用 GeoParquet の書き出しに失敗しました: {e}")

                # --- 学習データ側（train）の GPKG も出力 ---
                train_df = df.iloc[idx_train].copy()
                train_df["true_label"] = df[target_col].iloc[idx_train].values

                if label_encoder_info is not None:
                    try:
                        le_local = LabelEncoder()
                        le_local.classes_ = np.array(label_encoder_info["classes_"])
                        train_pred_labels = le_local.inverse_transform(y_pred_train)
                    except Exception:
                        train_pred_labels = y_pred_train
                else:
                    train_pred_labels = y_pred_train
                train_df["pred_label"] = train_pred_labels

                train_df["is_correct"] = (
                    train_df["true_label"].astype(str)
                    == train_df["pred_label"].astype(str)
                )

                thinning_train = (
                    eval_gpkg_thinning if eval_gpkg_thinning and eval_gpkg_thinning > 1 else 1
                )
                if thinning_train > 1:
                    before_n_train = len(train_df)
                    train_df = train_df.iloc[::thinning_train].copy()
                    print(
                        "  → ホールドアウト学習用 GPKG を "
                        f"{thinning_train} 行ごとに 1 行に間引き "
                        f"{before_n_train} 件 → {len(train_df)} 件を書き出します。"
                    )

                if save_gpkg:
                    train_gpkg_path = out_dir / f"{base.stem}_train_holdout_{run_id}.gpkg"
                    save_gpkg_with_points(
                        train_df,
                        train_gpkg_path,
                        x_col=x_col,
                        y_col=y_col,
                        crs_epsg=crs_epsg,
                        layer_name="train_holdout",
                    )
                    print(f"[保存] ホールドアウト学習用 GPKG: {train_gpkg_path}")

                if save_parquet:
                    # 学習データ側の GeoParquet も保存（geometry 付き）
                    train_parquet_path = out_dir / f"{base.stem}_train_holdout_{run_id}.parquet"
                    try:
                        save_geoparquet_with_points(
                            train_df,
                            train_parquet_path,
                            x_col=x_col,
                            y_col=y_col,
                            crs_epsg=crs_epsg,
                        )
                        print(f"[保存] ホールドアウト学習用 GeoParquet: {train_parquet_path}")
                    except Exception as e:
                        print(
                            f"[WARN] ホールドアウト学習用 GeoParquet の書き出しに失敗しました: {e}"
                        )
            else:
                print(
                    "  ⚠ 指定された座標列が df に存在しないため、ホールドアウト評価用 "
                    "GPKG の出力をスキップします。"
                )
        else:
            print(
                "  ⚠ 評価用 GPKG 出力は無効です（座標列が無いか、GPKG 出力オプションが OFF です）。"
            )
        eval_mode = "holdout"
        eval_info = {
            "test_size": test_size,
            "use_stratify": use_stratify,
        }
        if eval_gpkg_path is not None:
            eval_info["eval_gpkg_path"] = str(eval_gpkg_path)
        if 'train_gpkg_path' in locals() and train_gpkg_path is not None:
            eval_info["train_gpkg_path"] = str(train_gpkg_path)

    # モデル保存パラメータまとめ
    train_params = {
        "backend": "xgb" if backend == "xgb" else "rf",
        "random_state": random_state,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }
    if backend == "xgb":
        train_params.update({
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
        })
    else:
        train_params["class_weight"] = class_weight
    # Monte Carlo も test_size を持つので、メタ情報に含める
    if eval_mode in ("holdout", "montecarlo"):
        train_params["test_size"] = test_size

    # メタ情報 JSON 本体
    meta = {
        "run_id": run_id,
        "source_path": str(path),
        "target_col": target_col,
        "feature_cols": feature_cols,
        "class_names": class_names,
        "label_encoder": label_encoder_info,
        "validation": {
            "mode": eval_mode,
            **eval_info,
        },
        "train_params": train_params,
    }

    # 既存: run_id 付きのファイル
    joblib.dump(pipe, model_path)
    print(f"\n[保存] モデル: {model_path}")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[保存] メタ情報: {meta_path}")

    # オプション: 直近モデルへのショートカット
    latest_model = model_root / "rf_model.joblib"
    latest_meta  = model_root / "rf_meta.json"
    joblib.dump(pipe, latest_model)
    with open(latest_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[保存] 直近モデル: {latest_model}")

    # 特徴量重要度
    if backend == "xgb":
        est = pipe.named_steps.get("xgb")
        latest_imp = model_root / "xgb_feature_importance.csv"
    else:
        est = pipe.named_steps.get("rf")
        latest_imp = model_root / "rf_feature_importance.csv"

    if est is not None and hasattr(est, "feature_importances_"):
        importances = est.feature_importances_
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
        fig_path = out_dir / f"{run_id_prefix}_feature_importance_{run_id}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"[保存] 特徴量重要度 図: {fig_path}")

        imp_df.to_csv(latest_imp, index=False, encoding="utf-8")
        print(f"[保存] 直近特徴量重要度 CSV: {latest_imp}")
    else:
        print("\n[INFO] このモデルでは feature_importances_ が利用できないため、特徴量重要度の出力をスキップしました。")

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


def _predict_in_chunks(pipe, df, feature_cols, chunk_size: int | None):
    """
    メモリを抑えるため、チャンク単位で予測を行うヘルパー。
    - pipe         : 学習済み Pipeline
    - df           : 入力 DataFrame（特徴量列をすでに含む）
    - feature_cols : 特徴量列名リスト
    - chunk_size   : 1 回に処理する最大行数（None または 0 以下なら全件）
    """
    n_rows = len(df)
    if n_rows == 0:
        return np.empty((0,), dtype=float), None

    if not chunk_size or chunk_size <= 0 or chunk_size >= n_rows:
        chunk_size = n_rows

    use_proba = hasattr(pipe, "predict_proba")
    y_list = []
    proba_list = [] if use_proba else None

    print(f"\n[INFO] チャンク予測を実行します（チャンクサイズ: {chunk_size:,} 行）")

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        print(f"  - 行 {start:,} ～ {end-1:,} を予測中...", end="", flush=True)

        X_chunk = df[feature_cols].iloc[start:end].to_numpy(dtype=np.float32)
        y_chunk = pipe.predict(X_chunk)
        y_list.append(y_chunk)

        if use_proba:
            proba_chunk = pipe.predict_proba(X_chunk)
            proba_list.append(proba_chunk.max(axis=1))

        print(" done")

    y_pred = np.concatenate(y_list, axis=0)
    if use_proba:
        proba_max = np.concatenate(proba_list, axis=0)
    else:
        proba_max = None

    return y_pred, proba_max


def predict_mode():
    print("\n=== 予測モード（predict） ===")
    pipe, meta, model_path, meta_path = load_model_and_meta()

    feature_cols = meta["feature_cols"]
    target_col = meta["target_col"]
    class_names = meta.get("class_names")
    label_encoder_info = meta.get("label_encoder")

    # まずは入力テーブルのパスだけを取得
    in_path = input("入力テーブルのパス（CSV/Parquet/GPKG）: ").strip().strip('"')
    if not in_path:
        print("  ⚠ 入力テーブルのパスが空です。中断します。")
        return
    if not os.path.exists(in_path):
        print(f"  ⚠ 入力ファイルが見つかりません: {in_path}")
        return

    # テーブル全体の行数・列数を inspect_table で軽量に確認
    cols_meta, total_rows = inspect_table(in_path)
    if total_rows > 0:
        print(f"[INFO] テーブル行数: {total_rows:,}（inspect_table による推定）")
        if cols_meta:
            print(f"[INFO] 列数: {len(cols_meta)} 列（inspect_table）")
    else:
        print("[WARN] inspect_table でテーブル行数を取得できませんでした。読み込み後に行数を算出します。")
        total_rows = None

    # 何行予測するかをここで決める（GPKG の場合は random_sample_n によるランダムサンプリング）
    print("     まずは動作確認だけしたい場合に、ここで行数の上限を指定してください。")
    print("     GPKG の場合は全域からランダムサンプリング（random_sample_n）されます。")
    max_rows = None
    max_rows_in = input(
        "予測結果を出力する行数の上限（空=全件, 例: 50000 や 100000）: "
    ).strip()
    if max_rows_in:
        try:
            req = int(max_rows_in)
            if req <= 0:
                print("  → 0 以下は指定できません。全件を対象に予測します。")
            else:
                if total_rows is not None and req >= total_rows:
                    print("  → 上限行数が全件以上のため、全件を対象に予測します。")
                else:
                    max_rows = req
        except ValueError:
            print("  ⚠ 整数として解釈できなかったため、全件を対象に予測します。")

    # 実際のテーブル読み込み（GPKG の場合は RANDOM()＋LIMIT によるランダムサンプリング）
    df, is_random_sample, is_geopq = load_table_for_predict(
        in_path,
        max_rows=max_rows,
        random_state=42,
    )
    if is_geopq:
        print("[INFO] 入力テーブルは GeoParquet 由来です（geometry は除外済み）。")
    if total_rows is None:
        total_rows = len(df)
        print(f"[INFO] テーブル行数: {total_rows:,}")

    if max_rows is None:
        print(f"  → 全 {len(df):,} 行を対象に予測します。")
    else:
        if is_random_sample:
            print(f"  → {len(df):,} 行をランダムサンプリングして予測します。")
            print("     出力される予測ファイルにも、このサンプルされた行のみが含まれます。")
        else:
            print(f"  → {len(df):,} 行のみを読み込んで予測します（先頭から順に抽出）。")
            print("     出力される予測ファイルにも、この抽出された行のみが含まれます。")

    # 評価結果の出力用に run_id だけ先に取得
    # （実際の eval_prefix は、予測出力 out_path が決まってから組み立てる）
    base_in = str(Path(in_path).with_suffix(""))
    run_id = meta.get("run_id", "predict")

    # =====================================================
    # 予測結果の保存先を「最初に」決めておく
    # =====================================================
    low_in = in_path.lower()
    if low_in.endswith(".csv"):
        default_out_path = f"{base_in}_pred_{run_id}.csv"
    elif low_in.endswith(".parquet") or low_in.endswith(".pq"):
        default_out_path = f"{base_in}_pred_{run_id}.parquet"
    elif low_in.endswith(".gpkg"):
        default_out_path = f"{base_in}_pred_{run_id}.gpkg"
    else:
        default_out_path = f"{base_in}_pred_{run_id}.csv"

    print("\n[出力先の設定]")
    print(f"  デフォルト: {default_out_path}")
    out_path = input(
        "予測結果の保存先パス（空=上記デフォルト。拡張子を含めて指定可。"
        "ディレクトリのみ指定した場合は、その中にデフォルト名で保存）: "
    ).strip().strip('"').strip("'")
    if not out_path:
        out_path = default_out_path
    else:
        # ユーザーがディレクトリを指定した場合の扱い
        p = Path(out_path)
        # 既存ディレクトリ、または末尾が / or \ の場合は「ディレクトリ指定」とみなす
        if p.is_dir() or out_path.endswith(("/", "\\")):
            out_path = str(p / Path(default_out_path).name)
            print(f"  → ディレクトリ指定と判断し、{out_path} に保存します。")
    print(f"  → 予測結果は {out_path} に保存されます。")
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ここまでで out_path（予測結果を書き出すパス）が決まっている前提
    # 評価結果の出力用プレフィックスも、その out_path と同じディレクトリに揃える
    #   例）/.../predict/B/train_MeB_pred_rf_xxx.gpkg
    #        → /.../predict/B/train_MeB_pred_rf_xxx_eval_confusion_matrix.csv など
    base_out = str(Path(out_path).with_suffix(""))
    eval_prefix = f"{base_out}_eval"

    # 入力GPKGのCRSを取得（あれば、後の ask_crs の既定値に使う）
    source_crs_str = None    # ★ dfは属性のみ読み込み済みなので、ここで別途読む
    if in_path.lower().endswith(".gpkg"):
        try:
            gdf_src = _read_gpkg_with_geom(in_path)
            if gdf_src.crs is not None:
                source_crs_str = gdf_src.crs.to_string()
                print(f"[INFO] 入力GPKGのCRS: {source_crs_str}")
        except Exception as e:
            print(f"[WARN] 入力GPKGのCRS取得に失敗しました: {e}")

    # =====================================================
    # モデル側クラスと入力テーブル側ラベルの「事前確認」
    # =====================================================
    print("\n[モデル側クラス一覧（このモデルが出力し得るラベル）]")
    if class_names:
        for i, cname in enumerate(class_names):
            print(f"  [{i:02d}] {cname}")
    else:
        print("  （meta に class_names が無いため、クラス一覧は表示できません）")

    if target_col in df.columns:
        data_labels = sorted(set(df[target_col].dropna().astype(str)))
        print("\n[入力テーブル側のラベル一覧（ユニーク）]")
        for v in data_labels:
            print(f"  - {v}")

        if class_names:
            model_set = set(str(c) for c in class_names)
            data_set = set(data_labels)

            missing_in_data = sorted(model_set - data_set)
            extra_in_data   = sorted(data_set - model_set)

            print("\n[ラベル差分チェック]")
            if missing_in_data:
                print("  ・モデルには存在するが、このテーブルには出現していないラベル:")
                for v in missing_in_data:
                    print(f"      - {v}")
            else:
                print("  ・モデルにあって、このテーブルに無いラベル: なし")

            if extra_in_data:
                print("  ・テーブル側にあるが、モデルが学習していないラベル:")
                for v in extra_in_data:
                    print(f"      - {v}")
            else:
                print("  ・テーブル側にあって、モデルに無いラベル: なし")
        else:
            print("\n[注意] meta に class_names が無いため、差分の機械的チェックは行いません。")

        input("\nラベル一覧と差分を確認しました。続行するには Enter を押してください（中止=Ctrl+C）: ")
    else:
        print(f"\n[注意] 入力テーブルに target_col='{target_col}' 列が無いため、")
        print("       今回は評価（混同行列・classification_report）は行われません。")
        input("モデルのクラス一覧だけ確認しました。続行するには Enter を押してください（中止=Ctrl+C）: ")

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

    # -----------------------------------------------------
    # チャンクサイズの設定（大規模データ向け）
    # -----------------------------------------------------
    print("\n[予測設定: チャンクサイズ]")
    print("大量データの場合、チャンク単位で予測してメモリ使用量を抑えます。")
    print("空 Enter でデフォルト値（200,000 行）を使用します。")

    default_chunk_size = 200_000
    chunk_size = default_chunk_size
    ans_chunk = input(
        f"一度に処理する最大行数（例: 100000 / 200000 / 500000、空={default_chunk_size:,}）: "
    ).strip()
    if ans_chunk:
        try:
            v = int(ans_chunk)
            if v > 0:
                chunk_size = v
            else:
                print("  ⚠ 0 以下は指定できないため、デフォルト値を使用します。")
        except ValueError:
            print("  ⚠ 整数として解釈できないため、デフォルト値を使用します。")

    # -----------------------------------------------------
    # チャンク単位で予測
    # -----------------------------------------------------
    print("\n[予測中...]")
    y_pred, proba_max = _predict_in_chunks(pipe, df, saved_feats, chunk_size)

    # ラベルを元に戻す（予測側）
    if label_encoder_info and class_names:
        y_pred_labels = [class_names[int(i)] for i in y_pred]
    else:
        y_pred_labels = y_pred

    out = df.copy()
    out["y_pred"] = y_pred_labels
    if proba_max is not None:
        out["proba_max"] = proba_max

        # 予測確率の閾値による信頼度マスク列（オプション）
        if ask_yes_no(
            "\n[オプション] 予測確率の閾値で信頼度マスク列（is_confident / y_pred_masked）を追加しますか？",
            default=False,
        ):
            thr_in = input("  → 閾値（0〜1, 空=0.5 例: 0.7）: ").strip()
            try:
                thr = float(thr_in) if thr_in else 0.5
            except ValueError:
                print("  ⚠ 数値として解釈できなかったため 0.5 を使用します。")
                thr = 0.5
            # 0〜1 にクリップ
            if thr < 0.0:
                thr = 0.0
            elif thr > 1.0:
                thr = 1.0

            out["is_confident"] = out["proba_max"] >= thr
            # 閾値未満は欠損値としてマスク
            out["y_pred_masked"] = out["y_pred"].where(out["is_confident"])

    saved_eval = False  # 評価結果をファイル出力したかどうか

    # =====================================================
    # 正解ラベルがあれば評価 ＆ ラベル対応チェック
    # =====================================================
    if target_col in df.columns:
        print("\n[評価（入力テーブルに正解ラベルが含まれていたため）]")
        y_true_raw = df[target_col].values

        # ---------------------------
        # LabelEncoder を使っていた場合
        # ---------------------------
        if label_encoder_info and class_names:
            # --- 学習時クラスの一覧 ---
            print("\n[学習時クラス一覧]")
            for i, cname in enumerate(class_names):
                print(f"  [{i:02d}] {cname}")

            # 学習時クラス名 → クラスIDの辞書（文字列ベース＋int文字列ベース）
            norm_to_idx: dict[str, int] = {}
            for idx, name in enumerate(class_names):
                s = str(name).strip()
                norm_to_idx[s] = idx
                try:
                    iv = int(s)
                    norm_to_idx[str(iv)] = idx
                except Exception:
                    pass

            # 予測テーブル側ラベル（生のユニーク値）
            raw_labels = sorted({str(v).strip() for v in y_true_raw})
            print("\n[評価対象テーブル側のラベル一覧（ユニーク）]")
            for s in raw_labels:
                print(f"  - {s}")

            # 自動では拾えないラベル
            unmatched = [lab for lab in raw_labels if lab not in norm_to_idx]

            if unmatched:
                print("\n⚠ 自動マッピングでは対応が見つからないラベルがあります。")
                print("  対応させたい学習クラスがあれば、番号を指定してください。")
                print("  何も入力せずEnter → 評価から除外（その行は無視）")
                print("")
                for lab in unmatched:
                    print(f"  未対応ラベル: {lab}")
                    ans = input("    対応する学習クラス番号（空=除外）: ").strip()
                    if ans == "":
                        print("    → このラベルは評価から除外します。")
                        continue
                    try:
                        idx = int(ans)
                        if 0 <= idx < len(class_names):
                            target_name = class_names[idx]
                            print(f"    → {lab} を '{target_name}' として扱います。")
                            norm_to_idx[lab] = idx
                        else:
                            print("    ⚠ 範囲外の番号です。このラベルは除外します。")
                    except ValueError:
                        print("    ⚠ 整数として解釈できません。このラベルは除外します。")

            # y_true_raw → クラスID（int）に変換（マッピングできない行は -1 として除外）
            y_true_int = np.full(len(y_true_raw), -1, dtype=int)
            for i, v in enumerate(y_true_raw):
                s = str(v).strip()
                if s in norm_to_idx:
                    y_true_int[i] = norm_to_idx[s]

            valid_mask = y_true_int >= 0
            if not np.any(valid_mask):
                print("  ⚠ 有効な正解ラベルが 1 件もなかったため、評価をスキップします。")
            else:
                y_true_valid = y_true_int[valid_mask]
                y_pred_valid = y_pred[valid_mask]

                # 全クラス ID を対象に混同行列を作成
                all_labels = np.arange(len(class_names))

                cm = confusion_matrix(y_true_valid, y_pred_valid, labels=all_labels)
                print("混同行列（行: 真, 列: 予測）:")
                print(cm)

                print("\nclassification_report:")
                print(
                    classification_report(
                        y_true_valid,
                        y_pred_valid,
                        labels=all_labels,
                        target_names=[str(c) for c in class_names],
                        zero_division=0,  # support=0 クラスは 0 扱い
                    )
                )

                # --- 評価結果をファイル出力 ---
                try:
                    # 混同行列 CSV（全クラス分）
                    cm_df = pd.DataFrame(
                        cm,
                        index=[str(c) for c in class_names],
                        columns=[str(c) for c in class_names],
                    )
                    cm_path = f"{eval_prefix}_confusion_matrix.csv"
                    cm_df.to_csv(cm_path, encoding="utf-8-sig")
                    print(f"  → 混同行列を保存しました: {cm_path}")

                    # classification_report CSV
                    rep_dict = classification_report(
                        y_true_valid,
                        y_pred_valid,
                        labels=all_labels,
                        target_names=[str(c) for c in class_names],
                        output_dict=True,
                        zero_division=0,
                    )
                    rep_df = pd.DataFrame(rep_dict).T
                    rep_path = f"{eval_prefix}_classification_report.csv"
                    rep_df.to_csv(rep_path, encoding="utf-8-sig")
                    print(f"  → classification_report を保存しました: {rep_path}")

                    # 予測モード用の混同行列図（predict）も保存
                    cm_labels = [str(c) for c in class_names]
                    suffix = "predict"
                    cm_abs_path = out_dir / f"rf_confusion_matrix_{suffix}_{run_id}.png"
                    cm_norm_path = (
                        out_dir
                        / f"rf_confusion_matrix_{suffix}_normalized_{run_id}.png"
                    )
                    _plot_confusion_matrix(
                        cm,
                        cm_labels,
                        normalize=False,
                        title=f"Confusion matrix ({suffix})",
                        save_path=cm_abs_path,
                    )
                    _plot_confusion_matrix(
                        cm,
                        cm_labels,
                        normalize=True,
                        title=f"Confusion matrix (normalized, {suffix})",
                        save_path=cm_norm_path,
                    )

                    saved_eval = True
                except Exception as e:
                    print(f"  ⚠ 評価結果の保存に失敗しました: {e}")

        # ---------------------------
        # LabelEncoder 未使用の場合
        # ---------------------------
        else:
            y_true = y_true_raw
            # class_names があればそれを「全クラス」とみなす
            if class_names:
                all_labels = list(class_names)
                rep_target_names = [str(c) for c in all_labels]
            else:
                # なければ今回のデータに出てきたクラスだけでやる
                uniq = sorted(pd.unique(y_true))
                all_labels = list(uniq)
                rep_target_names = [str(u) for u in uniq]

            cm = confusion_matrix(y_true, y_pred_labels, labels=all_labels)
            print("混同行列（行: 真, 列: 予測）:")
            print(cm)

            print("\nclassification_report:")
            print(
                classification_report(
                    y_true,
                    y_pred_labels,
                    labels=all_labels,
                    target_names=rep_target_names,
                    zero_division=0,
                )
            )

            # --- 評価結果をファイル出力 ---
            try:
                cm_df = pd.DataFrame(
                    cm,
                    index=[str(c) for c in all_labels],
                    columns=[str(c) for c in all_labels],
                )
                cm_path = f"{eval_prefix}_confusion_matrix.csv"
                cm_df.to_csv(cm_path, encoding="utf-8-sig")
                print(f"  → 混同行列を保存しました: {cm_path}")

                rep_dict = classification_report(
                    y_true,
                    y_pred_labels,
                    labels=all_labels,
                    target_names=rep_target_names,
                    output_dict=True,
                    zero_division=0,
                )
                rep_df = pd.DataFrame(rep_dict).T
                rep_path = f"{eval_prefix}_classification_report.csv"
                rep_df.to_csv(rep_path, encoding="utf-8-sig")
                print(f"  → classification_report を保存しました: {rep_path}")

                # 予測モード用の混同行列図（predict）も保存
                cm_labels = [str(c) for c in all_labels]
                suffix = "predict"
                cm_abs_path = out_dir / f"rf_confusion_matrix_{suffix}_{run_id}.png"
                cm_norm_path = (
                    out_dir
                    / f"rf_confusion_matrix_{suffix}_normalized_{run_id}.png"
                )
                _plot_confusion_matrix(
                    cm,
                    cm_labels,
                    normalize=False,
                    title=f"Confusion matrix ({suffix})",
                    save_path=cm_abs_path,
                )
                _plot_confusion_matrix(
                    cm,
                    cm_labels,
                    normalize=True,
                    title=f"Confusion matrix (normalized, {suffix})",
                    save_path=cm_norm_path,
                )

                saved_eval = True
            except Exception as e:
                print(f"  ⚠ 評価結果の保存に失敗しました: {e}")

    # =====================================================
    # 予測結果の GPKG 出力（predict 実行時）
    #   pred / eval それぞれについて
    #     - GPKG を出すか？
    #     - 間引き率
    #     - x/y 列
    #     - CRS (EPSG)
    #   まず「設定フェーズ」でこれらを決めてから、
    #   最後にまとめて GPKG を書き出す。
    # =====================================================
    try:
        # -------------------------------
        # ベース DataFrame の準備
        # -------------------------------
        pred_df = out.copy()
        eval_df = None
        if target_col in df.columns:
            eval_df = out.copy()
            eval_df["true_label"] = df[target_col]
            eval_df["is_correct"] = (
                eval_df["true_label"].astype(str)
                == eval_df["y_pred"].astype(str)
            )

        # -------------------------------
        # 設定フェーズ
        # -------------------------------
        pred_gpkg_enable = False
        pred_parquet_enable = False
        pred_thinning = 1
        pred_x_col = None
        pred_y_col = None
        pred_crs_epsg = None

        eval_gpkg_enable = False
        eval_parquet_enable = False
        eval_thinning = 1
        eval_x_col = None
        eval_y_col = None
        eval_crs_epsg = None

        # --- pred 用設定 ---
        if pred_df is not None and not pred_df.empty:
            print("\n[オプション] 予測結果ベクタ（y_pred / proba_max のみ）を出力します。")
            if ask_yes_no("  → ベクタ出力を保存しますか？", default=False):
                pred_gpkg_enable, pred_parquet_enable = ask_vector_output_format(default="both")
                pred_thinning = ask_thinning_factor(
                    "  [pred] GPKG に書き出す間引き率を指定してください（例: 10=1/10, 100=1/100）",
                    default=1,
                )
                pred_x_col, pred_y_col = ensure_xy_columns(pred_df)
                pred_default_crs = source_crs_str or "EPSG:4326"
                pred_crs_epsg = ask_crs(default_epsg=pred_default_crs)
            else:
                print("  → 予測結果ベクタの出力はスキップします。")
        else:
            print("\n[INFO] 予測結果が空のため、pred GPKG の設定・出力はスキップします。")

        # --- eval 用設定（target_col があるときだけ） ---
        if eval_df is not None:
            if not eval_df.empty:
                print("\n[オプション] 評価用ベクタ（true_label / is_correct 付き）を出力します。")
                if ask_yes_no("  → 評価用ベクタを保存しますか？", default=False):
                    eval_gpkg_enable, eval_parquet_enable = ask_vector_output_format(default="both")
                    eval_thinning = ask_thinning_factor(
                        "  [eval] GPKG に書き出す間引き率を指定してください（例: 10=1/10, 100=1/100）",
                        default=1,
                    )
                    eval_x_col, eval_y_col = ensure_xy_columns(eval_df)
                    eval_default_crs = source_crs_str or "EPSG:4326"
                    eval_crs_epsg = ask_crs(default_epsg=eval_default_crs)
                else:
                    print("  → 評価用ベクタの出力はスキップします。")
            else:
                print("\n[INFO] 評価対象が空のため、eval GPKG の設定・出力はスキップします。")
        else:
            print("\n[INFO] 入力テーブルに正解ラベルが無いため、評価用 GPKG は作成できません。")

        # -------------------------------
        # 書き出しフェーズ
        # -------------------------------
        # pred ベクタ出力（GPKG / GeoParquet）
        if pred_gpkg_enable or pred_parquet_enable:
            # GPKG 用に thinning を先に適用しておく
            if pred_thinning and pred_thinning > 1:
                pred_gpkg_out = pred_df.iloc[::pred_thinning].copy()
                print(
                    f"  → [pred] {pred_thinning} 行ごとに 1 行を採用して "
                    f"{len(pred_gpkg_out)} 件を書き出します。"
                )
            else:
                pred_gpkg_out = pred_df

            # GPKG
            if pred_gpkg_enable:
                pred_gpkg_path = out_dir / f"{Path(base_out).name}_pred_{run_id}.gpkg"
                save_gpkg_with_points(
                    pred_gpkg_out,
                    pred_gpkg_path,
                    x_col=pred_x_col,
                    y_col=pred_y_col,
                    crs_epsg=pred_crs_epsg,
                    layer_name="pred",
                )
                print(f"  → 予測結果 GPKG を保存しました: {pred_gpkg_path}")

            # GeoParquet
            if pred_parquet_enable:
                pred_parquet_path = out_dir / f"{Path(base_out).name}_pred_{run_id}.parquet"
                try:
                    save_geoparquet_with_points(
                        pred_gpkg_out,
                        pred_parquet_path,
                        x_col=pred_x_col,
                        y_col=pred_y_col,
                        crs_epsg=pred_crs_epsg,
                    )
                    print(f"  → 予測結果 GeoParquet を保存しました: {pred_parquet_path}")
                except Exception as e:
                    print(f"  ⚠ 予測結果 GeoParquet の保存に失敗しました: {e}")

        # eval ベクタ出力（GPKG / GeoParquet）
        if eval_gpkg_enable or eval_parquet_enable:
            if eval_thinning and eval_thinning > 1:
                eval_gpkg_out = eval_df.iloc[::eval_thinning].copy()
                print(
                    f"  → [eval] {eval_thinning} 行ごとに 1 行を採用して "
                    f"{len(eval_gpkg_out)} 件を書き出します。"
                )
            else:
                eval_gpkg_out = eval_df

            # GPKG
            if eval_gpkg_enable:
                eval_gpkg_path = (
                    out_dir / f"{Path(base_out).name}_eval_predict_{run_id}.gpkg"
                )
                save_gpkg_with_points(
                    eval_gpkg_out,
                    eval_gpkg_path,
                    x_col=eval_x_col,
                    y_col=eval_y_col,
                    crs_epsg=eval_crs_epsg,
                    layer_name="eval_predict",
                )
                print(f"  → 評価用 GPKG を保存しました: {eval_gpkg_path}")

            # GeoParquet
            if eval_parquet_enable:
                eval_parquet_path = (
                    out_dir / f"{Path(base_out).name}_eval_predict_{run_id}.parquet"
                )
                try:
                    save_geoparquet_with_points(
                        eval_gpkg_out,
                        eval_parquet_path,
                        x_col=eval_x_col,
                        y_col=eval_y_col,
                        crs_epsg=eval_crs_epsg,
                    )
                    print(f"  → 評価用 GeoParquet を保存しました: {eval_parquet_path}")
                except Exception as e:
                    print(f"  ⚠ 評価用 GeoParquet の保存に失敗しました: {e}")

    except Exception as e:
        print(f"  ⚠ 予測結果 GPKG の保存に失敗しました: {e}")

    # =====================================================
    # 予測クラスの件数サマリ（常に出力）
    # =====================================================
    try:
        if label_encoder_info and class_names:
            tmp = pd.DataFrame({"cls_id": y_pred})
            grp = tmp.groupby("cls_id").size().sort_index()
            cls_labels = [class_names[int(i)] for i in grp.index]
        else:
            tmp = pd.DataFrame({"cls": y_pred_labels})
            grp = tmp.groupby("cls").size().sort_index()
            cls_labels = [str(c) for c in grp.index]

        summary_df = pd.DataFrame({
            "class": cls_labels,
            "count": grp.values,
        })
        if proba_max is not None:
            if label_encoder_info and class_names:
                tmp2 = pd.DataFrame({"cls_id": y_pred, "proba_max": proba_max})
                gm = tmp2.groupby("cls_id")["proba_max"].mean().reindex(grp.index)
            else:
                tmp2 = pd.DataFrame({"cls": y_pred_labels, "proba_max": proba_max})
                gm = tmp2.groupby("cls")["proba_max"].mean().reindex(grp.index)
            summary_df["proba_max_mean"] = gm.values

        summary_path = f"{eval_prefix}_pred_summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n  → 予測クラスの件数サマリを保存しました: {summary_path}")
    except Exception as e:
        print(f"\n  ⚠ 予測サマリの保存に失敗しました: {e}")

    # =====================================================
    # 特徴量重要度（predict 実行時にも出力: train と書式を揃える）
    #   ※ backend が RF でも XGBoost でも動くように、パイプライン内の
    #      ステップから自動でモデルを検出する
    # =====================================================
    try:
        model_step = None
        for key in ("rf", "xgb"):
            if key in pipe.named_steps:
                model_step = pipe.named_steps[key]
                break

        if model_step is None:
            raise KeyError("rf / xgb step not found in pipeline")

        importances = model_step.feature_importances_
        imp_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)

        # ファイル名は従来どおり rf_ プレフィックスで互換性を維持
        imp_path = out_dir / f"rf_feature_importance_{run_id}.csv"
        imp_df.to_csv(imp_path, index=False, encoding="utf-8")
        print(f"[保存] 特徴量重要度 CSV（predict）: {imp_path}")

        topn = min(25, len(imp_df))
        fig, ax = plt.subplots(figsize=(8, max(4, topn * 0.3)))
        ax.barh(np.arange(topn), imp_df["importance"].values[:topn][::-1])
        ax.set_yticks(np.arange(topn))
        ax.set_yticklabels(imp_df["feature"].values[:topn][::-1])
        ax.set_xlabel("Importance")
        ax.set_title("Feature importance (top 25)")
        plt.tight_layout()
        fig_path = out_dir / f"rf_feature_importance_{run_id}.png"
        fig.savefig(fig_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"[保存] 特徴量重要度 図（predict）: {fig_path}")
    except Exception as e:
        print(f"  ⚠ 特徴量重要度の保存（predict）に失敗しました: {e}")

    # =====================================================
    # 予測結果テーブルの保存（CSV / Parquet / GPKG）
    #   ※ out_path は関数冒頭で既に決定済み
    # =====================================================
    low = out_path.lower()
    if low.endswith(".csv"):
        out.to_csv(out_path, index=False, encoding="utf-8")
        print(f"\n✅ 予測結果（CSV）を書き出しました: {out_path}")

    elif low.endswith(".gpkg"):
        # 出力GPKGの場合、geometry を使うか / x,y から作るか
        if "geometry" in out.columns:
            use_geom = ask_yes_no("入力に geometry 列があるので、そのままGPKGに書き出しますか？", default=True)
        else:
            use_geom = False

        if use_geom:
            gtmp = gpd.GeoDataFrame(out, geometry="geometry")
            default_crs = source_crs_str or (str(gtmp.crs) if gtmp.crs else "EPSG:4326")
            crs = ask_crs(default_epsg=default_crs)
            gtmp.set_crs(crs, inplace=True)
            layer_name = input("GPKGのレイヤ名（空=pred）: ").strip() or "pred"
            gtmp.to_file(out_path, driver="GPKG", layer=layer_name)
            print(f"\n✅ 予測結果（GPKG）を書き出しました: {out_path}（レイヤ: {layer_name}, CRS={crs}）")
        else:
            if {"x", "y"}.issubset(out.columns):
                default_crs = source_crs_str or "EPSG:4326"
                crs = ask_crs(default_epsg=default_crs)
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
        # 拡張子が不明な場合も、そのまま CSV として扱う
        out.to_csv(out_path, index=False, encoding="utf-8")
        print(f"\n✅ 予測結果（CSV）を書き出しました: {out_path}")

    print("\n[完了] 予測モードが正常に終了しました。")


# =========================================================
# メイン
# =========================================================

def main():
    print("\n=== ランダムフォレスト（地形分類＋学習データ作成） ===")
    print("  1) 学習用データ作成（ラスター/ポリゴン → テーブル）")
    print("  2) 学習（RandomForest / CPU）")
    print("  3) 学習（RandomForest：XGBoost / GPU）")
    print("  4) 予測（学習済みモデルで予測：RF or XGBoost / CPU or GPU 自動）")
    print("  0) 終了")
    choice = input("番号を選んでください [0-4]: ").strip() or "2"

    if choice == "1":
        make_training_data_mode()
    elif choice == "2":
        # RandomForest（CPU）
        train_mode(backend="rf")
    elif choice == "3":
        # XGBoost（GPU）
        train_mode(backend="xgb")
    elif choice == "4":
        # 予測は読み込むモデルに応じて CPU / GPU, RF / XGBoost が自動で切り替わる
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
