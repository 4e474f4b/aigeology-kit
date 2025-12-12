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
        print("  1) GPKG（GeoPackage）のみ")
        print("  2) GeoParquet のみ")
        print("  3) 両方出力する")
        s = input(f"番号を選んでください [1-3]（空={default_choice}）: ").strip()
        if not s:
            s = default_choice

        if s == "1":
            return True, False
        if s == "2":
            return False, True
        if s == "3":
            return True, True

        print("  ⚠ 1〜3 の番号で入力してください。")

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
) -> tuple[pd.DataFrame, bool]:
    """
    予測用にテーブルを読み込むユーティリティ。

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
            return df, True  # ランダムサンプル
        else:
            df = _read_gpkg_attributes_with_limit(path, layer_name=None, limit=None)
            return df, False  # 全件・先頭から

    # CSV
    if suffix == ".csv":
        df = pd.read_csv(path)
        if max_rows is not None and 0 < max_rows < len(df):
            df = df.sample(n=max_rows, random_state=random_state)
            return df, True
        return df, False

    # Parquet
    if suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        if max_rows is not None and 0 < max_rows < len(df):
            df = df.sample(n=max_rows, random_state=random_state)
            return df, True
        return df, False

    # その他（とりあえず CSV として読む）
    df = pd.read_csv(path)
    if max_rows is not None and 0 < max_rows < len(df):
        df = df.sample(n=max_rows, random_state=random_state)
        return df, True
    return df, False


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
        # 出力形式を選択
        save_gpkg, save_parquet = ask_vector_output_format(default="both")

        # 間引き率を指定
        eval_gpkg_thinning = ask_thinning_factor(
            "  [eval] GPKG に書き出す間引き率を指定してください"
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
            print(
                "  ⚠ x/y 座標列の指定に失敗したため、このセッションでは"
                f"評価用ベクタ（GPKG / GeoParquet）の出力をスキップします: {e}"
            )
    else:
        print("  → 評価用ベクタの出力はスキップします。")

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

        eval_gpkg_path = None  # Monte Carlo 用の評価 GPKG パス

        if last_split is not None:
            idx_train_cv, idx_test_cv = last_split
            X_train_cv, X_test_cv = X[idx_train_cv], X[idx_test_cv]
            y_train_cv, y_test_cv = y[idx_train_cv], y[idx_test_cv]

            pipe.fit(X_train_cv, y_train_cv)
            y_pred_cv = pipe.predict(X_test_cv)
            # 学習データ側の予測（train GPKG / GeoParquet 用）
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

        # Monte Carlo: 最終分割の評価結果をベクタ（GPKG / GeoParquet）で出力（指定間引き対応）
            eval_gpkg_path = None
            eval_parquet_path = None
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
                            "  → Monte Carlo 最終分割の評価ベクタを "
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

                    # --- 学習データ側（train）の GPKG / GeoParquet も出力 ---
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

                    thinning_train = (
                        eval_gpkg_thinning
                        if eval_gpkg_thinning and eval_gpkg_thinning > 1
                        else 1
                    )
                    if thinning_train > 1:
                        before_n_train = len(train_df)
                        train_df = train_df.iloc[::thinning_train].copy()
                        print(
                            "  → Monte Carlo 最終分割の学習用 GPKG / GeoParquet を "
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
                        # Monte Carlo 最終分割の学習用 GeoParquet も保存
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
                            print(
                                f"[WARN] Monte Carlo 最終分割 学習用 GeoParquet の書き出しに失敗しました: {e}"
                            )

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
                            print(f"[保存] Monte Carlo 最終分割 評価 GeoParquet: {eval_parquet_path}")
                        except Exception as e:
                            print(f"[WARN] Monte Carlo 最終分割 評価用 GeoParquet の書き出しに失敗しました: {e}")
                else:
                    print(
                        "  ⚠ 指定された座標列が df に存在しないため、"
                        "Monte Carlo 評価用 GPKG / Parquet の出力をスキップします。"
                    )
            else:
                print(
                    "  ⚠ 評価用 GPKG / Parquet 出力は無効です（座標列が無いか、ベクタ出力オプションが OFF です）。"
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

        # Monte Carlo でも評価 GPKG を出力した場合は、そのパスをメタ情報に含める
        if eval_gpkg_path is not None:
            eval_info["eval_gpkg_path"] = str(eval_gpkg_path)

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
#   ※ k-fold CV では train 側 GPKG / GeoParquet は出力せず、
#      OOF 評価用の eval_cv_oof のみをベクタ化する。
        if xy_cols is not None:
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

        eval_cv_gpkg_path = None
        eval_cv_parquet_path = None

        if xy_cols is not None and eval_gpkg_thinning is not None and (save_gpkg or save_parquet):
            # x,y 列名の取り出し
            if isinstance(xy_cols, tuple):
                x_col, y_col = xy_cols
            else:
                x_col, y_col = xy_cols[0], xy_cols[1]

            if (x_col in df.columns) and (y_col in df.columns):
                # 事前に指定した eval_gpkg_thinning を適用（1 なら全件）
                if eval_gpkg_thinning and eval_gpkg_thinning > 1:
                    before_n = len(eval_df)
                    eval_df = eval_df.iloc[::eval_gpkg_thinning].copy()
                    print(
                        "  → CV OOF 評価ベクタを "
                        f"{eval_gpkg_thinning} 行ごとに 1 行に間引き "
                        f"{before_n} 件 → {len(eval_df)} 件を書き出します。"
                    )

                print("\n[オプション] CV OOF 評価結果を GPKG / GeoParquet として保存します。")
                # 事前に取得した EPSG を利用（念のため未設定ならここで聞く）
                crs_epsg = eval_crs_epsg or ask_crs(default_epsg=None)

                if save_gpkg:
                    eval_cv_gpkg_path = out_dir / f"{base.stem}_eval_cv_oof_{run_id}.gpkg"
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
                    eval_cv_parquet_path = out_dir / f"{base.stem}_eval_cv_oof_{run_id}.parquet"
                    try:
                        save_geoparquet_with_points(
                            eval_df,
                            eval_cv_parquet_path,
                            x_col=x_col,
                            y_col=y_col,
                            crs_epsg=crs_epsg,
                        )
                        print(f"[保存] CV OOF 評価 GeoParquet: {eval_cv_parquet_path}")
                    except Exception as e:
                        print(f"[WARN] CV OOF 評価用 GeoParquet の書き出しに失敗しました: {e}")
            else:
                eval_cv_gpkg_path = None
                eval_cv_parquet_path = None
                print(
                    "  ⚠ 指定された座標列が df に存在しないため、CV 用 GPKG / Parquet の出力をスキップします。"
                )
        else:
            eval_cv_gpkg_path = None
            eval_cv_parquet_path = None

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
        if eval_cv_parquet_path is not None:
            eval_info["eval_parquet_path"] = str(eval_cv_parquet_path)
         
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
        # 学習データ側の予測（train GPKG / GeoParquet 用）
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

        # オプション: ホールドアウト法のテストデータを評価用ベクタ（GPKG / GeoParquet）として出力（指定間引き対応）
        eval_gpkg_path = None
        eval_parquet_path = None
        if xy_cols is not None and eval_gpkg_thinning is not None and (save_gpkg or save_parquet):
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

                # 間引き（1 ならそのまま）
                thinning = eval_gpkg_thinning if eval_gpkg_thinning and eval_gpkg_thinning > 1 else 1
                if thinning > 1:
                    before_n = len(eval_df)
                    eval_df = eval_df.iloc[::thinning].copy()
                    print(
                        "  → ホールドアウト評価ベクタを "
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

                # --- 学習データ側（train）の GPKG / GeoParquet も出力 ---
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
                    eval_gpkg_thinning
                    if eval_gpkg_thinning and eval_gpkg_thinning > 1
                    else 1
                )
                if thinning_train > 1:
                    before_n_train = len(train_df)
                    train_df = train_df.iloc[::thinning_train].copy()
                    print(
                        "  → ホールドアウト学習用 GPKG / GeoParquet を "
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

                if save_parquet:
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
            else:
                print(
                    "  ⚠ 指定された座標列が df に存在しないため、ホールドアウト評価用 "
                    "GPKG / Parquet の出力をスキップします。"
                )
        else:
            print(
                "  ⚠ 評価用 GPKG / Parquet 出力は無効です（座標列が無いか、ベクタ出力オプションが OFF です）。"
            )
        eval_mode = "holdout"
        eval_info = {
            "test_size": test_size,
            "use_stratify": use_stratify,
        }
        if eval_gpkg_path is not None:
            eval_info["eval_gpkg_path"] = str(eval_gpkg_path)
        if eval_parquet_path is not None:
            eval_info["eval_parquet_path"] = str(eval_parquet_path)

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
# メイン
# =========================================================

def main():  
    """メインメニュー：学習専用モード"""  

    print("\n=== ランダムフォレスト（学習のみ） ===")  
    print(" 1) 学習（RandomForest / CPU）")            
    print(" 2) 学習（RandomForest：XGBoost / GPU）")   
    print(" 3) 終了")                                   

    choice = input("番号を選んでください [1-3]: ").strip() or "2"  

    if choice == "1":          
        # 学習（RandomForest / CPU）  
        train_mode(backend="rf")      
    elif choice == "2":        
        # 学習（RandomForest：XGBoost / GPU）  
        train_mode(backend="xgb")     
    elif choice == "3":        
        print("終了します。")  
    else:                      
        print("番号が正しくありません。")  

# フォントセット（スクリプト開始時に一度でOK。未設定ならここで）
if __name__ == "__main__":
    try:
        setup_matplotlib_japanese_font()  # ← ここで一度だけ
        main()
    except KeyboardInterrupt:
        print("\n中断しました。")
