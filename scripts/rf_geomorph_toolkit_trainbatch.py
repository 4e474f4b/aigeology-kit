#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rf_geomorph_toolkit_trainbatch.py

rf_geomorph_toolkit.py から train_mode(backend="rf" / "xgb") を
「事前に用意したパラメータセット(JSON)」に基づいて
バッチ実行するラッパースクリプト。

ポイント:
- rf_geomorph_toolkit_train.py（学習だけ） 本体は変更しない（I/O仕様を変えない）
- load_table_interactive / choose_target_and_features / ask_yes_no / input を
  一時的にモンキーパッチして「対話式入力の代わりに config の値」を渡す
- 設定内容は JSON に保存し、各ランごとにスナップショットも出力
- train_mode 内部の処理ロジック・出力仕様はそのままなので、
  手入力で回した場合と結果の仕様は同じになる
"""

# =========================================================
# 標準ライブラリ
# =========================================================
import json # 設定ファイル(JSON)読み書き用
import os
import sys
import platform # 環境情報表示用
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List
import builtins # input のモンキーパッチ用
from contextlib import contextmanager

# =========================================================
# 依存スクリプト: rf_geomorph_toolkit_train.py
#   同じディレクトリ(または PYTHONPATH 上)にある前提
#   ※ rf_geomorph_toolkit_train 内部の ImportError（numpy など）と
#      モジュールそのものが見つからないケースを分けて表示する  # ←変更
# =========================================================
try:
    import rf_geomorph_toolkit_train as rf
except ImportError as e:
    # モジュール名が rf_geomorph_toolkit_train の場合 → ファイル自体が見つからない
    mod_name = getattr(e, "name", None)
    if isinstance(e, ModuleNotFoundError) and mod_name == "rf_geomorph_toolkit_train":
        print("rf_geomorph_toolkit_train.py をインポートできません。")
        print("このバッチスクリプトと同じフォルダに rf_geomorph_toolkit_train.py を置くか、PYTHONPATH を調整してください。")
        print(f"詳細: {e}")
    else:
        # それ以外の ImportError（numpy / pandas / xgboost など）。
        print("rf_geomorph_toolkit_train.py の読み込み中に依存ライブラリの ImportError が発生しました。")
        print("rf_geomorph_toolkit_train.py を普段動かしている仮想環境（例: terrain-env）を有効化してから実行してください。")
        print(f"詳細: {e}")
    sys.exit(1)


# =========================================================
# 環境情報の表示（任意）
# =========================================================
def print_environment_info() -> None:
    """実行環境の簡易表示。"""
    print("=== 実行環境情報 ===")
    print(f"- OS         : {platform.system()} {platform.release()}")
    print(f"- Python     : {platform.python_version()}")
    # CuPy / XGBoost の有無だけ軽くチェック（GPU 利用可否の目安）
    try:
        import cupy  # type: ignore  # noqa: F401
        has_cupy = True
    except Exception:
        has_cupy = False

    try:
        import xgboost  # type: ignore  # noqa: F401
        has_xgb = True
    except Exception:
        has_xgb = False

    print(f"- CuPy       : {'あり' if has_cupy else 'なし'}")
    print(f"- XGBoost    : {'あり' if has_xgb else 'なし'}")
    print("============================================\n")


# =========================================================
# 設定ファイル関連
# =========================================================

DEFAULT_CONFIG_NAME = "rf_geomorph_batch_config.json"


def create_config_template(
    path: Path,
    backend: str = "rf",
    val_mode: int = 1,
    n_estimators_list: List[int] | None = None,   # ←変更: sweep 対応
    base_output_root: str | None = None,          # ←変更: バッチ出力ルート
    input_table_path: str | None = None,          # ←変更: テーブルパス
    target_col_spec: str | None = None,           # ←変更: 目的変数カラム
    feature_cols_spec: str | None = None,         # ←変更: 特徴量カラム指定
    thin_rate_y: int | None = None,               # ←追加: y の間引き率
    output_epsg: str | None = None,               # ←追加: 保存する座標系
    max_rows: int | None = None,                  # ←追加: 学習＋検証に使う最大行数
) -> None:
    """
    CPU/GPU × 検証法パターンに応じたサンプル設定 JSON を生成する。
    backend: "rf" or "xgb"
    val_mode: 1=ホールドアウト, 2=Monte Carlo, 3=k-fold
    n_estimators_list: 値が複数あれば runs を複数作成
    """
    # ひな形ラン名のデフォルト
    val_name = {1: "holdout", 2: "mc", 3: "kfold"}.get(val_mode, "run")

    # n_estimators の候補
    if not n_estimators_list:  # None または空なら 200 のみ
        n_estimators_list = [200]

    runs: List[Dict[str, Any]] = []

    # グローバル設定のデフォルト値  # ←変更
    if input_table_path is None:
        input_table_path = "/path/to/sample_ME-Grid01m.parquet"
    if target_col_spec is None or not str(target_col_spec).strip():
        target_col_spec = "LandClass"
    # feature_cols_spec は「未指定なら None」のままにしておく

    for n_est in n_estimators_list:
        run_name = f"{backend}_{val_name}_n{n_est}"
        run_cfg: Dict[str, Any] = {
            "name": run_name,  # ランの識別名（ディレクトリ名に使う）
            "backend": backend,  # "rf" or "xgb"

            # ターゲット列と特徴量列
            # - target_col: 列名 or 0始まりの列インデックス
            # - feature_cols:
            #     * "2-193" のような範囲指定文字列
            #     * または 列名/インデックスのリスト
            # 目的変数・特徴量カラム（未指定なら後で対話選択にフォールバック）  # ←変更
            "target_col": target_col_spec,
            "feature_cols": feature_cols_spec,

            # y の間引き率（例: 1=全件, 10=1/10, 1000=1/1000。None/空なら対話にフォールバック）  # ←追加
            "thin_rate_y": thin_rate_y,

            # 保存する座標系（例: "EPSG:6673" や "6673" など。None/空なら対話にフォールバック）  # ←追加
            "output_epsg": output_epsg,

            # デバッグ用に学習＋検証に使う最大行数（None なら全件）
            "max_rows": max_rows,  # ←変更

            # 検証法: 1=ホールドアウト, 2=Monte Carlo CV, 3=k-fold CV
            "val_mode": val_mode,

            # 共通パラメータ
            "random_state": 42,
            "use_stratify": True,
            # 学習曲線はバッチの主目的なのでデフォルト ON
            "enable_learning_curve": True,

            # 評価結果を GPKG として出力するか
            "output_gpkg": True,
            # ensure_xy_columns 内の「この列を x,y として使ってよいか」を自動承認するか
            "accept_xy_auto": True,

            # モデルパラメータ
            "n_estimators": int(n_est),
            "max_depth": None,  # None の場合は空Enter扱い（train_mode側の既定値）

            # XGBoost 用パラメータ (backend="xgb" のときのみ利用)
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,

            # Monte Carlo 用（val_mode=2 で有効）
            "mc_n_splits": 10,
            "mc_test_size": 0.2,

            # k-fold 用（val_mode=3 で有効）
            "k_splits": 5,
        }
        runs.append(run_cfg)

    # ★出力ルートが指定されていない場合は既定値 "./rf_batch_results" を使う  # ←変更
    if not base_output_root:
        base_output_root = "./rf_batch_results"
    # 絶対パスに正規化しておくと後続処理がわかりやすい  # ←変更
    base_output_root = str(Path(base_output_root).resolve())  # ←変更

    template: Dict[str, Any] = {
        "global": {
            # rf_geomorph_toolkit.py が読む学習用テーブル  # ←変更
            "input_table_path": input_table_path,
            # 各ランの結果をまとめるルートフォルダ（対話で指定した値）  # ←変更
            "base_output_root": base_output_root,  # ←変更
        },
        "runs": runs,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    print(f"[INFO] バッチ設定テンプレートを作成しました: {path}")
    print("  このファイルを編集してから、バッチ実行メニューの「2) 設定ファイルを読み込んで実行」を使ってください。")


def load_config(path: Path) -> Dict[str, Any]:
    """JSON 設定を読み込む。"""
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "runs" not in cfg or not isinstance(cfg["runs"], list):
        raise ValueError("設定ファイルに 'runs' リストがありません。")
    if "global" not in cfg or not isinstance(cfg["global"], dict):
        cfg["global"] = {}
    return cfg


def merge_global_and_run(global_cfg: Dict[str, Any], run_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """global 設定と run 設定をマージする（run が優先）。"""
    merged = dict(global_cfg)
    merged.update(run_cfg)
    return merged


# =========================================================
# カラム指定パース（feature_cols 用）
# =========================================================

def _parse_index_spec(spec: str, max_index: int) -> List[int]:
    """
    "0,2-5,10" のような文字列を 0-origin のインデックスリストに変換する。
    rf_geomorph_toolkit.input_indices と同じ表現だが、対話ではなく文字列から直接パースする。
    """
    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    out: List[int] = []
    for token in tokens:
        if "-" in token:
            a, b = token.split("-", 1)
            try:
                start = int(a)
                end = int(b)
            except ValueError:
                raise ValueError(f"インデックス範囲が解釈できません: {token}")
            if start > end:
                start, end = end, start
            if start < 0 or end > max_index:
                raise ValueError(f"インデックス範囲が列数を超えています: {token} (0〜{max_index})")
            out.extend(range(start, end + 1))
        else:
            try:
                idx = int(token)
            except ValueError:
                raise ValueError(f"インデックスが解釈できません: {token}")
            if not (0 <= idx <= max_index):
                raise ValueError(f"インデックスが列数を超えています: {idx} (0〜{max_index})")
            out.append(idx)
    # 重複除去（順序保持）
    seen = set()
    uniq: List[int] = []
    for idx in out:
        if idx not in seen:
            seen.add(idx)
            uniq.append(idx)
    return uniq


def _resolve_column_spec_single(spec: Any, columns: List[str]) -> str:
    """target_col 用: 名前 or インデックスを列名に解決する。"""
    if isinstance(spec, int):
        return columns[spec]
    if isinstance(spec, str):
        if spec in columns:
            return spec
        # 数字っぽく、かつインデックスとして有効ならインデックスとみなす
        if spec.isdigit():
            idx = int(spec)
            if 0 <= idx < len(columns):
                return columns[idx]
    raise ValueError(f"target_col の指定が不正です: {spec}")


def _resolve_column_spec_list(spec: Any, columns: List[str]) -> List[str]:
    """feature_cols 用: 文字列 or リストを列名リストに解決。"""
    if isinstance(spec, str):
        idxs = _parse_index_spec(spec, max_index=len(columns) - 1)
        return [columns[i] for i in idxs]
    if isinstance(spec, list):
        out: List[str] = []
        for v in spec:
            if isinstance(v, int):
                out.append(columns[v])
            elif isinstance(v, str):
                if v in columns:
                    out.append(v)
                elif v.isdigit():
                    idx = int(v)
                    if 0 <= idx < len(columns):
                        out.append(columns[idx])    # 列名を入れる
                    else:
                        raise ValueError(f"feature_cols 内のインデックスが範囲外です: {v}")
                else:
                    raise ValueError(f"feature_cols の指定が列名にもインデックスにも一致しません: {v}")
            else:
                raise ValueError(f"feature_cols の要素が解釈不能です: {v}")
        # 重複除去
        seen = set()
        uniq: List[str] = []
        for c in out:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq
    raise ValueError("feature_cols は文字列 or リストで指定してください。")


# =========================================================
# train_mode を非対話で回すためのモンキーパッチ
# =========================================================

@contextmanager
def patched_train_environment(config: Dict[str, Any], run_output_root: Path):
    """
    rf.train_mode を「設定ファイルに基づいて非対話で回す」ための一時的なパッチ。

    - load_table_interactive: input_table_path を使って自動読み込み
    - choose_target_and_features: target_col / feature_cols を使って自動選択
    - ask_yes_no: GPKG 出力・層化・学習曲線・XY 確認などを設定値で自動応答
    - builtins.input: max_rows / val_mode / random_state / n_estimators / XGB パラメータ /
                      Monte Carlo / k-fold の数値を設定値から返す
    """
    # 元の関数を待避
    orig_load_table_interactive = rf.load_table_interactive
    orig_choose_target_and_features = rf.choose_target_and_features
    orig_ask_yes_no = rf.ask_yes_no
    orig_input = builtins.input

    # -------- load_table_interactive の差し替え --------
    def load_table_interactive_override(*args, **kwargs):  # ←変更: preconfig など余分な引数を受け取る
        # preconfig = kwargs.get("preconfig", None)  # ←必要ならここで見る（今は未使用でOK）
        path = config.get("input_table_path")
        if not path:
            print("[BATCH] config に input_table_path が設定されていません。対話モードにフォールバックします。")
            return orig_load_table_interactive(*args, **kwargs)  # ←変更: そのまま委譲
        path_str = str(path)
        if not os.path.exists(path_str):
            print(f"[BATCH] input_table_path が存在しません: {path_str}")
            sys.exit(1)
        df = rf._safe_read_table(path_str)
        print(f"[BATCH] テーブル自動読み込み: {path_str} （{len(df):,} 行 × {len(df.columns)} 列）")
        return df, path_str

    # -------- choose_target_and_features の差し替え --------
    def choose_target_and_features_override(df, *args, **kwargs):  # ←変更
        target_spec = config.get("target_col")
        feature_spec = config.get("feature_cols")

        # 設定ファイル側で target / feature が未指定 or 空文字なら、元の対話 UI にフォールバック  # ←変更
        if (
            target_spec is None
            or str(target_spec).strip() == ""
            or feature_spec is None
            or (isinstance(feature_spec, str) and feature_spec.strip() == "")
        ):
            print("[BATCH] target_col / feature_cols が未設定のため、元の対話モードで列選択を行います。")  # ←変更
            return orig_choose_target_and_features(df, *args, **kwargs)  # ←変更
        cols = list(df.columns)
        target_col = _resolve_column_spec_single(target_spec, cols)
        feature_cols = _resolve_column_spec_list(feature_spec, cols)
        print("[BATCH] 目的変数・特徴量列を自動設定します。")
        print(f"  target_col   : {target_col}")
        print(f"  feature_cols : {feature_cols}")
        return target_col, feature_cols

    # -------- ask_yes_no の差し替え --------
    def ask_yes_no_override(prompt: str, default: bool | None = None, *args, **kwargs) -> bool:  # ←変更
        text = str(prompt)
        # GPKG 出力有無
        if "GPKG（ポイント）として出力しますか" in text:
            return bool(config.get("output_gpkg", False))
        # 層化
        if "層化サンプリング / 層化CV を使いますか" in text:
            return bool(config.get("use_stratify", True))
        # 学習曲線
        if "学習曲線の PNG" in text:
            return bool(config.get("enable_learning_curve", False))
        # x,y 列の確認
        if "この列を x,y として使ってよいですか" in text:
            return bool(config.get("accept_xy_auto", True))
        # それ以外は元の挙動
        return orig_ask_yes_no(prompt, default, *args, **kwargs)  # ←変更

    # -------- input の差し替え --------
    def input_override(prompt: str = "") -> str:
        text = str(prompt)

        # 学習＋検証に使う行数
        if "学習＋検証に使う行数の上限" in text:
            v = config.get("max_rows")
            return "" if v in (None, "", "None") else str(v)

        # モデル保存ルートフォルダ
        if "モデル保存ルートフォルダ" in text:
            # ここは各ランごとに run_output_root を固定で返す
            return str(run_output_root)


        # y の間引き率（train_mode 側のプロンプトに対応）  # ←追加
        # 例: "  → y の場合、間引き率を指定してください（例: 1=全件, 10=1/10, 1000=1/1000。空=1）: （1=間引きなし）: "
        if "間引き率を指定してください" in text:
            v = config.get("thin_rate_y")
            return "" if v in (None, "", "None") else str(v)

        # 保存する座標系（EPSG）  # ←追加
        # 例: "保存する座標系（空=None。例: EPSG:4326 / EPSG:6677）: "
        if "保存する座標系" in text:
            v = config.get("output_epsg")
            return "" if v in (None, "", "None") else str(v)

        # 学習＋検証に使う最大行数 / 上限行数  # ←追加
        # ログ上のメッセージ:
        #  「まずは動作確認だけしたい場合に、上限行数を指定してください。」
        #  などを想定して、「最大行数」「上限行数」が含まれていれば捕まえる。
        if "最大行数" in text or "上限行数" in text:
            v = config.get("max_rows")
            return "" if v in (None, "", "None") else str(v)

        # 検証方法の選択
        if "番号を選択してください [1-3]" in text:
            v = config.get("val_mode", 1)
            return str(v)

        # random_state
        if text.startswith("random_state"):
            v = config.get("random_state")
            return "" if v in (None, "", "None") else str(v)

        # n_estimators
        if text.startswith("n_estimators"):
            v = config.get("n_estimators")
            return "" if v in (None, "", "None") else str(v)

        # ホールドアウト: テストデータ割合 test_size（0〜0.5 程度, 空=0.2）  # ←変更
        if "テストデータ割合（0〜0.5 程度, 空=0.2）" in text:  # ←変更
            v = config.get("test_size")  # ←変更
            return "" if v in (None, "", "None") else str(v)  # ←変更

        # max_depth
        if text.startswith("max_depth"):
            v = config.get("max_depth")
            return "" if v in (None, "", "None") else str(v)

        # XGBoost: learning_rate
        if "learning_rate（学習率" in text:
            v = config.get("learning_rate")
            return "" if v in (None, "", "None") else str(v)

        # XGBoost: subsample
        if "subsample（サンプル行の割合" in text:
            v = config.get("subsample")
            return "" if v in (None, "", "None") else str(v)

        # XGBoost: colsample_bytree
        if "colsample_bytree（サンプル特徴量の割合" in text:
            v = config.get("colsample_bytree")
            return "" if v in (None, "", "None") else str(v)

        # Monte Carlo: n_splits
        if "繰り返し回数 n_splits" in text:
            v = config.get("mc_n_splits")
            return "" if v in (None, "", "None") else str(v)

        # Monte Carlo: test_size
        if "テストデータ割合 test_size" in text:
            v = config.get("mc_test_size")
            return "" if v in (None, "", "None") else str(v)

        # k-fold: k-分割数
        if "k-分割数（空=5）" in text:
            v = config.get("k_splits")
            return "" if v in (None, "", "None") else str(v)

        # 保存する座標系  # ←追加
        # 例: "保存する座標系（空=None。例: EPSG:4326 / EPSG:6677）: "
        if "保存する座標系" in text:
            v = config.get("output_epsg")
            return "" if v in (None, "", "None") else str(v)

        # それ以外は元の input にフォールバック（想定外の入力があれば対話的に聞く）
        return orig_input(prompt)

    try:
        rf.load_table_interactive = load_table_interactive_override  # type: ignore
        rf.choose_target_and_features = choose_target_and_features_override  # type: ignore
        rf.ask_yes_no = ask_yes_no_override  # type: ignore
        builtins.input = input_override  # type: ignore
        yield
    finally:
        rf.load_table_interactive = orig_load_table_interactive  # type: ignore
        rf.choose_target_and_features = orig_choose_target_and_features  # type: ignore
        rf.ask_yes_no = orig_ask_yes_no  # type: ignore
        builtins.input = orig_input  # type: ignore


# =========================================================
# バッチ 1 ランを実行
# =========================================================

def run_single_batch(global_cfg: Dict[str, Any], run_cfg: Dict[str, Any], batch_root: Path) -> None:
    """1つのラン設定に基づき train_mode を1回実行する。"""
    merged = merge_global_and_run(global_cfg, run_cfg)
    backend = merged.get("backend", "rf")
    if backend not in ("rf", "xgb"):
        raise ValueError(f"backend は 'rf' または 'xgb' を指定してください: {backend}")

    run_name = merged.get("name") or f"{backend}_run"

    # 出力ディレクトリ: batch_root / YYYYMMDD_HHMMSS_runname
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = batch_root / f"{now}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n======================================")
    print(f"[BATCH] ラン開始: {run_name} (backend={backend})")
    print(f"[BATCH] 出力ディレクトリ: {run_dir}")
    print("======================================")

    # このランで実際に使った設定を run_dir に保存
    config_snapshot_path = run_dir / "batch_effective_config.json"
    with config_snapshot_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"[BATCH] このランの設定を保存しました: {config_snapshot_path}")

    # train_mode 実行（モンキーパッチ環境）
    with patched_train_environment(merged, run_output_root=run_dir):
        if backend == "xgb":
            rf.train_mode(backend="xgb")
        else:
            rf.train_mode(backend="rf")

    print(f"[BATCH] ラン終了: {run_name}")


# =========================================================
# 設定ファイル作成メニュー（6パターン）
# =========================================================

def _parse_int_list_from_input(s: str) -> List[int]:
    """カンマ区切りの整数列文字列を List[int] に変換する。"""
    if not s.strip():
        return []
    out: List[int] = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            raise ValueError(f"整数に変換できません: {token}")
    return out


def interactive_create_config(script_dir: Path) -> None:
    """CPU/GPU × 検証法 6パターンのテンプレート生成メニュー。"""
    print("\n=== 設定ファイルひな形作成メニュー ===")
    print("  1) CPU + ホールドアウト（val_mode=1）")
    print("  2) CPU + Monte Carlo（val_mode=2）")
    print("  3) CPU + k-fold（val_mode=3）")
    print("  4) GPU(XGBoost) + ホールドアウト（val_mode=1）")
    print("  5) GPU(XGBoost) + Monte Carlo（val_mode=2）")
    print("  6) GPU(XGBoost) + k-fold（val_mode=3）")

    choice = input("番号を選んでください [1-6]: ").strip() or "1"

    mapping = {
        "1": ("rf", 1),
        "2": ("rf", 2),
        "3": ("rf", 3),
        "4": ("xgb", 1),
        "5": ("xgb", 2),
        "6": ("xgb", 3),
    }

    if choice not in mapping:
        print("[WARN] 1〜6 の番号を選んでください。")
        return

    backend, val_mode = mapping[choice]
    val_name = {1: "holdout", 2: "mc", 3: "kfold"}[val_mode]
    default_name = f"rf_batch_{backend}_{val_name}.json"
    default_path = script_dir / default_name

    print(f"\n[INFO] ひな形種別: backend={backend}, val_mode={val_mode} ({val_name})")
    print(f"      デフォルトファイル名: {default_path}")

    user_path_str = input(
        f"設定ファイルの保存パス（空= {default_path}）: "
    ).strip()
    # 入力から前後のクォートを除去
    user_path_str = user_path_str.strip().strip('"').strip("'")

    if user_path_str:
        user_path = Path(user_path_str)
        if not user_path.is_absolute():
            user_path = script_dir / user_path
        # 拡張子が無い場合は .json を付与
        if user_path.suffix == "":
            user_path = user_path.with_suffix(".json")
        target_path = user_path
    else:
        target_path = default_path


    # === バッチ出力ルートフォルダを指定させる（任意） ===  # ←変更
    # 既定値は「このスクリプトのあるディレクトリ配下の rf_batch_results」
    default_root = (script_dir / "rf_batch_results").resolve()
    root_str = input(
        f"バッチ出力ルートフォルダ（空= {default_root}）: "
    ).strip()
    root_str = root_str.strip().strip('"').strip("'")
    if root_str:
        base_output_root = root_str
    else:
        base_output_root = str(default_root)

    # === 学習用テーブルパスを指定させる ===  # ←変更
    default_input = "/path/to/sample_ME-Grid01m.parquet"
    input_path_str = input(
        f"学習用テーブルのパス（空= {default_input}）: "
    ).strip()
    input_path_str = input_path_str.strip().strip('"').strip("'")
    if input_path_str:
        input_table_path = input_path_str
    else:
        input_table_path = default_input

    # === 一度テーブルを読み込んでカラム一覧を表示（rf_geomorph_toolkit の流れをバッチ設定で再現） ===  # ←変更
    cols = None
    try:
        preview_df = rf._safe_read_table(input_table_path)
        cols = list(preview_df.columns)
        print("\n[INFO] カラム一覧（index: name）")
        for idx, name in enumerate(cols):
            print(f"  {idx:3d}: {name}")
        print("")
    except Exception as e:
        print(f"[WARN] テーブルをプレビューできませんでした: {e}")

    # === 目的変数と特徴量カラム指定を聞く ===  # ←変更
    target_spec = input(
        "目的変数（クラス）に使うカラム（名前 or index / 空=LandClass）: "
    ).strip()
    if not target_spec:
        target_spec = "LandClass"

    feat_spec = input(
        "特徴量に使うカラム指定（例: 2-193 / 0,2-5 / カンマ区切り / 空=対話選択）: "
    ).strip()
    if not feat_spec:
        print("[WARN] 特徴量カラム指定が空です。バッチ実行時に列選択の対話が発生します。")
        feat_spec = None

    # n_estimators の sweep を対話的に指定
    try:
        ne_str = input(
            "n_estimators の値をカンマ区切りで指定してください（例: 15,50,100,200 / 空=200のみ）: "
        ).strip()
        ne_list = _parse_int_list_from_input(ne_str)
    except ValueError as e:
        print(f"[WARN] n_estimators の解析に失敗しました: {e}")
        print("       既定値の 200 のみを使用します。")
        ne_list = []

    # === y の間引き率 ===  # ←追加
    thin_str = input(
        "y の間引き率（例: 1=全件,10=1/10,1000=1/1000 / 空=1）: "
    ).strip()
    if thin_str:
        try:
            thin_rate_y = int(thin_str)
        except ValueError:
            print("[WARN] y の間引き率が整数でないため、1（間引きなし）にします。")
            thin_rate_y = 1
    else:
        thin_rate_y = 1

    # === 保存する座標系 ===  # ←追加
    epsg_str = input(
        "保存する座標系（空=None。例: EPSG:4326 / EPSG:6677）: "
    ).strip()
    output_epsg = epsg_str if epsg_str else None

    # === 学習＋検証に使う最大行数（メモリ対策） ===  # ←追加
    maxrows_str = input(
        "学習＋検証に使う最大行数（空=None=全件。例: 500000, 2000000）: "
    ).strip()
    if maxrows_str:
        try:
            max_rows = int(maxrows_str)
        except ValueError:
            print("[WARN] 最大行数が整数でないため、None（全件）として扱います。")
            max_rows = None
    else:
        max_rows = None

    create_config_template(
        target_path,
        backend=backend,
        val_mode=val_mode,
        base_output_root=base_output_root,   # ←変更
        n_estimators_list=ne_list,
        input_table_path=input_table_path,   # ←変更
        target_col_spec=target_spec,         # ←変更
        feature_cols_spec=feat_spec,         # ←変更
        thin_rate_y=thin_rate_y,             # ←追加
        output_epsg=output_epsg,             # ←追加
        max_rows=max_rows,                   # ←追加
    )


# =========================================================
# バッチ実行（設定ファイルをロードして一括学習）
# =========================================================

def run_batch_from_config(config_path: Path) -> None:
    """指定された JSON 設定ファイルを読み込み、全 runs を順番に実行。"""
    if not config_path.exists():
        print(f"[ERROR] 設定ファイルが存在しません: {config_path}")
        return

    print(f"[INFO] 設定ファイルを読み込みます: {config_path}")
    cfg = load_config(config_path)

    global_cfg = cfg.get("global", {})
    runs = cfg["runs"]
    if not runs:
        print("[WARN] runs が空です。設定ファイルを編集してください。")
        return

    # バッチ全体のルート出力ディレクトリ
    base_root = global_cfg.get("base_output_root") or "./rf_batch_results"
    batch_root = Path(base_root).resolve()
    batch_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] バッチ出力ルート: {batch_root}")

    total = len(runs)
    for i, run_cfg in enumerate(runs, start=1):
        print(f"\n[BATCH] ===== {i}/{total} ランを開始します =====")
        run_single_batch(global_cfg, run_cfg, batch_root=batch_root)

    print("\n[BATCH] すべてのランが完了しました。")


# =========================================================
# main
# =========================================================

def main() -> None:
    print("\n=== RF/XGB 学習バッチランチャー（rf_geomorph_toolkit 用） ===")
    print_environment_info()

    script_dir = Path(__file__).resolve().parent

    while True:
        print("メニューを選んでください:")
        print("  1) 設定ファイルひな形を作成する（CPU/GPU × 検証法 6パターン）")
        print("  2) 設定ファイルを読み込んでバッチ学習を実行する")
        print("  3) 終了")

        choice = input("番号を選んでください [1-3]（空=2）: ").strip() or "2"

        if choice == "1":
            interactive_create_config(script_dir)
        elif choice == "2":
            # 引数で渡された場合はそれを優先
            if len(sys.argv) > 1:
                config_path = Path(sys.argv[1])
                if not config_path.is_absolute():
                    config_path = script_dir / config_path
            else:
                default_config = script_dir / DEFAULT_CONFIG_NAME
                prompt = f"設定ファイルのパス（空= {default_config}）: "
                p = input(prompt).strip()
                # 前後のクォート除去
                p = p.strip().strip('"').strip("'")
                if p:
                    config_path = Path(p)
                    if not config_path.is_absolute():
                        config_path = script_dir / config_path
                else:
                    config_path = default_config

            run_batch_from_config(config_path)

        elif choice == "3" or choice == "0":
            print("終了します。")
            break
        else:
            print("[WARN] 1〜3 の番号を選んでください。")


if __name__ == "__main__":
    main()
