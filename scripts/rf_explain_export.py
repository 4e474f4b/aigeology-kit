#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter

import json
import joblib
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline


def _parse_cols_input(user_in: str, all_cols: list[str]) -> list[str]:
    """
    入力例:
      - 名前: colA,colB
      - インデックス: 0,2,5
      - 範囲: 2-9
      - 混在: 0,2-5,colX
    """
    user_in = (user_in or "").strip()
    if not user_in:
        raise ValueError("列指定が空です。")

    tokens = [t.strip() for t in user_in.split(",") if t.strip()]
    picked: list[str] = []

    def add_by_index(idx: int):
        if idx < 0 or idx >= len(all_cols):
            raise ValueError(f"列インデックス範囲外: {idx}")
        picked.append(all_cols[idx])

    for t in tokens:
        if "-" in t:
            a, b = t.split("-", 1)
            a = a.strip()
            b = b.strip()
            if a.isdigit() and b.isdigit():
                ia = int(a)
                ib = int(b)
                if ia > ib:
                    ia, ib = ib, ia
                for i in range(ia, ib + 1):
                    add_by_index(i)
            else:
                raise ValueError(f"範囲指定が不正: {t}")
        elif t.isdigit():
            add_by_index(int(t))
        else:
            if t not in all_cols:
                raise ValueError(f"列名が存在しません: {t}")
            picked.append(t)

    # 重複排除（順序維持）
    seen = set()
    uniq = []
    for c in picked:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def _save_df(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

def _run_first_diagnostics(*, Xs: pd.DataFrame, Xs_rf, rf, rf_pred, classes, y_true_s):
     """
     まず入れるべき最小診断：
       - サンプリング後の整合（shape）
       - クラス分布（y_true_s / rf_pred）
       - 代表木抽出が成立するだけの「クラス別 正解件数」があるか
     """
     n_samples = len(Xs)
     print("\n=== DIAG: sampled data consistency ===")
     print(f"[DIAG] Xs rows                : {n_samples}")
     try:
         print(f"[DIAG] Xs_rf shape           : {getattr(Xs_rf, 'shape', None)}")
     except Exception:
         print("[DIAG] Xs_rf shape           : (unavailable)")
     print(f"[DIAG] rf_pred len            : {len(rf_pred)}")

     if y_true_s is None:
         print("[DIAG] y_true_s               : None (no stratified sampling / no true-based contrib)")
     else:
         print(f"[DIAG] y_true_s len           : {len(y_true_s)}")
         if len(y_true_s) != len(rf_pred):
             print("[DIAG][FATAL] y_true_s and rf_pred length mismatch -> representative tree calc will break.")
 
     # classes / dtype
     try:
         print(f"[DIAG] rf.classes_            : {list(getattr(rf, 'classes_', []))}")
     except Exception:
         pass
     print(f"[DIAG] classes used            : {list(classes)}")
     print(f"[DIAG] rf_pred dtype           : {np.asarray(rf_pred).dtype}")
     if y_true_s is not None:
         print(f"[DIAG] y_true_s dtype          : {np.asarray(y_true_s).dtype}")

     # distribution
     pred_cnt = Counter(map(str, rf_pred))
     print("[DIAG] pred class counts       : " + ", ".join([f"{k}={v}" for k, v in pred_cnt.most_common()]))
     if y_true_s is not None:
         true_cnt = Counter(map(str, y_true_s))
         print("[DIAG] true class counts       : " + ", ".join([f"{k}={v}" for k, v in true_cnt.most_common()]))

         rf_correct = (rf_pred == y_true_s)
         acc = float(np.mean(rf_correct)) if len(rf_correct) else float("nan")
         print(f"[DIAG] RF accuracy on sample   : {acc:.4f}")

         # per-class correct counts (this is the key for representative trees)
         print("[DIAG] per-class correct count :")
         for c in classes:
             c_mask = (y_true_s == c)
             n_c = int(np.sum(c_mask))
             n_ok = int(np.sum(rf_correct & c_mask))
             print(f"  - class={c}  n_true={n_c}  n_correct={n_ok}")

def _ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

def _sample_X_y(
    X: pd.DataFrame,
    y_true: np.ndarray | None,
    sample_n: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """
    y_true があれば層化サンプル、なければランダムサンプル。
    戻り: (Xs, y_true_s or None)
    """
    n = len(X)
    if n == 0:
        return X, None
    n_s = min(int(sample_n), n)

    if y_true is not None:
        y_series = pd.Series(y_true, index=X.index, name="_y")
        n_classes = int(y_series.nunique())
        per_class = max(1, n_s // max(1, n_classes))
        sampled_y = y_series.groupby(y_series).sample(
            n=per_class, replace=False, random_state=seed
        )
        Xs = X.loc[sampled_y.index]
        y_true_s = y_series.loc[Xs.index].to_numpy()
        print(f"[INFO] Stratified sampling: total={len(Xs)} (per_class≈{per_class}, classes={n_classes})")
        return Xs, y_true_s

    # fallback: random
    Xs = X.sample(n=n_s, random_state=seed) if n_s < n else X
    print(f"[INFO] Random sampling: total={len(Xs)} (sample_n={n_s})")
    return Xs, None

def _load_label_mapping_from_meta(meta_path: Path, rf_classes) -> dict | None:
    """
    メタJSONから「元ラベル -> 内部クラス(0..K-1)」の対応表を抽出する。
    想定フォーマット例:
      - {"class_mapping": {"1010101": 0, ...}}
      - {"classes": [1010101, 3010101, ...]}  # index が内部クラス
      - {"label_encoder_classes": [1010101, ...]}  # 同上
    """
    if meta_path is None or (not meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        # UTF-8以外の可能性
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        except Exception as e:
            print(f"[WARN] meta json read failed: {e}", file=sys.stderr)
            return None

    # 1) dict形式: 元ラベル->内部クラス
    for k in ["class_mapping", "label_mapping", "class_to_index", "label_to_index"]:
        if isinstance(meta.get(k), dict):
            mp = {}
            for kk, vv in meta[k].items():
                try:
                    kk2 = int(kk)  # "1010101" -> 1010101
                except Exception:
                    kk2 = kk
                mp[kk2] = int(vv)
            return mp

    # 2) list形式: index が内部クラス
    for k in ["classes", "label_encoder_classes", "class_labels", "labels"]:
        if isinstance(meta.get(k), list) and len(meta[k]) > 0:
            lst = meta[k]
            mp = {}
            for i, v in enumerate(lst):
                try:
                    v2 = int(v)
                except Exception:
                    v2 = v
                mp[v2] = int(i)
            return mp

    # 3) 見つからない
    return None


def _meta_find_mapping_recursive(obj) -> dict | None:
    """
    入れ子の dict/list を再帰的に探索し、
      - {label: index} の dict（値が int） もしくは
      - classes の list（index が内部クラス）
    を見つけたら返す。
    """
    # dict: 直接 mapping っぽいものを探す
    if isinstance(obj, dict):
        # (A) 典型キー（トップ/ネスト共通）
        for k in ["class_mapping", "label_mapping", "class_to_index", "label_to_index",
                  "classes", "label_encoder_classes", "class_labels", "labels"]:
            if k in obj:
                v = obj[k]
                if isinstance(v, dict):
                    mp = {}
                    for kk, vv in v.items():
                        try:
                            kk2 = int(kk)
                        except Exception:
                            kk2 = kk
                        try:
                            mp[kk2] = int(vv)
                        except Exception:
                            mp = None
                            break
                    if mp:
                        return mp
                if isinstance(v, list) and len(v) > 0:
                    mp = {}
                    for i, x in enumerate(v):
                        try:
                            x2 = int(x)
                        except Exception:
                            x2 = x
                        mp[x2] = int(i)
                    return mp

        # (B) sklearnのLabelEncoder風: {"label_encoder": {"classes_": [...]}} 等
        for k in ["label_encoder", "le", "encoder", "class_encoder", "y_encoder"]:
            if isinstance(obj.get(k), dict) and isinstance(obj[k].get("classes_"), list):
                lst = obj[k]["classes_"]
                mp = {}
                for i, x in enumerate(lst):
                    try:
                        x2 = int(x)
                    except Exception:
                        x2 = x
                    mp[x2] = int(i)
                return mp

        # (C) 深掘り
        for vv in obj.values():
            found = _meta_find_mapping_recursive(vv)
            if found:
                return found

    # list: 要素を再帰探索
    if isinstance(obj, list):
        for it in obj:
            found = _meta_find_mapping_recursive(it)
            if found:
                return found
    return None

def _guess_meta_path(model_path: Path) -> Path | None:
    """
    joblibと同ディレクトリにあるメタJSONを推測する（空入力時用）。
    例:
      rf_model_xxx.joblib -> rf_model_xxx_meta.json
      rf_model_xxx.joblib -> rf_model_xxx.meta.json
    """
    cand = []
    stem = model_path.stem
    cand.append(model_path.with_name(stem + "_meta.json"))
    cand.append(model_path.with_name(stem + ".meta.json"))
    for p in cand:
        if p.exists():
            return p
    return None

def estimate_zero_cross_thresholds(
    x: np.ndarray,
    shap_v: np.ndarray,
    n_bins: int = 40,
) -> dict:
    """
    SHAP dependence の 0-cross（平均SHAPが0を跨ぐ付近）を推定する簡易法。
    """
    x = np.asarray(x).astype(float)
    shap_v = np.asarray(shap_v).astype(float)

    m = np.isfinite(x) & np.isfinite(shap_v)
    x = x[m]
    shap_v = shap_v[m]
    if x.size < 200:
        return {"threshold": np.nan, "note": "too_few_samples"}

    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(x, qs))
    if edges.size < 5:
        return {"threshold": np.nan, "note": "low_variance"}

    bin_means_x = []
    bin_means_s = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == len(edges) - 2:
            sel = (x >= lo) & (x <= hi)
        else:
            sel = (x >= lo) & (x < hi)
        if sel.sum() < 10:
            continue
        bin_means_x.append(float(np.mean(x[sel])))
        bin_means_s.append(float(np.mean(shap_v[sel])))

    if len(bin_means_x) < 5:
        return {"threshold": np.nan, "note": "bins_too_sparse"}

    bx = np.array(bin_means_x)
    bs = np.array(bin_means_s)

    thr = np.nan
    for i in range(len(bs) - 1):
        s0, s1 = bs[i], bs[i + 1]
        if (s0 <= 0 and s1 > 0) or (s0 >= 0 and s1 < 0):
            x0, x1 = bx[i], bx[i + 1]
            if s1 != s0:
                thr = x0 + (0 - s0) * (x1 - x0) / (s1 - s0)
            else:
                thr = (x0 + x1) / 2
            break

    return {"threshold": float(thr) if np.isfinite(thr) else np.nan, "note": "ok"}


def _get_tree_feature_thresholds(tree, feature_names: list[str]) -> tuple[Counter, list[dict]]:
    """
    1本の決定木から
      - 使用特徴量頻度（splitに出た回数）
      - split閾値一覧（feature, threshold, node_id, depth）
    を抽出する
    """
    t = tree.tree_
    feat = t.feature
    thr = t.threshold
    children_left = t.children_left
    children_right = t.children_right

    usage = Counter()
    thresholds = []

    # depth を DFS で計算
    stack = [(0, 0)]  # (node_id, depth)
    while stack:
        node_id, depth = stack.pop()
        f = feat[node_id]
        is_split = (children_left[node_id] != children_right[node_id])
        if is_split and f >= 0:
            fname = feature_names[f]
            usage[fname] += 1
            thresholds.append({
                "node_id": int(node_id),
                "depth": int(depth),
                "feature": fname,
                "threshold": float(thr[node_id]),
            })
        # traverse
        cl = children_left[node_id]
        cr = children_right[node_id]
        if cl != -1:
            stack.append((cl, depth + 1))
        if cr != -1:
            stack.append((cr, depth + 1))

    return usage, thresholds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="joblib path (empty=prompt)")
    ap.add_argument("--parquet", default="", help="training parquet path (empty=prompt)")
    ap.add_argument("--outdir", default="", help="output directory (empty=prompt)")
    ap.add_argument("--meta", default="", help="meta json path for label mapping (empty=prompt/guess)")    
    ap.add_argument("--meta_debug", action="store_true", help="print meta json top-level keys")    
    # NOTE: args.sample は「メイン処理のサンプル数」のデフォルトとして使用（対話入力が優先）
    ap.add_argument("--sample", type=int, default=5000, help="rows to sample for SHAP (default 5000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=5, help="top-k features for dependence plots per class")
    ap.add_argument("--rep_trees", type=int, default=3, help="representative trees per class (default 3)")
    ap.add_argument("--rep_candidates", type=int, default=20, help="candidate trees per class to export ranking CSV (default 20)")
    ap.add_argument("--tree_text_max_depth", type=int, default=6, help="max depth to print in export_text (default 6)")
    # --- Tree text export mode (unified) ---
    ap.add_argument(
        "--tree_text_mode",
        choices=["rep", "all", "both"],
        default="rep",
        help="tree text export mode: rep=representative only, all=export all trees, both=rep+all (default: rep)",
    )
    ap.add_argument(
        "--tree_text_prompt",
        action="store_true",
        help="force interactive prompt for tree_text_mode (even if CLI specified)",
    )
    # --- SHAP controls ---
    ap.add_argument("--no_shap", action="store_true", help="disable SHAP entirely (fastest)")
    ap.add_argument(
        "--shap_mode",
        choices=["none", "light", "full"],
        default="light",
        help="SHAP mode: none/light/full (default: light)",
    )
    ap.add_argument(
        "--shap_sample",
        type=int,
        default=2000,
        help="rows for SHAP in light mode (default 2000)",
    )
    ap.add_argument(
        "--shap_classes",
        default="",
        help="comma list of class ids to run SHAP on (e.g., 0,3). empty=all",
    )
    args = ap.parse_args()

    # ---------------------------
    # Interactive: SHAP output mode (A/B)
    # - If CLI flags are explicitly given, do not prompt.
    # ---------------------------
    argv = " ".join(sys.argv[1:])
    cli_specified = any(
        k in argv
        for k in ["--no_shap", "--shap_mode", "--shap_sample", "--shap_classes"]
    )
    if (not cli_specified) and (not args.no_shap) and sys.stdin.isatty():
        print("\nSHAP 出力モード")
        print("  A) SHAP を完全に無効化（最軽量）")
        print("  B) SHAP は実行するが計算量を落とす（軽量）")
        m = input("選択 [A/B]（空=B）: ").strip().lower() or "b"
        if m in ("a", "1"):
            args.no_shap = True
            args.shap_mode = "none"
        else:
            args.no_shap = False
            args.shap_mode = "light"
            s = input(f"SHAP用サンプル数（空={args.shap_sample}。例: 2000）: ").strip()
            if s:
                try:
                    args.shap_sample = int(s)
                except ValueError:
                    print("[WARN] SHAP用サンプル数が整数ではないため、既定値を使用します。", file=sys.stderr)
            c = input("SHAP対象クラス（空=全クラス。例: 0,3）: ").strip()
            if c:
                args.shap_classes = c

    model_in = (args.model or "").strip()
    if not model_in:
        model_in = input("モデル(joblib)のパス: ").strip().strip('"')
    parquet_in = (args.parquet or "").strip()
    if not parquet_in:
        parquet_in = input("学習Parquetのパス: ").strip().strip('"')
    outdir_in = (args.outdir or "").strip()
    if not outdir_in:
        outdir_in = input("出力フォルダ(outdir)のパス: ").strip().strip('"')

    meta_in = (args.meta or "").strip()
    if not meta_in:
        # 自動推測（見つかればそれを使う）
        _g = _guess_meta_path(Path(model_in))
        if _g is not None:
            meta_in = str(_g)
            print(f"[INFO] meta json guessed: {meta_in}")
        else:
            meta_in = input("メタJSON（ラベル対応表）のパス（空=スキップ）: ").strip().strip('"')

    model_path = Path(model_in)
    pq_path = Path(parquet_in)
    outdir = Path(outdir_in)
    meta_path = Path(meta_in) if meta_in else None

    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not pq_path.exists():
        raise FileNotFoundError(f"parquet not found: {pq_path}")
    _ensure_outdir(outdir)

    # Load parquet
    df = pd.read_parquet(pq_path)
    all_cols = list(df.columns)

    # Load model（特徴量数を出すため）
    model_raw = joblib.load(model_path)

    # Pipeline 対応
    if isinstance(model_raw, Pipeline):
        # 最終ステップを RandomForest とみなす
        rf = model_raw.steps[-1][1]
        preprocess = model_raw
        print(f"[INFO] Pipeline detected. RF step: {model_raw.steps[-1][0]}")
    else:
        rf = model_raw
        preprocess = None

    expected_n = int(getattr(rf, "n_features_in_", 0)) or None

    print("\n=== Parquet columns (index: name) ===")
    for i, c in enumerate(all_cols):
        print(f"{i:>4}: {c}")

    print("\n列指定を入力（例: colA,colB / 0,2,5 / 2-9 / 混在可）")
    if expected_n is None:
        prompt = "特徴量列: "
    else:
        prompt = f"特徴量列（{expected_n}列）: "
    col_in = input(prompt).strip()
    feature_cols = _parse_cols_input(col_in, all_cols)

    # Extract X
    X = df[feature_cols].copy()
    if hasattr(rf, "n_features_in_"):
        if int(rf.n_features_in_) != len(feature_cols):
            print(
                f"[WARN] rf.n_features_in_={rf.n_features_in_} != selected_cols={len(feature_cols)}",
                file=sys.stderr,
            )

    # サンプリング数（対話入力を優先。空なら args.sample）
    sample_in = input("サンプリング数（空=--sampleの値。推奨: 20000）: ").strip()
    sample_n = int(sample_in) if sample_in else int(args.sample)

    # 目的変数列（層化に必要）
    print("\n目的変数列を指定（空=層化サンプリング不可。例: LandClass / code / 195 等）")
    y_in = input("目的変数列（空=層化なし/正解ベース集計スキップ）: ").strip()
    y_true = None
    if y_in:
        if y_in.isdigit():
            yi = int(y_in)
            if yi < 0 or yi >= len(all_cols):
                raise ValueError(f"目的変数列インデックス範囲外: {yi}")
            y_col = all_cols[yi]
        else:
            y_col = y_in
            if y_col not in all_cols:
                raise ValueError(f"目的変数列名が存在しません: {y_col}")
        # y_true は X と同じ index（df の index）で持つ
        y_true = df.loc[X.index, y_col].to_numpy()

    # Main sample (for quantiles / representative trees / votes etc.)
    Xs, y_true_s = _sample_X_y(X, y_true, sample_n, args.seed)

    # RFに渡す行列（Pipeline前処理がある場合は transform 後）
    if preprocess is not None:
        Xs_rf = preprocess[:-1].transform(Xs)
    else:
        Xs_rf = Xs.to_numpy()

    # Predict by RF on sampled
    rf_pred = rf.predict(Xs_rf)
    classes = getattr(rf, "classes_", np.unique(rf_pred))
    # y_true_s の型を rf_pred に合わせる（例: "1010101" vs 1010101 問題を回避）
    if y_true_s is not None:
        try:
            if y_true_s.dtype != rf_pred.dtype:
                # 文字列クラスの場合は str に寄せるのが安全
                if rf_pred.dtype.kind in ("O", "U", "S"):
                    y_true_s = y_true_s.astype(str)
                else:
                    y_true_s = y_true_s.astype(rf_pred.dtype)
        except Exception as e:
            print(f"[WARN] y_true_s dtype cast failed: {e}", file=sys.stderr)

    # --- メタJSONのラベル対応で y_true_s を rf.classes_ 側へ変換（例: 1010101 -> 0） ---
    if y_true_s is not None and meta_path is not None and meta_path.exists():
        # meta読込（診断用に一度パース）
        try:
            meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta_obj = json.loads(meta_path.read_text(encoding="utf-8-sig"))

        if args.meta_debug and isinstance(meta_obj, dict):
            print(f"[DIAG] meta top-level keys: {list(meta_obj.keys())}")

        # 既存のトップレベル探索 + 追加の再帰探索
        mp = _load_label_mapping_from_meta(meta_path, rf_classes=classes)
        if mp is None:
            mp = _meta_find_mapping_recursive(meta_obj)
        if mp is not None:
            y_m = pd.Series(y_true_s).map(mp)
            if y_m.isna().any():
                bad = pd.Series(y_true_s)[y_m.isna()].unique().tolist()
                print(f"[WARN] meta mapping missing labels (sample): {bad[:20]}", file=sys.stderr)
                print("[WARN] true-based集計をスキップし、rf_pred-onlyへフォールバックします。", file=sys.stderr)
                y_true_s = None
            else:
                y_true_s = y_m.to_numpy().astype(rf_pred.dtype, copy=False)
                print(f"[INFO] y_true mapped by meta json: {meta_path}")
        else:
            print(f"[WARN] meta json has no usable mapping keys: {meta_path}", file=sys.stderr)

     # --- 1) まず入れるべき診断（結論を出すため） ---
    _run_first_diagnostics(
        Xs=Xs,
        Xs_rf=Xs_rf,
        rf=rf,
        rf_pred=rf_pred,
        classes=classes,
        y_true_s=y_true_s,
    )
 
    # =========================================================
    # Tree text export mode (unified prompt)
    # - CLIで --tree_text_mode が指定されていればそれを優先
    # - ただし --tree_text_prompt があれば必ず聞く
    # =========================================================
    tree_text_mode = (args.tree_text_mode or "rep").strip().lower()
    if tree_text_mode not in ("rep", "all", "both"):
        tree_text_mode = "rep"
    argv_tree = " ".join(sys.argv[1:])
    tree_cli_specified = ("--tree_text_mode" in argv_tree)
    if (args.tree_text_prompt or (not tree_cli_specified)) and sys.stdin.isatty():
        print("\n決定木テキスト出力モード")
        print("  1) rep（従来：代表木のみ、クラス別複製あり）")
        print("  2) all（全木：tree_000.txt ... tree_049.txt）")
        print("  3) both（rep + all）")
        m = input("番号を選択 [1-3]（空=1）: ").strip() or "1"
        tree_text_mode = {"1": "rep", "2": "all", "3": "both"}.get(m, "rep")
    print(f"[INFO] tree_text_mode={tree_text_mode}")
 
    n_trees = len(getattr(rf, "estimators_", []))
    if n_trees == 0:
        raise ValueError("rf.estimators_ がありません。RandomForestClassifier を想定しています。")

    # (optional) y_true for contribution: 目的変数列はparquetに無い前提でも回るようにする
    # ただし「正解時に貢献した木」を出すには y_true が必要。
    # y_true は上で確定済み（または None）

    # ---------------------------
    # 既存: 分位点（予測クラス別）
    # ---------------------------
    q_list = []
    for c in classes:
        sub = Xs[rf_pred == c]
        if len(sub) == 0:
            continue
        q = sub.quantile([0.05, 0.5, 0.95]).T
        q.columns = ["p05", "p50", "p95"]
        q.insert(0, "class", c)
        q.insert(1, "feature", q.index)
        q_list.append(q.reset_index(drop=True))
    q_df = pd.concat(q_list, ignore_index=True) if q_list else pd.DataFrame()
    _save_df(q_df, outdir / "quantiles_by_predclass_p05_p50_p95.csv")

    # ---------------------------
    # 追加: 木×クラス貢献度 / 代表木抽出
    # ---------------------------
    # 各木の予測
    tree_preds = np.vstack([est.predict(Xs_rf) for est in rf.estimators_])  # (n_trees, n_samples)

    # 木の単体精度（y_trueがある場合）と、RF予測一致率（常に計算可能）
    rows_score = []
    for t in range(n_trees):
        agree_rf = float(np.mean(tree_preds[t] == rf_pred))
        if y_true_s is not None:
            acc = float(np.mean(tree_preds[t] == y_true_s))
        else:
            acc = np.nan
        rows_score.append({
            "tree_id": t,
            "tree_acc_vs_true": acc,
            "tree_agree_with_rf": agree_rf,
        })
    score_df = pd.DataFrame(rows_score)
    _save_df(score_df, outdir / "tree_global_scores.csv")

    # 貢献度（正解ベース）：RFが正解のとき、その木が正解に投票した回数（クラス別）
    contrib_rows = []
    rep_rows = []

    cand_rows_true = []   # NEW: ranking candidates (true-based)
    cand_rows_pred = []   # NEW: ranking candidates (rf_pred-only)

    # NEW: selection params (for reproducibility / criterion visibility)
    sel_param_rows = [{
        "mode": "true-based" if (y_true_s is not None) else "rf_pred-only",
        "rep_trees_per_class": int(args.rep_trees),
        "rep_candidates_per_class": int(args.rep_candidates),
        "sample_n_main": int(sample_n),
        "seed": int(args.seed),
    }]
    _save_df(pd.DataFrame(sel_param_rows), outdir / "representative_tree_selection_params.csv")

    if y_true_s is not None:
        rf_correct = (rf_pred == y_true_s)
        for c in classes:
            idx = np.where(rf_correct & (y_true_s == c))[0]
            if idx.size == 0:
                continue
            # そのクラスで正解したサンプルに対し、各木が正解クラスを出した回数
            votes = np.sum(tree_preds[:, idx] == c, axis=1)  # (n_trees,)
            for t in range(n_trees):
                contrib_rows.append({
                    "class": c,
                    "tree_id": t,
                    "correct_votes_when_rf_correct": int(votes[t]),
                    "n_samples_rf_correct_in_class": int(idx.size),
                    "rate": float(votes[t] / idx.size),
                })

            # 代表木：このクラスで correct_votes が大きい上位N本
            topn = int(args.rep_trees)
            order = np.argsort(votes)[::-1]  # full ranking
            top_ids = order[:topn]

            # NEW: export candidate ranking (criterion③ visualization)
            cand_k = int(args.rep_candidates)
            if cand_k <= 0:
                cand_k = n_trees
            cand_k = min(cand_k, n_trees)
            for r_all, t_all in enumerate(order[:cand_k], start=1):
                cand_rows_true.append({
                    "class": c,
                    "rank_all": int(r_all),
                    "selected": bool(r_all <= topn),
                    "tree_id": int(t_all),
                    "correct_votes_when_rf_correct": int(votes[t_all]),
                    "n_samples_rf_correct_in_class": int(idx.size),
                    "rate": float(votes[t_all] / idx.size),
                })
            for rank, t in enumerate(top_ids, start=1):
                rep_rows.append({
                    "class": c,
                    "rank": rank,
                    "tree_id": int(t),
                    "correct_votes_when_rf_correct": int(votes[t]),
                    "rate": float(votes[t] / idx.size),
                })

        contrib_df = pd.DataFrame(contrib_rows)
        _save_df(contrib_df, outdir / "tree_contrib_by_class.csv")

        if rep_rows:
            rep_df = pd.DataFrame(rep_rows).sort_values(["class", "rank"])
            _save_df(rep_df, outdir / "representative_trees_by_class.csv")
            # NEW: candidate ranking CSV (true-based)
            if cand_rows_true:
                cand_df = pd.DataFrame(cand_rows_true).sort_values(["class", "rank_all"])
                _save_df(cand_df, outdir / "representative_tree_candidates_by_class.csv")
        else:
            print("[WARN] representative trees not found in true-based mode. Fallback to rf_pred-only.", file=sys.stderr)
            rep_rows = []
            for c in classes:
                idx = np.where(rf_pred == c)[0]
                if idx.size == 0:
                    continue
                votes = np.sum(tree_preds[:, idx] == c, axis=1)
                topn = int(args.rep_trees)
                order = np.argsort(votes)[::-1]
                top_ids = order[:topn]

                # NEW: export candidate ranking (rf_pred-only)
                cand_k = int(args.rep_candidates)
                if cand_k <= 0:
                    cand_k = n_trees
                cand_k = min(cand_k, n_trees)
                for r_all, t_all in enumerate(order[:cand_k], start=1):
                    cand_rows_pred.append({
                        "class": c,
                        "rank_all": int(r_all),
                        "selected": bool(r_all <= topn),
                        "tree_id": int(t_all),
                        "votes_for_class_when_rf_pred_class": int(votes[t_all]),
                        "n_samples_rf_pred_in_class": int(idx.size),
                        "rate": float(votes[t_all] / idx.size),
                    })
                for rank, t in enumerate(top_ids, start=1):
                    rep_rows.append({
                        "class": c,
                        "rank": rank,
                        "tree_id": int(t),
                        "votes_for_class_when_rf_pred_class": int(votes[t]),
                        "rate": float(votes[t] / idx.size),
                    })
            rep_df = pd.DataFrame(rep_rows).sort_values(["class", "rank"])
            _save_df(rep_df, outdir / "representative_trees_by_class__by_rf_pred_only.csv")
            # NEW: candidate ranking CSV (rf_pred-only fallback)
            if cand_rows_pred:
                cand_df = pd.DataFrame(cand_rows_pred).sort_values(["class", "rank_all"])
                _save_df(cand_df, outdir / "representative_tree_candidates_by_class__by_rf_pred_only.csv")
    else:
        # y_true が無い場合は「RFの予測クラスごとに、そのクラスを投票した回数（頻度）」を代表として出す
        # これは“正解貢献”ではなく“出力傾向”になるためファイル名で分ける
        rep_rows = []
        for c in classes:
            idx = np.where(rf_pred == c)[0]
            if idx.size == 0:
                continue
            votes = np.sum(tree_preds[:, idx] == c, axis=1)
            topn = int(args.rep_trees)
            order = np.argsort(votes)[::-1]
            top_ids = order[:topn]

            # NEW: export candidate ranking (rf_pred-only)
            cand_k = int(args.rep_candidates)
            if cand_k <= 0:
                cand_k = n_trees
            cand_k = min(cand_k, n_trees)
            for r_all, t_all in enumerate(order[:cand_k], start=1):
                cand_rows_pred.append({
                    "class": c,
                    "rank_all": int(r_all),
                    "selected": bool(r_all <= topn),
                    "tree_id": int(t_all),
                    "votes_for_class_when_rf_pred_class": int(votes[t_all]),
                    "n_samples_rf_pred_in_class": int(idx.size),
                    "rate": float(votes[t_all] / idx.size),
                })
            for rank, t in enumerate(top_ids, start=1):
                rep_rows.append({
                    "class": c,
                    "rank": rank,
                    "tree_id": int(t),
                    "votes_for_class_when_rf_pred_class": int(votes[t]),
                    "rate": float(votes[t] / idx.size),
                })
        rep_df = pd.DataFrame(rep_rows).sort_values(["class", "rank"])
        _save_df(rep_df, outdir / "representative_trees_by_class__by_rf_pred_only.csv")
        # NEW: candidate ranking CSV (rf_pred-only)
        if cand_rows_pred:
            cand_df = pd.DataFrame(cand_rows_pred).sort_values(["class", "rank_all"])
            _save_df(cand_df, outdir / "representative_tree_candidates_by_class__by_rf_pred_only.csv")

    # 代表木の export_text + 閾値/特徴量使用集計
    from sklearn.tree import export_text

    trees_dir = outdir / "trees"
    trees_dir.mkdir(parents=True, exist_ok=True)

    # 代表木ID集合を決める
    rep_tree_ids = set()
    rep_df = None
    if (outdir / "representative_trees_by_class.csv").exists():
        rep_df = pd.read_csv(outdir / "representative_trees_by_class.csv")
        rep_tree_ids = set(rep_df["tree_id"].astype(int).tolist())
    elif (outdir / "representative_trees_by_class__by_rf_pred_only.csv").exists():
        rep_df = pd.read_csv(outdir / "representative_trees_by_class__by_rf_pred_only.csv")
        rep_tree_ids = set(rep_df["tree_id"].astype(int).tolist())

    rep_usage_rows = []
    rep_thr_rows = []

    # (A) all: 全木 export_text（クラス複製なし）
    if tree_text_mode in ("all", "both"):
        for t in range(n_trees):
            tree = rf.estimators_[t]
            txt = export_text(
                tree,
                feature_names=feature_cols,
                max_depth=int(args.tree_text_max_depth),
            )
            (trees_dir / f"tree_{t:03d}.txt").write_text(txt, encoding="utf-8")
        print(f"[INFO] tree_text_mode=all: exported {n_trees} tree texts -> {trees_dir}")

    # (B) rep: 代表木のみ解析し、クラス別ファイルとして複製
    rep_tree_cache = {}
    if tree_text_mode in ("rep", "both"):
        for t in rep_tree_ids:
            tree = rf.estimators_[t]
            txt = export_text(tree, feature_names=feature_cols, max_depth=int(args.tree_text_max_depth))
            usage, thresholds = _get_tree_feature_thresholds(tree, feature_cols)
            rep_tree_cache[t] = (txt, usage, thresholds)

            # usage集計        
            for feat, cnt in usage.items():
                rep_usage_rows.append({
                    "tree_id": int(t),
                    "feature": feat,
                    "split_count": int(cnt),
                })
            # threshold一覧
            for d in thresholds:
                rep_thr_rows.append({
                    "tree_id": int(t),
                    **d
                })

    # export_text をクラス別ファイルとして出力（代表木表の class と紐付け）
    if tree_text_mode in ("rep", "both") and rep_df is not None and len(rep_tree_cache) > 0:
        for _, r in rep_df.iterrows():
            c = r["class"]
            t = int(r["tree_id"])
            if t not in rep_tree_cache:
                continue
            txt, _, _ = rep_tree_cache[t]
            out_txt = trees_dir / f"tree_{t:03d}__class_{c}.txt"
            out_txt.write_text(txt, encoding="utf-8")
        print(f"[INFO] tree_text_mode=rep: exported representative tree texts -> {trees_dir}")

    if tree_text_mode in ("rep", "both"):
        if rep_usage_rows:
            rep_usage_df = pd.DataFrame(rep_usage_rows).sort_values(["tree_id", "split_count"], ascending=[True, False])
            _save_df(rep_usage_df, outdir / "rep_tree_feature_usage.csv")

        if rep_thr_rows:
            rep_thr_df = pd.DataFrame(rep_thr_rows).sort_values(["tree_id", "depth", "feature"])
            _save_df(rep_thr_df, outdir / "rep_tree_thresholds.csv")

    # ---------------------------
    # SHAP (selectable: none/light/full)
    # ---------------------------
    if args.no_shap or args.shap_mode == "none":
        print("\n=== DONE (SHAP disabled) ===")
        print(f"outdir: {outdir}")
        return

    try:
        import shap
    except Exception:
        print("[WARN] shap が import できません。SHAP出力はスキップします（他の出力は継続）。", file=sys.stderr)
        print("\n=== DONE (SHAP skipped) ===")
        print(f"outdir: {outdir}")
        return

    # build SHAP sample
    if args.shap_mode == "full":
        Xs_shap = Xs
        y_true_shap = y_true_s
        Xs_shap_rf = Xs_rf
        print(f"[INFO] SHAP mode=full (rows={len(Xs_shap)})")
    else:
        # light: separate small sample (keep stratified if y_true exists)
        shap_n = int(args.shap_sample)
        Xs_shap, y_true_shap = _sample_X_y(X, y_true, shap_n, args.seed)
        # apply meta mapping to y_true_shap (if possible) so class restriction can accept original labels
        if y_true_shap is not None and meta_path is not None and meta_path.exists():
            try:
                meta_obj2 = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta_obj2 = json.loads(meta_path.read_text(encoding="utf-8-sig"))
            mp2 = _load_label_mapping_from_meta(meta_path, rf_classes=classes)
            if mp2 is None:
                mp2 = _meta_find_mapping_recursive(meta_obj2)
            if mp2 is not None:
                y_m2 = pd.Series(y_true_shap).map(mp2)
                if (not y_m2.isna().any()):
                    y_true_shap = y_m2.to_numpy().astype(rf_pred.dtype, copy=False)
        if preprocess is not None:
            Xs_shap_rf = preprocess[:-1].transform(Xs_shap)
        else:
            Xs_shap_rf = Xs_shap.to_numpy()
        print(f"[INFO] SHAP mode=light (rows={len(Xs_shap)}, shap_sample={shap_n})")

    # class restriction (internal class ids, same as rf.classes_)
    class_list = list(getattr(rf, "classes_", []))
    want_classes = None
    if (args.shap_classes or "").strip():
        toks = [t.strip() for t in args.shap_classes.split(",") if t.strip()]
        wc = []
        for t in toks:
            try:
                wc.append(int(t))
            except Exception:
                # keep as string if needed
                wc.append(t)
        want_classes = wc
        # filter to existing
        want_classes = [c for c in want_classes if c in class_list]
        if not want_classes:
            print(f"[WARN] --shap_classes matched nothing. available={class_list}", file=sys.stderr)
            want_classes = None

    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(Xs_shap_rf)

    if isinstance(sv, list):
        sv_list = sv
        class_list = list(getattr(rf, "classes_", classes))
    else:
        sv_list = [sv]
        class_list = [classes[1] if hasattr(classes, "__len__") and len(classes) > 1 else "class_1"]

    # apply class restriction to SHAP outputs
    if want_classes is not None and isinstance(sv, list):
        idxs = [i for i, c in enumerate(class_list) if c in want_classes]
        sv_list = [sv_list[i] for i in idxs]
        class_list = [class_list[i] for i in idxs]
        print(f"[INFO] SHAP classes restricted: {class_list}")

    # mean abs shap per class -> csv
    rows = []
    for k, c in enumerate(class_list):
        s = sv_list[k]
        mean_abs = np.mean(np.abs(s), axis=0)
        order = np.argsort(mean_abs)[::-1]
        for rank, j in enumerate(order, start=1):
            rows.append({
                "class": c,
                "rank": rank,
                "feature": feature_cols[j],
                "mean_abs_shap": float(mean_abs[j]),
            })
    imp_df = pd.DataFrame(rows)
    _save_df(imp_df, outdir / "shap_mean_abs_by_class.csv")

    # summary plot per class
    for k, c in enumerate(class_list):
        plt.figure()
        shap.summary_plot(sv_list[k], Xs_shap, feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(outdir / f"shap_summary_class_{c}.png", dpi=200)
        plt.close()

    # dependence plots: top-k + threshold estimation
    thr_rows = []
    for k, c in enumerate(class_list):
        sub_imp = imp_df[imp_df["class"] == c].sort_values("mean_abs_shap", ascending=False)
        top_feats = sub_imp["feature"].head(int(args.topk)).tolist()

        for feat in top_feats:
            j = feature_cols.index(feat)
            plt.figure()
            shap.dependence_plot(feat, sv_list[k], Xs_shap, feature_names=feature_cols, show=False)
            plt.tight_layout()
            plt.savefig(outdir / f"shap_dependence_class_{c}__{feat}.png", dpi=200)
            plt.close()

            est = estimate_zero_cross_thresholds(
                x=Xs_shap[feat].to_numpy(),
                shap_v=sv_list[k][:, j],
                n_bins=40,
            )
            thr_rows.append({
                "class": c,
                "feature": feat,
                "threshold_zero_cross": est["threshold"],
                "note": est["note"],
            })

    thr_df = pd.DataFrame(thr_rows)
    _save_df(thr_df, outdir / "thresholds_zero_cross_by_class.csv")

    print("\n=== DONE ===")
    print(f"outdir: {outdir}")
    print("generated:")
    print(" - quantiles_by_predclass_p05_p50_p95.csv")
    print(" - shap_mean_abs_by_class.csv")
    print(" - shap_summary_class_*.png")
    print(" - shap_dependence_class_*__*.png")
    print(" - thresholds_zero_cross_by_class.csv")
    print(" - tree_global_scores.csv")
    if y_true is not None:
        print(" - tree_contrib_by_class.csv")
        print(" - representative_trees_by_class.csv")
    else:
        print(" - representative_trees_by_class__by_rf_pred_only.csv")
    print(" - trees/tree_XXX__class_YYY.txt")
    print(" - rep_tree_feature_usage.csv")
    print(" - rep_tree_thresholds.csv")


if __name__ == "__main__":
    main()
