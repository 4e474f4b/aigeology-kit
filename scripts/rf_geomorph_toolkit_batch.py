#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rf_geomorph_toolkit.py の 2) / 3) を「そのまま」呼び出すバッチランチャー。

- 機能1: 軽量スイープ
    - RFだけ / XGBだけ / 両方 を選択
    - 何回くり返すかを指定
    - 各ループで rf_geomorph_toolkit.train_mode() を呼ぶ
    - preconfig={"disable_eval_gpkg": True, ...} を渡して GPKG だけ強制OFF
      （train_mode が preconfig 非対応な場合は、自動でフォールバックして単純呼び出し）

- 機能2: 本番精査バッチ
    - RFだけ / XGBだけ / 両方 を選択
    - 何回くり返すかを指定
    - 各ループで rf_geomorph_toolkit.train_mode() をそのまま呼ぶ
    - 学習曲線 / GPKG などの中の質問は、通常どおりプロンプトで聞かれる
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# 同じディレクトリに rf_geomorph_toolkit.py がある前提
# （aigeology-kit/scripts 配下で実行を想定）
try:
    from rf_geomorph_toolkit import train_mode
except Exception as e:
    print("⚠ rf_geomorph_toolkit から train_mode を import できません。")
    print("  - このファイルは rf_geomorph_toolkit.py と同じ scripts ディレクトリに置いてください。")
    print(f"  詳細: {e}")
    sys.exit(1)


# ------------------------------------------------------------
# 共通: backend の選択
# ------------------------------------------------------------
def select_backends_for_batch() -> list[str]:
    """
    バッチ実行で使用する backend を選ぶ。
      - "rf"  : RandomForest / CPU
      - "xgb" : XGBoost     / GPU（環境次第）
    """
    print("\n[バックエンドの選択]")
    print("  1) RandomForest だけ（CPU）")
    print("  2) XGBoost だけ（GPU / CPU 自動）")
    print("  3) 両方（RF → XGB の順に実行）")
    ans = input("番号を選んでください [1-3]（空=1）: ").strip() or "1"

    if ans == "2":
        return ["xgb"]
    elif ans == "3":
        return ["rf", "xgb"]
    else:
        return ["rf"]


def ask_num_runs(prompt: str = "何回くり返しますか？（例: 3）") -> int:
    """
    バッチの繰り返し回数を聞く。1 回以上。
    """
    while True:
        s = input(f"{prompt}（空=1）: ").strip()
        if not s:
            return 1
        try:
            v = int(s)
            if v <= 0:
                print("  ⚠ 0 以下は無効です。1 以上の整数を指定してください。")
                continue
            return v
        except ValueError:
            print("  ⚠ 整数として解釈できません。もう一度入力してください。")


def ask_base_seed() -> Optional[int]:
    """
    グローバル random_state を「固定 or ループ毎に変える」ためのベースシードを聞く。
    preconfig["global_random_state"] に入れる想定。
    train_mode 側が preconfig 非対応なら無視される。
    """
    print("\n[ランダムシード設定（任意）]")
    print("  ・ホールドアウト / k-Fold の random_state をそろえたい場合は整数を指定してください。")
    print("  ・ループごとにシードをずらす場合は、ここでベース値だけ決めておきます。")
    s = input("ベース random_state（空=未指定・毎回 train_mode 側に任せる）: ").strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        print("  ⚠ 整数として解釈できなかったため、random_state は未指定とします。")
        return None


# ------------------------------------------------------------
# train_mode を「preconfig 付きで安全に呼ぶ」ためのラッパ
# ------------------------------------------------------------
def call_train_mode_safe(backend: str, preconfig: Optional[Dict[str, Any]] = None) -> None:
    """
    train_mode(backend, preconfig=...) を呼ぶが、
    手元の rf_geomorph_toolkit が preconfig 非対応でも落ちないように TypeError を吸収してフォールバックする。
    """
    try:
        # preconfig 未指定ならそのまま
        if preconfig is None:
            train_mode(backend=backend)
        else:
            train_mode(backend=backend, preconfig=preconfig)
    except TypeError as e:
        # 古い版の train_mode(backend) には preconfig 引数が存在しない場合がある
        msg = str(e)
        if "preconfig" in msg or "unexpected keyword argument 'preconfig'" in msg:
            print("\n[WARN] 現在の rf_geomorph_toolkit.train_mode() は preconfig 未対応のようです。")
            print("       disable_eval_gpkg などの事前設定は効かず、通常の対話フローで実行します。")
            print("       （バッチとしては backend だけ切り替える最低限の動作になります）")
            train_mode(backend=backend)
        else:
            # 別の TypeError はそのまま投げる
            raise


# ------------------------------------------------------------
# 機能1: 軽量スイープ（学習曲線あり / 評価GPKGなし）
# ------------------------------------------------------------
def run_light_sweep():
    """
    機能1: 「軽量スイープ」

    - GPKG 出力は常に OFF（train_mode 側で '評価用 GPKG を出力しますか？' をスキップ）
    - 学習曲線は train_mode 内の質問に従う（Y を選べば普通に学習曲線を出す）
    - それ以外のプロンプト（入力パス・特徴量列・検証方式・ハイパラなど）は
      rf_geomorph_toolkit.py の 2) / 3) とまったく同じ挙動。

    ※ つまり「中身は完全に本物」を呼ぶだけで、GPKG だけ強制 OFF にした状態を
       指定回数くり返すランチャー。
    """
    print("\n=== 機能 1) 軽量スイープ（RF/XGB） ===")
    backends = select_backends_for_batch()
    n_runs = ask_num_runs("何回くり返しますか？（同じ条件で繰り返し実行）")
    base_seed = ask_base_seed()

    print("\n[INFO] これから軽量スイープを開始します。")
    print("      ※ 各ループ内では rf_geomorph_toolkit.train_mode() をそのまま呼びます。")
    print("      ※ 評価用 GPKG の質問は preconfig.disable_eval_gpkg=True によりスキップされます。")
    print("      ※ それ以外の質問（入力パスなど）は毎回、toolkit 側のプロンプトがそのまま出ます。\n")

    for run_idx in range(1, n_runs + 1):
        print("\n" + "=" * 70)
        print(f"[RUN {run_idx}/{n_runs}]")
        print("=" * 70)

        # ループごとに preconfig を組み立て
        for backend in backends:
            print(f"\n--- backend = {backend} ---")

            preconfig: Dict[str, Any] = {
                # 軽量スイープなので「評価用 GPKG は強制 OFF」
                "disable_eval_gpkg": True,
            }
            # ベースシードが指定されていれば、ループ毎に少しずらして渡す
            if base_seed is not None:
                preconfig["global_random_state"] = base_seed + (run_idx - 1)

            call_train_mode_safe(backend=backend, preconfig=preconfig)

    print("\n[完了] 機能1: 軽量スイープが終了しました。")


# ------------------------------------------------------------
# 機能2: 本番精査バッチ（学習曲線 + GPKGあり）
# ------------------------------------------------------------
def run_full_batch():
    """
    機能2: 「本番精査バッチ」

    - 中身は 2) / 3) とまったく同じ（学習曲線も GPKG も含めてフル機能）
    - RF だけ / XGB だけ / 両方 を選択し、指定回数くり返し実行するだけ。

    ※ GPKG ON/OFF や学習曲線 ON/OFF は train_mode 内の質問にそのまま従う。
      （ここからは何も抑制しない）
    """
    print("\n=== 機能 2) 本番精査バッチ（RF/XGB） ===")
    backends = select_backends_for_batch()
    n_runs = ask_num_runs("何回くり返しますか？（本番条件で繰り返し実行）")
    base_seed = ask_base_seed()

    print("\n[INFO] これから本番精査バッチを開始します。")
    print("      ※ 各ループ内では rf_geomorph_toolkit.train_mode() をそのまま呼びます。")
    print("      ※ 評価用 GPKG / 学習曲線の挙動も、toolkit 側の質問に完全に従います。\n")

    for run_idx in range(1, n_runs + 1):
        print("\n" + "=" * 70)
        print(f"[RUN {run_idx}/{n_runs}]")
        print("=" * 70)

        for backend in backends:
            print(f"\n--- backend = {backend} ---")
            preconfig: Optional[Dict[str, Any]] = None

            # ベースシードが指定されていれば、ループ毎にずらして渡す
            if base_seed is not None:
                preconfig = {"global_random_state": base_seed + (run_idx - 1)}

            call_train_mode_safe(backend=backend, preconfig=preconfig)

    print("\n[完了] 機能2: 本番精査バッチが終了しました。")


# ------------------------------------------------------------
# メインメニュー
# ------------------------------------------------------------
def main():
    print("\n=== RF/XGB 学習バッチランチャー（rf_geomorph_toolkit 用） ===")
    print("  1) 軽量スイープ（学習曲線あり / 評価GPKGなし）")
    print("  2) 本番精査バッチ（学習曲線 + GPKG あり）")
    print("  0) 終了")

    choice = input("番号を選んでください [0-2]: ").strip() or "0"

    if choice == "1":
        run_light_sweep()
    elif choice == "2":
        run_full_batch()
    else:
        print("終了します。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中断] Ctrl+C が押されたため終了しました。")
