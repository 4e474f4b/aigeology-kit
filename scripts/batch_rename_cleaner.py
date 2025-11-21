#!/usr/bin/env python3
"""
あるディレクトリ以下のファイル名から、指定した文字列（複数可）を削除してリネームするスクリプト。

機能:
- ファイル名に含まれる特定の文字列（複数可）を削除
- さらに、ファイル名に対して
    1) 単純に「先頭 / 末尾 / 先頭＆末尾」に文字列を追加 する、
    2) または「ファイル名に特定の文字列を含む場合のみ」追加する、
  のいずれかを選び、一括リネームする。

例）削除対象: "Cliped-"
Clipped-WS025-Cliped-1mDEM_epsg6673_relief_r12m.tif
→ Clipped-WS025-1mDEM_epsg6673_relief_r12m.tif

例）削除対象: "Cliped-,_tmp"
AAA_Cliped-TEST_tmp.txt
→ AAA_TEST.txt

例）無条件で末尾に "_R12m" を追加:
Clipped-WS025-1mDEM_epsg6673_relief.tif
→ Clipped-WS025-1mDEM_epsg6673_relief_R12m.tif

例）「_relief を含む場合のみ」末尾に "_R12m" を追加:
WS025-1mDEM_epsg6673_relief.tif    → WS025-1mDEM_epsg6673_relief_R12m.tif
WS025-1mDEM_epsg6673_slope.tif     → （変更なし）
"""

from pathlib import Path


def ask_directory() -> Path:
    """対象ディレクトリを対話的に取得する。"""
    while True:
        dir_str = input('対象ディレクトリのパスを入力してください（ドラッグ＆ドロップ可）: ').strip()

        # ドラッグ＆ドロップ時のクォート除去
        if (dir_str.startswith('"') and dir_str.endswith('"')) or \
           (dir_str.startswith("'") and dir_str.endswith("'")):
            dir_str = dir_str[1:-1]

        p = Path(dir_str).expanduser()

        if not p.exists():
            print(f'[ERROR] パスが存在しません: {p}')
            continue
        if not p.is_dir():
            print(f'[ERROR] ディレクトリではありません: {p}')
            continue
        return p


def ask_remove_patterns():
    """
    削除したい文字列をカンマ区切りで入力してもらう。
    例: Cliped-,_tmp,TEST-
    空Enterなら「削除は行わない」。
    """
    s = input(
        'ファイル名から「削除したい文字列」があればカンマ区切りで入力してください\n'
        '例: Cliped-,_tmp,TEST-  （空Enterで削除処理なし）\n> '
    ).strip()
    if not s:
        print('[INFO] 削除処理は行いません。')
        return []

    parts = [p.strip() for p in s.split(',')]
    patterns = [p for p in parts if p]  # 空文字は除外
    if not patterns:
        print('[INFO] 有効な削除文字列がないため、削除処理は行いません。')
        return []

    print('\n[INFO] 削除対象の文字列一覧:')
    for i, pat in enumerate(patterns, start=1):
        print(f'  {i}: "{pat}"')

    ans = input('\nこの文字列を削除対象としてよろしいですか？ [y/N]: ').strip().lower()
    if ans == 'y':
        return patterns
    else:
        print('[INFO] 削除処理は行いません。')
        return []


def ask_insert_option():
    """
    ファイル名への文字列追加について問い合わせる。

    戻り値:
        insert_source: "fixed" / "parent" / None
                       "fixed"  → 固定文字列を使う
                       "parent" → 各ファイルの親ディレクトリ名を使う
                       None     → 追加しない
        insert_str   : insert_source == "fixed" のときに使う文字列
        position     : "head" / "tail" / "both" / None
        cond_patterns: 追加条件となるパターンのリスト
                       None の場合は「条件なし（全ファイル対象）」を意味する。
        cond_mode    : "or" / "and" / None
                       cond_patterns が None の場合は None。
    """
    while True:
        print('\n[追加オプション] ファイル名に文字列を追加しますか？')
        print('  0) 追加しない')
        print('  1) ファイル名の先頭に追加')
        print('  2) ファイル名の末尾に追加（拡張子の手前）')
        print('  3) ファイル名の先頭および末尾に追加（拡張子の前）')
        choice = input('番号を選んでください [0-3]: ').strip()

        if choice == '':
            choice = '0'

        if choice not in {'0', '1', '2', '3'}:
            print('[ERROR] 0〜3 のいずれかを入力してください。')
            continue

        if choice == '0':
            print('[INFO] 文字列の追加は行いません。')
            return None, None, None, None, None

        # 位置の決定
        if choice == '1':
            pos = 'head'
        elif choice == '2':
            pos = 'tail'
        else:
            pos = 'both'

        # 追加文字列の種類（固定 / 親ディレクトリ名）
        while True:
            print('\n[追加文字列の種類]')
            print('  1) 固定の文字列を指定する')
            print('  2) 各ファイルの親ディレクトリ名を使う')
            src_choice = input('番号を選んでください [1-2]: ').strip()

            if src_choice not in {'1', '2'}:
                print('[ERROR] 1 または 2 を入力してください。')
                continue

            if src_choice == '1':
                insert_source = 'fixed'
                ins = input('ファイル名に追加する文字列を入力してください（空Enterでキャンセル）:\n> ').strip()
                if not ins:
                    print('[INFO] 文字列が空のため、追加処理は行いません。')
                    return None, None, None, None, None
            else:
                insert_source = 'parent'
                ins = None  # 親ディレクトリ名を都度使うのでここでは不要

            # 条件設定：無条件 or 特定文字列を含む場合のみ
            while True:
                print('\n[条件設定] どのファイルに追加しますか？')
                print('  1) すべての対象ファイルに追加する')
                print('  2) ファイル名に特定の文字列を含む場合のみ追加する')
                cond_choice = input('番号を選んでください [1-2]: ').strip()

                if cond_choice not in {'1', '2'}:
                    print('[ERROR] 1 または 2 を入力してください。')
                    continue

                if cond_choice == '1':
                    cond_patterns = None  # 無条件
                    cond_mode = None
                    cond_label = 'すべての対象ファイル'
                else:
                    s = input(
                        '「追加条件」とする文字列をカンマ区切りで入力してください\n'
                        '（例: _relief,_slope。空Enterで条件設定をやり直し）\n> '
                    ).strip()
                    if not s:
                        print('[INFO] 条件文字列が空です。もう一度条件を設定してください。')
                        continue
                    parts = [p.strip() for p in s.split(',')]
                    cond_patterns = [p for p in parts if p]
                    if not cond_patterns:
                        print('[INFO] 有効な条件文字列がないため、もう一度条件を設定してください。')
                        continue

                    # AND / OR の選択
                    while True:
                        print('\n[条件の組み合わせ方法]')
                        print('  1) OR条件（いずれかを含めば追加する）')
                        print('  2) AND条件（すべて含む場合のみ追加する）')
                        mode_choice = input('番号を選んでください [1-2]: ').strip()
                        if mode_choice == '1':
                            cond_mode = 'or'
                            op_label = 'OR（いずれかを含む）'
                            break
                        elif mode_choice == '2':
                            cond_mode = 'and'
                            op_label = 'AND（すべてを含む）'
                            break
                        else:
                            print('[ERROR] 1 または 2 を入力してください。')

                    joined = ', '.join(f'"{p}"' for p in cond_patterns)
                    cond_label = f'ファイル名に {joined} を {op_label} 条件で含む場合のみ'

                # 確認表示
                if pos == "head":
                    pos_label = "先頭"
                elif pos == "tail":
                    pos_label = "末尾（拡張子の前）"
                else:
                    pos_label = "先頭および末尾（拡張子の前）"

                if insert_source == 'fixed':
                    src_label = f'固定文字列 "{ins}"'
                else:
                    src_label = '親ディレクトリ名'

                print('\n[INFO] 追加設定:')
                print(f'  位置: {pos_label}')
                print(f'  文字列の種類: {src_label}')
                if insert_source == 'fixed':
                    print(f'  実際に追加する文字列: "{ins}"')
                print(f'  条件: {cond_label}')
                ans = input('この設定でよろしいですか？ [y/N]: ').strip().lower()
                if ans == 'y':
                    return insert_source, ins, pos, cond_patterns, cond_mode
                else:
                    print('[INFO] もう一度、追加設定をやり直します。')
                    break  # 条件設定からやり直し（src_choiceのところには戻らない）

            # 条件設定でキャンセルされた場合は、もう一度「文字列の種類」からやり直す
            # → while True の先頭に戻る


def apply_patterns(name: str, patterns) -> str:
    """ファイル名に対して、指定パターンをすべて削除した新しい名前を返す。"""
    new_name = name
    for pat in patterns:
        if pat:
            new_name = new_name.replace(pat, "")
    return new_name


def apply_insert(name: str, insert_str: str | None, position: str | None) -> str:
    """
    ファイル名に insert_str を追加した新しい名前を返す。
    position: "head" → 先頭, "tail" → 末尾（拡張子の手前）, "both" → 先頭＆末尾
    """
    if not insert_str or not position:
        return name

    # 拡張子を分離（最後の . を基準）
    dot_idx = name.rfind(".")
    if dot_idx == -1:
        base = name
        ext = ""
    else:
        base = name[:dot_idx]
        ext = name[dot_idx:]

    if position == "head":
        new_base = insert_str + base
    elif position == "tail":
        new_base = base + insert_str
    elif position == "both":
        new_base = insert_str + base + insert_str
    else:
        # 想定外の値が来た場合は無変更
        return name

    return new_base + ext


def main():
    print("=== ファイル名から指定文字列（複数可）を削除し、無条件または条件付きで文字列を追加してリネームするスクリプト ===")

    target_dir = ask_directory()
    remove_patterns = ask_remove_patterns()
    insert_source, insert_str, insert_pos, cond_patterns, cond_mode = ask_insert_option()

    if not remove_patterns and insert_source is None:
        print('\n[INFO] 削除も追加も指定されていないため、処理を終了します。')
        return

    print(f'\n[INFO] 対象ディレクトリ: {target_dir}')
    if remove_patterns:
        print('[INFO] 削除対象文字列:', ', '.join(f'"{p}"' for p in remove_patterns))
    else:
        print('[INFO] 削除対象文字列: なし')

    if insert_source is not None:
        if insert_pos == "head":
            pos_label = "先頭"
        elif insert_pos == "tail":
            pos_label = "末尾（拡張子の前）"
        else:
            pos_label = "先頭および末尾（拡張子の前）"

        if cond_patterns is None:
            cond_label = "すべての対象ファイル"
        else:
            if cond_mode == 'and':
                op_label = 'AND（すべてを含む）'
            else:
                op_label = 'OR（いずれかを含む）'
            joined = ', '.join(f'"{p}"' for p in cond_patterns)
            cond_label = f'ファイル名に {joined} を {op_label} 条件で含む場合のみ'

        if insert_source == 'fixed':
            src_label = f'固定文字列 "{insert_str}"'
        else:
            src_label = '親ディレクトリ名'

        print(f'[INFO] 追加: {src_label} （位置: {pos_label}, 条件: {cond_label}）')
    else:
        print('[INFO] 追加文字列: なし')

    # 対象拡張子（必要ならここに ".gpkg" など足してOK）
    exts = {".tif", ".tiff"}

    # 再帰的にファイルを探索
    candidates = []
    for p in target_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue

        old_name = p.name
        new_name = old_name

        # 1) 削除処理
        if remove_patterns:
            new_name = apply_patterns(new_name, remove_patterns)

        # 2) 追加処理（無条件 or 条件付き）
        if insert_source is not None:
            # 挿入に使う文字列を決める
            if insert_source == 'fixed':
                current_insert = insert_str
            else:
                # 親ディレクトリ名を使用
                current_insert = p.parent.name

            # 条件判定
            if cond_patterns is None:
                # 条件なし（すべての対象ファイル）
                should_add = True
            else:
                # 条件あり：AND / OR で評価
                if cond_mode == 'and':
                    should_add = all(cp in new_name for cp in cond_patterns)
                else:
                    # デフォルトは OR
                    should_add = any(cp in new_name for cp in cond_patterns)

            if should_add:
                new_name = apply_insert(new_name, current_insert, insert_pos)

        if new_name != old_name:
            candidates.append((p, old_name, new_name))

    if not candidates:
        print('[INFO] リネーム対象となるファイル（かつ対象拡張子）は見つかりませんでした。')
        return

    print(f'\n[INFO] リネーム候補ファイル数: {len(candidates)} 件')

    # プレビュー表示
    print('\n--- リネームプレビュー（最大20件） ---')
    for i, (_, old_name, new_name) in enumerate(candidates[:20], start=1):
        print(f'{i:2d}) {old_name}  →  {new_name}')

    if len(candidates) > 20:
        print(f'  ... 他 {len(candidates) - 20} 件')

    ans = input('\nこの内容で実際にリネームを実行しますか？ [y/N]: ').strip().lower()
    if ans != 'y':
        print('キャンセルしました。')
        return

    # 実リネーム
    renamed = 0
    skipped_exists = 0
    errors = 0

    for path, old_name, new_name in candidates:
        new_path = path.with_name(new_name)

        if new_path.exists():
            print(f'[WARN] 既に同名ファイルが存在するためスキップ: {new_path}')
            skipped_exists += 1
            continue

        try:
            path.rename(new_path)
            renamed += 1
        except Exception as e:
            print(f'[ERROR] リネーム失敗: {path} → {new_path} ({e})')
            errors += 1

    print('\n=== 結果 ===')
    print(f'  リネーム実行: {renamed} 件')
    print(f'  同名ファイルがありスキップ: {skipped_exists} 件')
    print(f'  エラー: {errors} 件')


if __name__ == "__main__":
    main()
