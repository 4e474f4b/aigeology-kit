# aigeology-kit

AI × 地質・地形解析のためのユーティリティ集。
バグ報告・機能追加の提案などがあれば、Issue や Pull Request で教えてください。

## ディレクトリ構成

- `scripts/`
  - `clip_all_tifs_same_extent_interactive.py`
  - `make_features_RSandSM_export_points.py`
  - `rf_geomorph_interactive.py`
- `src/`
  - `aigeology_kit/` （共通関数を今後まとめる予定）
- `docs/`
- `tests/`

## 想定している使い方（メモ）

- 地形特徴量生成（RS/SM 系）
- ランダムフォレストによる地形分類
- DEM / GeoTIFF の前処理 など
