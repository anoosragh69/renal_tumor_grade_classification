'''Top‑16 radiomic feature selection per CV fold (train‑only)'''

import json
from pathlib import Path
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

def load_radiomics() -> pd.DataFrame:
    csv_path = Path(__file__).resolve().parents[2] / 'data' / 'processed' / 'radiomics_all.csv'
    return pd.read_csv(csv_path)

def load_folds() -> list:
    folds_path = Path(__file__).resolve().parents[2] / 'data' / 'processed' / 'cv_folds.json'
    with open(folds_path, 'r') as f:
        folds_dict = json.load(f)
    # sort to keep order
    return [value for _, value in sorted(folds_dict.items(), key=lambda kv: kv[0])]

def select_top_k(X, y, k=16):
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    mask = selector.get_support()
    return list(X.columns[mask])

def main():
    df = load_radiomics()
    folds = load_folds()
    feature_cols = [c for c in df.columns if c not in {'patient_id', 'slice_idx', 'label'}]
    per_fold = {}
    union_features = set()
    for i, fold in enumerate(folds):
        train_patients = set(fold['train'])
        train_df = df[df['patient_id'].isin(train_patients)]
        y = train_df['label'].values
        X = train_df[feature_cols]
        selected = select_top_k(X, y, k=16)
        per_fold[f'fold_{i}'] = selected
        union_features.update(selected)
        print(f'Fold {i}: selected {len(selected)} features')
    result = per_fold.copy()
    result['union'] = list(sorted(union_features))
    out_path = Path(__file__).resolve().parents[2] / 'data' / 'processed' / 'top16_features.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved top-16 selection to {out_path}')

if __name__ == '__main__':
    main()
