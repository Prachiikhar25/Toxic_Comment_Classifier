"""Train a multi-label classifier and create an exact-match lookup.

Usage:
        python src/train.py [--csv PATH] [--out MODEL_PATH]

CSV columns required:
    - comment_text
    - toxic,severe_toxic,obscene,threat,insult,identity_hate

CSV selection order (lowest friction first):
    1) CLI arg --csv
    2) Env var DATA_CSV_PATH
    3) `data/comments_sample.csv` if it exists
    4) Fallback to `data/comments.csv`

Outputs a model bundle to `models/model.joblib` by default.
"""
import os
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from src.preprocess import normalize_text


def main(csv_path='data/comments.csv', model_out='models/model.joblib'):
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}. Place your dataset at that path.")

    df = pd.read_csv(csv_path)
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for c in label_cols:
        if c not in df.columns:
            raise ValueError(f"Required column missing in CSV: {c}")

    df['comment_text'] = df['comment_text'].fillna('')
    # normalized text will be used for lookup and modeling
    df['normalized'] = df['comment_text'].apply(normalize_text)

    # build exact-match lookup: normalized -> list of label ints
    lookup = {row['normalized']: row[label_cols].astype(int).tolist() for _, row in df.iterrows()}

    X = df['normalized']
    y = df[label_cols].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000)))
    ])

    print('Training model...')
    pipeline.fit(X_train, y_train)

    print('Evaluating on test set...')
    y_pred = pipeline.predict(X_test)
    try:
        print(classification_report(y_test, y_pred, target_names=label_cols))
    except Exception:
        # if sklearn version differences, just ignore
        pass

    bundle = {'pipeline': pipeline, 'label_cols': label_cols, 'lookup': lookup}
    joblib.dump(bundle, model_out)
    print('Saved model bundle to', model_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train toxic comment classifier with exact-match lookup')
    parser.add_argument('--csv', dest='csv_path', default=None, help='Path to input CSV')
    parser.add_argument('--out', dest='model_out', default='models/model.joblib', help='Output model bundle path')
    args = parser.parse_args()

    chosen_csv = (
        args.csv_path
        or os.environ.get('DATA_CSV_PATH')
        or ('data/comments_sample.csv' if os.path.exists('data/comments_sample.csv') else None)
        or 'data/comments.csv'
    )
    main(csv_path=chosen_csv, model_out=args.model_out)
