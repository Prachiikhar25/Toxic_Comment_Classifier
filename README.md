# Toxic Comment Classifier (minimal)

This small project demonstrates:

- preprocessing of comment text
- building an exact-match lookup from your dataset
- training a multi-label classifier (One-vs-Rest LogisticRegression)
- a Flask web interface to classify new text

Files created:

- `src/preprocess.py` - normalization utilities
- `src/train.py` - training script (saves `models/model.joblib`)
- `src/predict.py` - loads model bundle and exposes `predict_text`
- `app.py` - minimal Flask UI
- `requirements.txt` - Python deps

Quick start (Windows PowerShell):

1. Create and activate a virtualenv (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Place your dataset at `data/comments.csv`. It must have columns:
   `comment_text,toxic,severe_toxic,obscene,threat,insult,identity_hate`

4. Train the model and build the exact-match lookup:

```powershell
python src/train.py
```

5. Run the Flask app:

```powershell
python app.py
```

6. Open http://127.0.0.1:5000 in your browser and enter a comment.

Behavior notes:

- If the normalized input exactly matches a normalized comment from your CSV, the app returns the labels from the CSV (exact-match lookup) — this respects your requirement to classify sentences according to the provided data first.
- If there is no exact match, the trained model is used to predict labels.

Next steps / improvements:

- Add model persistence path customization via env vars/CLI args.
- Add more preprocessing (lemmatization) if desired.
- Add unit tests and a small example `data/comments.csv` subset for smoke tests.
