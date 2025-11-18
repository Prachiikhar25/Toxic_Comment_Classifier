# Data Folder

- `comments_sample.csv`: A tiny, sanitized sample for smoke tests and demos. It contains the required columns: `comment_text,toxic,severe_toxic,obscene,threat,insult,identity_hate`.
- Full/large datasets should be stored locally under `data/` but are ignored by git to keep the repository lightweight.

## Using your own dataset

1. Prepare a CSV with columns:
   - `comment_text`
   - `toxic,severe_toxic,obscene,threat,insult,identity_hate` (0/1 per label)
2. Train using any of the following:
   - `python src/train.py --csv path/to/your.csv`
   - `setx DATA_CSV_PATH path\\to\\your.csv` (then restart shell) or in PowerShell session: `$env:DATA_CSV_PATH = "path/to/your.csv"`; then run `python src/train.py`.

Notes:
- The sample is intentionally small and may not yield strong model performance. Use your full dataset for meaningful results.
- The exact-match lookup prioritizes your provided data when an input exactly matches a normalized comment.
