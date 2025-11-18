"""Load model bundle and provide prediction helper.

Behaviour:
 - normalize input
 - if normalized text is in lookup -> return exact-match labels
 - else -> use model pipeline to predict
"""
import os
import joblib
from src.preprocess import normalize_text


MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.joblib')
MODEL_PATH = os.path.abspath(MODEL_PATH)


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run the training script first.")
    bundle = joblib.load(path)
    pipeline = bundle['pipeline']
    label_cols = bundle['label_cols']
    lookup = bundle.get('lookup', {})
    return pipeline, label_cols, lookup


_pipeline = None
_label_cols = None
_lookup = None


def ensure_loaded():
    global _pipeline, _label_cols, _lookup
    if _pipeline is None:
        _pipeline, _label_cols, _lookup = load_model()


def predict_text(text: str):
    """Return dict: {match: bool, labels: {label_name: 0/1}}
    Exact match is preferred.
    """
    ensure_loaded()
    norm = normalize_text(text)
    if norm in _lookup:
        vals = list(map(int, _lookup[norm]))
        return {'match': True, 'labels': dict(zip(_label_cols, vals))}

    y = _pipeline.predict([norm])[0]
    y = list(map(int, y))
    return {'match': False, 'labels': dict(zip(_label_cols, y))}


if __name__ == '__main__':
    sample = "I will kill you"
    print(predict_text(sample))
