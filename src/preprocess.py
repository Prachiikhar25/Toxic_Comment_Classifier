"""Simple text preprocessing utilities.

We normalize comments for exact-matching and for model input.
"""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def normalize_text(text: str) -> str:
    """Normalize a text string:

    - lowercase
    - remove urls and @mentions
    - remove non-alphanumeric characters (keeps apostrophes)
    - remove stopwords (scikit-learn's English stop words)
    - collapse whitespace
    """
    if text is None:
        return ""
    # ensure string
    text = str(text)
    text = text.lower()
    # remove urls
    text = re.sub(r"http\S+|www\.\S+|https\S+", "", text)
    # remove mentions
    text = re.sub(r"@\w+", "", text)
    # keep letters, numbers and apostrophes
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # remove stopwords and 1-char tokens
    tokens = [t for t in text.split() if t not in ENGLISH_STOP_WORDS and len(t) > 1]
    return " ".join(tokens)

if __name__ == '__main__':
    samples = [
        "Hey, I hate you!!! Visit http://example.com",
        "You're great :)",
    ]
    for s in samples:
        print(s, "->", normalize_text(s))
