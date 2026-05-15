from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
import nltk
from nltk.tokenize import word_tokenize

POS_BUCKETS = ["N", "V", "J", "R", "P", "M", "D", "W", "O"]  # Noun, Verb, Adj, Adv, Pron, Num, Det, Wh, Other

def bucket_pos(tag: str) -> str:
    if tag.startswith("N"):
        return "N"
    if tag.startswith("V"):
        return "V"
    if tag.startswith("J"):
        return "J"
    if tag.startswith("R"):
        return "R"
    if tag.startswith(("PRP", "WP")):
        return "P"
    if tag.startswith(("CD",)):
        return "M"
    if tag.startswith(("DT", "PDT")):
        return "D"
    if tag.startswith(("W",)):
        return "W"
    return "O"


class PosTagCountsTransformer:
    """Convert text to POS tag distribution (normalized counts) as sparse features."""

    def __init__(self):
        self.buckets = POS_BUCKETS
        self.bucket_index = {b: i for i, b in enumerate(self.buckets)}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            tokens = word_tokenize(text)
            tags = nltk.pos_tag(tokens)
            counts = Counter(bucket_pos(tag) for _, tag in tags)
            total = sum(counts.values())
            vec = np.zeros(len(self.buckets), dtype=float)
            if total > 0:
                for bucket, count in counts.items():
                    idx = self.bucket_index.get(bucket, None)
                    if idx is not None:
                        vec[idx] = count / total
            rows.append(vec)
        data = np.vstack(rows) if rows else np.zeros((0, len(self.buckets)))
        return csr_matrix(data)
