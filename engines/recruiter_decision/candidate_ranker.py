from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from core.config import ModelPaths


class CandidateRanker:
    """LightGBM LambdaRank model for bulk candidate ranking."""

    def __init__(self):
        self._model = None
        self._load_model()

    def _load_model(self):
        path = Path(ModelPaths.CANDIDATE_RANKER)
        if path.exists():
            with open(path, "rb") as f:
                self._model = pickle.load(f)

    def rank(self, feature_vectors: list[np.ndarray]) -> list[int]:
        """
        Return indices sorted from best to worst candidate.
        Falls back to ranking by weighted score sum.
        """
        if not feature_vectors:
            return []

        X = np.vstack(feature_vectors)

        if self._model is not None:
            scores = self._model.predict(X)
        else:
            # Fallback: weighted sum of positive features
            weights = np.array([
                0.30, 0.20, 0.15, 0.10, 0.10,  # positive
                -0.10, -0.05,                    # negative (hallucination, stuffing)
                0.03, 0.03, 0.03, -0.03, 0.01   # breadth, exp, projects, missing, tenure
            ], dtype=np.float32)
            scores = X @ weights

        # Higher score = better rank = lower rank index
        ranked_indices = list(np.argsort(scores)[::-1])
        return ranked_indices
