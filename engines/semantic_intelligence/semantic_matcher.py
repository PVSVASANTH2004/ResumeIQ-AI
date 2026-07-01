from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import SENTENCE_MODEL


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    return SentenceTransformer(SENTENCE_MODEL)


class SemanticMatcher:
    def __init__(self):
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = _load_model()
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def score_resume_vs_jd(self, resume_text: str, jd_text: str) -> float:
        """Cosine similarity between resume and JD embeddings."""
        vecs = self._embed([resume_text[:4000], jd_text[:4000]])
        return float(np.dot(vecs[0], vecs[1]))

    def match_skills(
        self,
        resume_skills: list[str],
        jd_skills: list[str],
        threshold: float = 0.75,
    ) -> dict[str, str]:
        """
        For each JD skill, find the closest resume skill above threshold.
        Returns {jd_skill: resume_skill} for matched pairs.
        """
        if not resume_skills or not jd_skills:
            return {}

        r_vecs = self._embed(resume_skills)
        j_vecs = self._embed(jd_skills)

        matched: dict[str, str] = {}
        sim_matrix = j_vecs @ r_vecs.T  # (n_jd, n_resume)

        for j_idx, jd_skill in enumerate(jd_skills):
            best_r_idx = int(np.argmax(sim_matrix[j_idx]))
            best_score = float(sim_matrix[j_idx, best_r_idx])
            if best_score >= threshold:
                matched[jd_skill] = resume_skills[best_r_idx]

        return matched

    def embed_skill(self, skill: str) -> np.ndarray:
        return self._embed([skill])[0]

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
