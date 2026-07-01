from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from core.models import ProjectEntry
from engines.semantic_intelligence.semantic_matcher import SemanticMatcher

# Corpus of common/generic student projects — low novelty anchors
_COMMON_PROJECTS = [
    "todo app CRUD application simple web",
    "calculator basic arithmetic operations",
    "weather app API integration simple dashboard",
    "e-commerce basic shopping cart product listing",
    "chat application basic messaging websocket",
    "portfolio website personal HTML CSS JavaScript",
    "login registration authentication system",
    "blog CRUD post comment like",
    "student management system database",
    "library management system CRUD",
]


class ProjectNoveltyScorer:
    def __init__(self):
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=500, stop_words="english")
        self._tfidf.fit(_COMMON_PROJECTS)
        self._matcher = SemanticMatcher()
        self._common_vecs = np.array([
            self._matcher.embed_skill(p) for p in _COMMON_PROJECTS
        ])

    def score(self, projects: list[ProjectEntry]) -> list[float]:
        """
        Return novelty score 0–1 for each project.
        Score = 1 - similarity_to_most_common_project.
        """
        scores: list[float] = []
        for proj in projects:
            text = f"{proj.title} {proj.description}"
            if not text.strip():
                scores.append(0.5)
                continue

            proj_vec = self._matcher.embed_skill(text[:200])
            sims = np.array([self._matcher.cosine(proj_vec, cv) for cv in self._common_vecs])
            max_sim = float(sims.max())
            novelty = round(1.0 - max_sim, 4)
            scores.append(max(0.0, novelty))

        return scores
