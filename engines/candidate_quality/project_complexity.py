from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from core.config import ModelPaths
from core.models import ComplexityLevel, ProjectEntry, ProjectScore


_COMPLEXITY_LABELS = [
    ComplexityLevel.BEGINNER,
    ComplexityLevel.INTERMEDIATE,
    ComplexityLevel.ADVANCED,
    ComplexityLevel.PRODUCTION,
]

# Keyword tiers — must match training script exactly
_TIER_KW = {
    3: ['production', 'deployed', 'million users', 'billion', 'scalable', 'distributed',
        'microservices', 'kubernetes', 'ci/cd', 'real-time', 'sla', 'team of',
        'led', 'architected', 'reduced', 'percent'],
    2: ['api', 'database', 'authentication', 'docker', 'cloud', 'aws', 'gcp', 'azure',
        'machine learning', 'deep learning', 'neural network', 'nlp', 'redis', 'kafka',
        'fine-tuned', 'bert', 'pipeline'],
    1: ['crud', 'rest', 'web app', 'mobile app', 'backend', 'frontend', 'react',
        'django', 'flask', 'sql', 'javascript', 'jwt', 'dashboard'],
    0: ['calculator', 'todo', 'hello world', 'game', 'basic', 'simple',
        'tkinter', 'pygame', 'html css'],
}

# Kept for heuristic fallback
_TIER_KEYWORDS = {
    ComplexityLevel.PRODUCTION: _TIER_KW[3],
    ComplexityLevel.ADVANCED:   _TIER_KW[2],
    ComplexityLevel.INTERMEDIATE: _TIER_KW[1],
    ComplexityLevel.BEGINNER:   _TIER_KW[0],
}


def _handcrafted_features(text: str) -> list[float]:
    """11 features — must match training/train_models_2_3.py hand_feats() exactly."""
    t = text.lower()
    hits = [sum(1 for kw in _TIER_KW[i] if kw in t) for i in [0, 1, 2, 3]]
    return [
        len(t.split()) / 100.0,
        hits[0] / 5.0,
        hits[1] / 8.0,
        hits[2] / 10.0,
        hits[3] / 12.0,
        float(bool(re.search(r"\d+%|\d+x|million|billion|\d+k users", t))),
        float(bool(re.search(r"deploy|production|cloud|k8s|docker", t))),
        float(bool(re.search(r"\d+[km]|million|billion|thousand", t))),
        float(bool(re.search(r"reduc|increas|improv|optim|achiev", t))),
        float(bool(re.search(r"team|led|manag|collaborat", t))),
        sum(1 for kws in _TIER_KW.values() for kw in kws if kw in t) / 20.0,
    ]


def _heuristic_complexity(text: str) -> ComplexityLevel:
    text_lower = text.lower()
    for level in [ComplexityLevel.PRODUCTION, ComplexityLevel.ADVANCED,
                  ComplexityLevel.INTERMEDIATE]:
        hits = sum(1 for kw in _TIER_KEYWORDS[level] if kw in text_lower)
        if hits >= 2:
            return level
    return ComplexityLevel.BEGINNER


class ProjectComplexityModel:
    def __init__(self):
        self._clf: RandomForestClassifier | None = None
        self._tfidf: TfidfVectorizer | None = None
        self._load_model()

    def _load_model(self):
        path = Path(ModelPaths.PROJECT_COMPLEXITY)
        if path.exists():
            with open(path, "rb") as f:
                bundle = pickle.load(f)
                self._clf = bundle["clf"]
                self._tfidf = bundle["tfidf"]

    def predict(self, projects: list[ProjectEntry]) -> list[ProjectScore]:
        results: list[ProjectScore] = []
        for proj in projects:
            text = f"{proj.title} {proj.description} {' '.join(proj.technologies)}"
            if self._clf and self._tfidf:
                tfidf_vec = self._tfidf.transform([text]).toarray()
                hand_vec = np.array([_handcrafted_features(text)])
                X = np.hstack([tfidf_vec, hand_vec])
                label_idx = int(self._clf.predict(X)[0])
                complexity = _COMPLEXITY_LABELS[label_idx]
            else:
                complexity = _heuristic_complexity(text)

            results.append(ProjectScore(title=proj.title, complexity=complexity))
        return results

    def complexity_to_score(self, complexity: ComplexityLevel) -> float:
        mapping = {
            ComplexityLevel.BEGINNER:     0.25,
            ComplexityLevel.INTERMEDIATE: 0.50,
            ComplexityLevel.ADVANCED:     0.75,
            ComplexityLevel.PRODUCTION:   1.00,
        }
        return mapping[complexity]
