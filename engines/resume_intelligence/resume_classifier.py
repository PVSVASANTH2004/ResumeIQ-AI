import os
import pickle
from pathlib import Path

from core.config import ModelPaths
from core.models import ParsedResume

_ROLE_LABELS = [
    "Software Engineer", "Backend Engineer", "Frontend Engineer",
    "Full Stack Engineer", "ML Engineer", "Data Scientist",
    "Data Engineer", "DevOps Engineer", "Cloud Engineer",
    "AI Engineer", "Mobile Developer", "Android Developer",
    "iOS Developer", "Embedded Systems", "Security Engineer",
    "QA Engineer", "Product Manager", "Business Analyst",
    "Database Administrator", "Network Engineer",
    "System Administrator", "Technical Writer",
    "Blockchain Developer", "Game Developer", "Research Scientist",
]


class ResumeClassifier:
    """
    DistilBERT-based resume role classifier.
    Falls back to a keyword heuristic when the trained model is not present,
    so the rest of the pipeline remains functional during development.
    """

    def __init__(self):
        self._pipeline = None
        self._fallback_mode = False
        self._try_load_model()

    def _try_load_model(self):
        model_path = ModelPaths.RESUME_CLASSIFIER
        if not Path(model_path).exists():
            self._fallback_mode = True
            return

        try:
            from transformers import pipeline as hf_pipeline
            self._pipeline = hf_pipeline(
                "text-classification",
                model=str(model_path),
                top_k=1,
            )
        except Exception:
            self._fallback_mode = True

    def _heuristic_classify(self, text: str) -> str:
        text_lower = text.lower()
        scores: dict[str, int] = {}

        keyword_map = {
            "AI Engineer":        ["langchain", "huggingface", "rag", "llm", "embeddings", "vector", "generative", "openai", "gpt", "prompt"],
            "ML Engineer":        ["pytorch", "tensorflow", "scikit", "model training", "neural network", "deep learning", "snn", "gan", "transformer"],
            "Data Scientist":     ["data science", "statistics", "hypothesis", "jupyter", "matplotlib", "seaborn", "pandas", "numpy"],
            "Data Engineer":      ["etl", "pipeline", "spark", "airflow", "kafka", "data warehouse"],
            "DevOps Engineer":    ["devops", "kubernetes", "docker", "terraform", "ci/cd", "jenkins"],
            "Backend Engineer":   ["fastapi", "django", "spring boot", "node.js", "microservices", "rest api", "graphql"],
            "Frontend Engineer":  ["react", "angular", "vue", "redux", "next.js", "tailwind", "figma"],
            "Full Stack Engineer":["full stack", "fullstack", "frontend and backend"],
            "Cloud Engineer":     ["aws", "gcp", "azure", "cloud infrastructure", "serverless"],
            "Mobile Developer":   ["flutter", "react native", "swift", "kotlin", "android", "ios"],
            "Security Engineer":  ["penetration", "vulnerability", "soc", "siem", "firewall", "cybersecurity"],
            "Research Scientist":  ["publication", "phd", "arxiv", "novel approach", "ieee", "conference paper"],
        }

        for role, keywords in keyword_map.items():
            scores[role] = sum(kw in text_lower for kw in keywords)

        if not scores or max(scores.values()) == 0:
            return "Software Engineer"

        return max(scores, key=lambda r: scores[r])

    def predict(self, resume: ParsedResume) -> str:
        text = resume.raw_text[:512]

        if self._fallback_mode or self._pipeline is None:
            return self._heuristic_classify(resume.raw_text)

        result = self._pipeline(text)
        label = result[0][0]["label"] if result else "Software Engineer"
        # Map model index labels to human-readable roles if needed
        if label.startswith("LABEL_"):
            idx = int(label.split("_")[1])
            label = _ROLE_LABELS[idx] if idx < len(_ROLE_LABELS) else "Software Engineer"
        return label
