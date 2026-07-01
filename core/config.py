import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent


class ScoreWeights:
    # Recruiter-centric weights: required skill match and experience
    # dominate; readability and learning are supporting signals only.
    SEMANTIC_MATCH = 0.35      # required skill coverage vs JD
    EXPERIENCE_QUALITY = 0.20  # quality of internship / work bullets
    PROJECT_COMPLEXITY = 0.15  # depth and novelty of projects
    EVIDENCE_SCORE = 0.10      # skills backed by projects / experience
    ATS_COMPATIBILITY = 0.10   # parse-ability for ATS systems
    READABILITY = 0.05         # resume structure and formatting
    LEARNING_PROGRESSION = 0.05  # skill growth over time
    # SKILL_CONFIDENCE is computed and displayed but not in final score


class ModelPaths:
    RESUME_CLASSIFIER = BASE_DIR / "data" / "models" / "resume_classifier"
    PROJECT_COMPLEXITY = BASE_DIR / "data" / "models" / "project_complexity.pkl"
    HIRING_RECOMMENDER = BASE_DIR / "data" / "models" / "hiring_recommender.pkl"
    CANDIDATE_RANKER = BASE_DIR / "data" / "models" / "candidate_ranker.pkl"
    HALLUCINATION_DETECTOR = BASE_DIR / "data" / "models" / "hallucination_detector.pkl"


class Thresholds:
    SEMANTIC_MATCH_MIN = 0.30
    EVIDENCE_REQUIRED = 0.50
    HALLUCINATION_ANOMALY = 0.15
    KEYWORD_STUFFING_RISK = 0.70
    SKILL_CONFIDENCE_LOW = 0.40
    DUPLICATE_EDIT_DISTANCE = 2
    DUPLICATE_COSINE_MIN = 0.92


OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'resumeiq.db'}"
SENTENCE_MODEL: str = "all-MiniLM-L6-v2"
SKILL_TAXONOMY_PATH = BASE_DIR / "data" / "skill_taxonomy.json"
