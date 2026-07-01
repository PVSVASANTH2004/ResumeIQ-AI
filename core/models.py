from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ─── Enumerations ────────────────────────────────────────────────────────────

class ComplexityLevel(str, Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    PRODUCTION = "Production"


class HiringRecommendation(str, Enum):
    REJECT = "Reject"
    MAYBE = "Maybe"
    INTERVIEW = "Interview"
    STRONG_HIRE = "Strong Hire"


class GrowthTrend(str, Enum):
    ACCELERATING = "accelerating"
    STEADY = "steady"
    STAGNANT = "stagnant"


class SkillImportance(str, Enum):
    REQUIRED = "required"
    PREFERRED = "preferred"


# ─── Resume Building Blocks ───────────────────────────────────────────────────

class ExperienceEntry(BaseModel):
    company: str = ""
    role: str = ""
    duration_months: int = 0
    bullets: list[str] = Field(default_factory=list)
    start_year: Optional[int] = None
    end_year: Optional[int] = None


class ProjectEntry(BaseModel):
    title: str = ""
    description: str = ""
    technologies: list[str] = Field(default_factory=list)


class EducationEntry(BaseModel):
    institution: str = ""
    degree: str = ""
    field: str = ""
    graduation_year: Optional[int] = None
    gpa: Optional[float] = None


class ExtractedSkills(BaseModel):
    programming: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    cloud: list[str] = Field(default_factory=list)
    databases: list[str] = Field(default_factory=list)
    ai_ml: list[str] = Field(default_factory=list)
    devops: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)

    @property
    def all_skills(self) -> list[str]:
        return (
            self.programming + self.frameworks + self.cloud
            + self.databases + self.ai_ml + self.devops + self.soft_skills
        )


# ─── Parser Outputs ───────────────────────────────────────────────────────────

class ParsedResume(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    raw_text: str = ""
    sections: dict[str, str] = Field(default_factory=dict)
    skills_raw: list[str] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)


class ParsedJD(BaseModel):
    title: str = ""
    raw_text: str = ""
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    min_experience_years: int = 0
    responsibilities: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)


# ─── Scoring Components ───────────────────────────────────────────────────────

class MissingSkill(BaseModel):
    skill: str
    importance: SkillImportance
    current_match_pct: float = 0.0
    estimated_match_pct: float = 0.0


class LearningProgression(BaseModel):
    timeline: list[dict] = Field(default_factory=list)
    velocity: float = 0.0
    growth_trend: GrowthTrend = GrowthTrend.STEADY


class ProjectScore(BaseModel):
    title: str
    complexity: ComplexityLevel = ComplexityLevel.BEGINNER
    novelty_score: float = 0.0


class ScoreBreakdown(BaseModel):
    semantic_match: float = 0.0
    education_score: float = 0.0
    evidence_score: float = 0.0
    experience_quality: float = 0.0
    project_complexity: float = 0.0
    skill_confidence: float = 0.0
    ats_compatibility: float = 0.0
    readability: float = 0.0
    learning_progression: float = 0.0
    active_weights: dict[str, float] = Field(default_factory=dict)
    final_score: float = 0.0


# ─── Engine Outputs ───────────────────────────────────────────────────────────

class Engine1Output(BaseModel):
    parsed_resume: ParsedResume
    parsed_jd: ParsedJD
    extracted_skills: ExtractedSkills = Field(default_factory=ExtractedSkills)
    predicted_role: str = ""
    ats_issues: list[str] = Field(default_factory=list)
    stuffing_risk: float = 0.0
    keyword_stuffing_detected: bool = False


class Engine2Output(BaseModel):
    semantic_score: float = 0.0
    matched_skills: dict[str, str] = Field(default_factory=dict)
    missing_skills: list[MissingSkill] = Field(default_factory=list)
    role_fit_scores: dict[str, float] = Field(default_factory=dict)


class Engine3Output(BaseModel):
    evidence_scores: dict[str, float] = Field(default_factory=dict)
    skill_confidence: dict[str, float] = Field(default_factory=dict)
    hallucinated_skills: list[str] = Field(default_factory=list)


class Engine4Output(BaseModel):
    project_scores: list[ProjectScore] = Field(default_factory=list)
    experience_quality_score: float = 0.0
    learning_progression: LearningProgression = Field(default_factory=LearningProgression)
    avg_project_complexity_score: float = 0.0


class Engine5Output(BaseModel):
    ats_score: float = 0.0
    hiring_recommendation: HiringRecommendation = HiringRecommendation.MAYBE
    interview_probability: float = 0.0
    shortlist_probability: float = 0.0
    candidate_rank: Optional[int] = None


class Engine6Output(BaseModel):
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    shap_values: dict[str, float] = Field(default_factory=dict)
    improvement_suggestions: list[str] = Field(default_factory=list)
    explanation_text: str = ""
    # OpenAI-powered outputs (empty when API key not set)
    recruiter_summary: str = ""
    candidate_strengths: list[str] = Field(default_factory=list)
    candidate_weaknesses: list[str] = Field(default_factory=list)
    interview_questions: list[str] = Field(default_factory=list)
    ai_review: str = ""


# ─── Final Result ─────────────────────────────────────────────────────────────

class AnalysisResult(BaseModel):
    resume_intelligence: Engine1Output
    semantic_intelligence: Engine2Output
    candidate_verification: Engine3Output
    candidate_quality: Engine4Output
    recruiter_decision: Engine5Output
    explainability: Engine6Output
    created_at: datetime = Field(default_factory=datetime.now)


# ─── Bulk Mode ────────────────────────────────────────────────────────────────

class BulkCandidate(BaseModel):
    filename: str
    result: AnalysisResult
    rank: int = 0


class BulkRankingResult(BaseModel):
    jd: ParsedJD
    candidates: list[BulkCandidate]
    total: int = 0
