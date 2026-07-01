from collections import defaultdict
from datetime import datetime

from core.models import GrowthTrend, LearningProgression
from knowledge_graph.resume_graph import ResumeGraph


def compute_learning_progression(graph: ResumeGraph) -> LearningProgression:
    """
    Build a learning timeline from the knowledge graph (skill → experience year),
    compute skills-per-year velocity, and classify the growth trend.
    Falls back to a topology-based estimate when dates cannot be parsed.
    """
    skill_years = graph.skills_by_year()  # [(year, skill), ...]

    if not skill_years:
        # No dated experience entries — estimate from graph topology.
        # Assume a 2-year active development period as a conservative baseline.
        all_skills = graph.all_skills()
        all_projects = graph.all_projects()
        if not all_skills:
            return LearningProgression(timeline=[], velocity=0.0, growth_trend=GrowthTrend.STAGNANT)
        estimated_velocity = round(min(len(all_skills) / 2.0, 10.0), 2)
        trend = GrowthTrend.STEADY if len(all_projects) >= 2 else GrowthTrend.STAGNANT
        return LearningProgression(timeline=[], velocity=estimated_velocity, growth_trend=trend)

    # Group skills by year
    by_year: dict[int, list[str]] = defaultdict(list)
    for year, skill in skill_years:
        by_year[year].append(skill)

    timeline = [
        {"year": year, "skills": skills, "count": len(skills)}
        for year, skills in sorted(by_year.items())
    ]

    # Velocity = average new skills per year
    years = sorted(by_year.keys())
    total_skills = sum(len(by_year[y]) for y in years)
    year_span = max(years[-1] - years[0], 1)
    velocity = round(total_skills / year_span, 2)

    # Growth trend: compare first half vs second half
    mid = len(years) // 2
    first_half_avg = sum(len(by_year[y]) for y in years[:mid]) / max(mid, 1)
    second_half_avg = sum(len(by_year[y]) for y in years[mid:]) / max(len(years) - mid, 1)

    if second_half_avg > first_half_avg * 1.2:
        trend = GrowthTrend.ACCELERATING
    elif second_half_avg < first_half_avg * 0.8:
        trend = GrowthTrend.STAGNANT
    else:
        trend = GrowthTrend.STEADY

    return LearningProgression(timeline=timeline, velocity=velocity, growth_trend=trend)


def progression_to_score(progression: LearningProgression) -> float:
    """Convert learning progression to a 0–1 score for the final weighted scoring."""
    velocity_score = min(progression.velocity / 5.0, 1.0)  # 5 skills/year = 1.0

    trend_bonus = {
        GrowthTrend.ACCELERATING: 0.20,
        GrowthTrend.STEADY:       0.10,
        GrowthTrend.STAGNANT:     0.00,
    }

    return round(min(velocity_score + trend_bonus[progression.growth_trend], 1.0), 4)
