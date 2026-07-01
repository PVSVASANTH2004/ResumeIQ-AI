from __future__ import annotations

import json

from core.config import OPENAI_API_KEY
from core.models import AnalysisResult

_SYSTEM_PROMPT = """You are an expert technical recruiter and career coach with 15 years of experience evaluating software engineering candidates.

You will receive a detailed, deterministic resume analysis (scores computed by ML models) plus the original resume text and job description. Your job is to ADD HUMAN-READABLE INTELLIGENCE on top of those scores.

Return a JSON object with EXACTLY these keys:

"suggestions"        – array of 5-7 specific, actionable resume improvement suggestions written as full natural-language sentences (not just "Add Docker"). Explain WHY the change matters in context of the JD.
"strengths"          – array of 3-5 short bullet strings describing genuine candidate strengths.
"weaknesses"         – array of 2-4 short bullet strings describing gaps relative to this JD.
"interview_questions" – array of 5 targeted interview questions based on the resume AND JD (mix of technical and behavioral).
"recruiter_summary"  – exactly 2-3 sentences summarizing the candidate for a hiring manager.
"ai_review"          – 1-2 paragraph comprehensive recruiter assessment; mention fit, standout qualities, key gaps, and a recommendation.

Critical rules for suggestions:
- Treat as interchangeable (never suggest alternatives if one is present): React/Angular/Vue/Svelte · FastAPI/Django/Flask/Express · AWS/Azure/GCP · PostgreSQL/MySQL/MariaDB · Docker+Kubernetes
- Only suggest a skill if it is genuinely absent from the resume AND not covered by an equivalent.
- Be specific: reference actual project names, company names, and JD requirements by name.
- Avoid generic advice like "improve your resume" or "add more details"."""


def _build_prompt(result: AnalysisResult) -> str:
    sb   = result.explainability.score_breakdown
    e1   = result.resume_intelligence
    e2   = result.semantic_intelligence
    e3   = result.candidate_verification
    e4   = result.candidate_quality

    missing     = [ms.skill for ms in e2.missing_skills[:6]]
    hallucinated = e3.hallucinated_skills[:3]
    rec         = result.recruiter_decision.hiring_recommendation.value
    projects    = [ps.title for ps in e4.project_scores]
    skills      = e1.extracted_skills.all_skills
    edu         = e1.parsed_resume.education
    edu_str     = "; ".join(
        f"{e.degree} in {e.field} ({e.institution}, GPA {e.gpa})" for e in edu
    ) if edu else "Not found"

    resume_snippet = e1.parsed_resume.raw_text[:2000]
    jd_snippet     = e1.parsed_jd.raw_text[:1000]

    return f"""=== DETERMINISTIC SCORES (computed by ML models — do NOT re-score) ===
Final Score:          {sb.final_score}/100
Hiring Recommendation:{rec}
Semantic Match:       {sb.semantic_match * 100:.1f}%
Education Match:      {sb.education_score * 100:.1f}%
Evidence Score:       {sb.evidence_score * 100:.1f}%
Experience Quality:   {sb.experience_quality * 100:.1f}%
Project Complexity:   {sb.project_complexity * 100:.1f}%
ATS Compatibility:    {sb.ats_compatibility * 100:.1f}%

=== RESUME CONTEXT ===
Education:            {edu_str}
Skills:               {', '.join(skills[:30])}
Projects:             {', '.join(projects)}
Missing JD Skills:    {', '.join(missing) if missing else 'None'}
Unsupported Claims:   {', '.join(hallucinated) if hallucinated else 'None'}
ATS Issues:           {', '.join(e1.ats_issues[:3]) if e1.ats_issues else 'None'}

=== RESUME TEXT (first 2000 chars) ===
{resume_snippet}

=== JOB DESCRIPTION (first 1000 chars) ===
{jd_snippet}

Now generate the JSON response with all 6 required keys."""


def generate_ai_content(result: AnalysisResult) -> dict:
    """
    Single OpenAI call that returns all language-intelligence outputs:
    suggestions, strengths, weaknesses, interview_questions,
    recruiter_summary, ai_review.

    Falls back to rule-based content when the API key is absent.
    """
    if not OPENAI_API_KEY:
        return _fallback_content(result)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _build_prompt(result)},
            ],
            temperature=0.4,
            max_tokens=1400,
            response_format={"type": "json_object"},
        )
        raw    = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        return {
            "suggestions":         _extract_list(parsed, "suggestions"),
            "strengths":           _extract_list(parsed, "strengths"),
            "weaknesses":          _extract_list(parsed, "weaknesses"),
            "interview_questions": _extract_list(parsed, "interview_questions"),
            "recruiter_summary":   parsed.get("recruiter_summary", ""),
            "ai_review":           parsed.get("ai_review", ""),
        }
    except Exception as e:
        fb = _fallback_content(result)
        fb["suggestions"].append(f"(OpenAI unavailable: {str(e)[:80]})")
        return fb


def _extract_list(parsed: dict, key: str) -> list[str]:
    val = parsed.get(key, [])
    return val if isinstance(val, list) else []


# ── Fallback (no API key) ─────────────────────────────────────────────────────

_TECH_GROUPS: list[set[str]] = [
    {"react", "angular", "vue", "svelte"},
    {"django", "spring boot", "express", "fastapi", "flask"},
    {"aws", "azure", "google cloud", "gcp"},
    {"postgresql", "mysql", "mariadb"},
    {"docker", "kubernetes"},
]


def _redundant_with_resume(skill: str, resume_skills: set[str]) -> bool:
    skill_l = skill.lower()
    for group in _TECH_GROUPS:
        if skill_l in group and any(r.lower() in group for r in resume_skills):
            return True
    return False


def _fallback_content(result: AnalysisResult) -> dict:
    sb           = result.explainability.score_breakdown
    resume_skills = set(result.resume_intelligence.extracted_skills.all_skills)
    missing       = [
        ms for ms in result.semantic_intelligence.missing_skills
        if not _redundant_with_resume(ms.skill, resume_skills)
    ][:3]
    rec = result.recruiter_decision.hiring_recommendation.value

    suggestions: list[str] = []
    for ms in missing:
        suggestions.append(
            f"Add '{ms.skill}' — it's a {ms.importance.value} skill for this role "
            f"(currently {ms.current_match_pct:.0f}% match)."
        )
    if result.resume_intelligence.keyword_stuffing_detected:
        suggestions.append(
            "Reduce skill repetition. Move skills into project/experience bullet points as evidence."
        )
    if result.candidate_verification.hallucinated_skills:
        skills = ", ".join(result.candidate_verification.hallucinated_skills[:2])
        suggestions.append(f"Add concrete evidence for: {skills}. Reference a project or work task.")
    if sb.experience_quality < 0.4:
        suggestions.append(
            "Strengthen experience bullets with numbers: 'reduced latency by 40%' "
            "instead of 'improved backend performance'."
        )
    if result.resume_intelligence.ats_issues:
        suggestions.append(f"Fix ATS issue: {result.resume_intelligence.ats_issues[0]}")

    strengths: list[str] = []
    if sb.project_complexity > 0.6:
        strengths.append("Strong project portfolio with advanced complexity")
    if sb.experience_quality > 0.5:
        strengths.append("Solid professional experience with quality bullet points")
    if sb.semantic_match > 0.6:
        strengths.append("Good skill alignment with job requirements")
    if result.resume_intelligence.parsed_resume.education:
        strengths.append("Relevant educational background")
    if not strengths:
        strengths.append("Resume is well structured and ATS-compatible")

    weaknesses: list[str] = []
    if missing:
        weaknesses.append(f"Missing required skills: {', '.join(m.skill for m in missing[:2])}")
    if sb.evidence_score < 0.4:
        weaknesses.append("Skills lack supporting evidence in projects/experience")
    if sb.experience_quality < 0.4:
        weaknesses.append("Experience bullets could be stronger and more quantified")

    interview_questions = [
        "Walk me through your most complex project and the technical decisions you made.",
        "How have you handled a situation where a technical approach wasn't working?",
        "Describe your experience integrating external APIs or third-party services.",
        "How do you approach debugging a performance issue in production?",
        "What's your process for picking up a new technology quickly?",
    ]

    recruiter_summary = (
        f"Candidate shows {rec.lower()} potential for this role with a score of "
        f"{sb.final_score:.0f}/100. "
        f"Semantic skill match is {sb.semantic_match*100:.0f}%. "
        f"{'Strong project portfolio.' if sb.project_complexity > 0.6 else 'Projects could demonstrate more complexity.'}"
    )

    ai_review = (
        f"This candidate scored {sb.final_score:.0f}/100 against the provided job description. "
        f"Skill coverage is {sb.semantic_match*100:.0f}% with evidence confidence of "
        f"{sb.evidence_score*100:.0f}%. "
        f"{'The project portfolio demonstrates advanced work.' if sb.project_complexity > 0.6 else 'Project complexity is moderate.'} "
        f"Recommendation: {rec}. "
        f"(Add your OpenAI API key to .env for a detailed AI-generated review.)"
    )

    return {
        "suggestions":         suggestions,
        "strengths":           strengths,
        "weaknesses":          weaknesses,
        "interview_questions": interview_questions,
        "recruiter_summary":   recruiter_summary,
        "ai_review":           ai_review,
    }


# Keep this for backwards compatibility
def generate_suggestions(result: AnalysisResult) -> list[str]:
    return generate_ai_content(result).get("suggestions", [])
