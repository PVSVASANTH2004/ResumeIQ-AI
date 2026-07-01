import json
import re
from pathlib import Path

from core.config import SKILL_TAXONOMY_PATH
from core.models import ParsedJD

# Load all known skill names for matching
with open(SKILL_TAXONOMY_PATH, encoding="utf-8") as _f:
    _TAX = json.load(_f)

_ALL_SKILL_NAMES: list[str] = []
for _cat_skills in _TAX["categories"].values():
    for _entry in _cat_skills:
        _ALL_SKILL_NAMES.append(_entry["name"])
        _ALL_SKILL_NAMES.extend(_entry["aliases"])

_SKILL_RE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in sorted(_ALL_SKILL_NAMES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_REQUIRED_MARKERS = re.compile(
    r"\b(required|must have|must-have|mandatory|essential|you must|minimum)\b", re.I
)
_PREFERRED_MARKERS = re.compile(
    r"\b(preferred|nice to have|bonus|plus|advantageous|desired|ideally)\b", re.I
)
_YEAR_EXP_RE = re.compile(r"(\d+)\+?\s*years?\s*(?:of\s+)?(?:experience|exp)", re.I)
_RESPONSIBILITY_BULLETS = re.compile(r"(?:^|\n)\s*[•\-*–]\s*(.+)", re.M)

_TITLE_RE = re.compile(
    r"(?:job title|position|role)[:\s]+([^\n]+)", re.I
)


def _extract_skills_from_text(text: str) -> list[str]:
    found = _SKILL_RE.findall(text)
    seen: dict[str, str] = {}
    for skill in found:
        key = skill.lower()
        if key not in seen:
            seen[key] = skill
    return list(seen.values())


def _split_required_preferred(text: str) -> tuple[list[str], list[str]]:
    """
    Split JD text into required vs preferred blocks, then extract skills.

    When a JD has no explicit Required / Preferred section headers (common
    for informal postings), treat all detected skills as preferred and promote
    only those mentioned early in the JD (first 40% of text) to required.
    This prevents the entire skill list from being marked required, which
    would unfairly deflate skill-coverage scores.
    """
    sections = re.split(r"\n(?=.{0,40}(?:required|preferred|nice to have)[:\s])", text, flags=re.I)

    has_explicit_markers = any(
        _REQUIRED_MARKERS.search(s[:80]) or _PREFERRED_MARKERS.search(s[:80])
        for s in sections
    )

    required_skills: list[str] = []
    preferred_skills: list[str] = []

    if has_explicit_markers:
        for section in sections:
            skills = _extract_skills_from_text(section)
            if _REQUIRED_MARKERS.search(section[:80]):
                required_skills.extend(skills)
            elif _PREFERRED_MARKERS.search(section[:80]):
                preferred_skills.extend(skills)
            else:
                preferred_skills.extend(skills)
    else:
        # No explicit markers: skills in the first 40% of the JD → required,
        # the rest → preferred. Caps required at 8 to stay realistic.
        cutoff = int(len(text) * 0.40)
        early_skills = _extract_skills_from_text(text[:cutoff])
        all_skills = _extract_skills_from_text(text)
        required_skills = early_skills[:8]
        preferred_skills = [s for s in all_skills if s not in required_skills]

    req_set = list(dict.fromkeys(required_skills))
    pref_set = [s for s in dict.fromkeys(preferred_skills) if s not in req_set]
    return req_set, pref_set


def _extract_responsibilities(text: str) -> list[str]:
    bullets = _RESPONSIBILITY_BULLETS.findall(text)
    return [b.strip() for b in bullets if len(b.strip()) > 10][:20]


def _extract_title(text: str) -> str:
    match = _TITLE_RE.search(text)
    if match:
        return match.group(1).strip()
    # Fallback: first non-empty line
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and len(stripped) < 80:
            return stripped
    return ""


class JDParser:
    def parse(self, jd_text: str) -> ParsedJD:
        title = _extract_title(jd_text)
        required_skills, preferred_skills = _split_required_preferred(jd_text)
        responsibilities = _extract_responsibilities(jd_text)

        year_match = _YEAR_EXP_RE.search(jd_text)
        min_experience_years = int(year_match.group(1)) if year_match else 0

        # Technologies = union of required + preferred (all detected skills)
        all_skills = list(dict.fromkeys(required_skills + preferred_skills))

        return ParsedJD(
            title=title,
            raw_text=jd_text,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            min_experience_years=min_experience_years,
            responsibilities=responsibilities,
            technologies=all_skills,
        )
