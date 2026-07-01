import json
import re
from functools import lru_cache

import spacy
from spacy.language import Language
from spacy.pipeline import EntityRuler

from core.config import SKILL_TAXONOMY_PATH
from core.models import ExtractedSkills, ParsedResume


@lru_cache(maxsize=1)
def _load_nlp() -> Language:
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        nlp = spacy.blank("en")

    ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
    patterns = _build_patterns()
    ruler.add_patterns(patterns)
    return nlp


def _build_patterns() -> list[dict]:
    with open(SKILL_TAXONOMY_PATH, encoding="utf-8") as f:
        taxonomy = json.load(f)

    patterns = []
    for category, skills in taxonomy["categories"].items():
        for skill_entry in skills:
            canonical = skill_entry["name"]
            all_variants = [canonical] + skill_entry.get("aliases", [])
            for variant in all_variants:
                patterns.append({
                    "label": f"SKILL_{category.upper()}",
                    "pattern": variant,
                })
                # Also add lower-case version
                patterns.append({
                    "label": f"SKILL_{category.upper()}",
                    "pattern": variant.lower(),
                })
    return patterns


def _canonical_name(text: str, taxonomy: dict) -> str:
    """Return the canonical skill name for a matched text."""
    text_lower = text.lower()
    for skills in taxonomy["categories"].values():
        for entry in skills:
            if entry["name"].lower() == text_lower:
                return entry["name"]
            if text_lower in [a.lower() for a in entry.get("aliases", [])]:
                return entry["name"]
    return text.strip()


class SkillExtractor:
    def __init__(self):
        self._nlp: Language | None = None
        with open(SKILL_TAXONOMY_PATH, encoding="utf-8") as f:
            self._taxonomy = json.load(f)

    @property
    def nlp(self) -> Language:
        if self._nlp is None:
            self._nlp = _load_nlp()
        return self._nlp

    def _label_to_category(self, label: str) -> str:
        return label.replace("SKILL_", "").lower()

    def extract(self, resume: ParsedResume) -> ExtractedSkills:
        # Combine all resume text for extraction (skills section weighted)
        full_text = " ".join([
            resume.sections.get("skills", "") * 2,  # double-weight skills section
            resume.sections.get("experience", ""),
            resume.sections.get("projects", ""),
            resume.raw_text,
        ])

        doc = self.nlp(full_text)

        categorized: dict[str, set[str]] = {
            "programming": set(),
            "frameworks": set(),
            "cloud": set(),
            "databases": set(),
            "ai_ml": set(),
            "devops": set(),
            "soft_skills": set(),
        }

        for ent in doc.ents:
            if not ent.label_.startswith("SKILL_"):
                continue
            category = self._label_to_category(ent.label_)
            if category in categorized:
                canonical = _canonical_name(ent.text, self._taxonomy)
                categorized[category].add(canonical)

        # Also extract from raw skills list (already parsed)
        for raw_skill in resume.skills_raw:
            doc_s = self.nlp(raw_skill)
            for ent in doc_s.ents:
                if not ent.label_.startswith("SKILL_"):
                    continue
                category = self._label_to_category(ent.label_)
                if category in categorized:
                    canonical = _canonical_name(ent.text, self._taxonomy)
                    categorized[category].add(canonical)

        return ExtractedSkills(
            programming=sorted(categorized["programming"]),
            frameworks=sorted(categorized["frameworks"]),
            cloud=sorted(categorized["cloud"]),
            databases=sorted(categorized["databases"]),
            ai_ml=sorted(categorized["ai_ml"]),
            devops=sorted(categorized["devops"]),
            soft_skills=sorted(categorized["soft_skills"]),
        )
