from __future__ import annotations

import json
from functools import lru_cache

from Levenshtein import distance as levenshtein_distance

from core.config import SKILL_TAXONOMY_PATH, Thresholds


@lru_cache(maxsize=1)
def _load_alias_map() -> dict[str, str]:
    """alias_lower → canonical name"""
    with open(SKILL_TAXONOMY_PATH, encoding="utf-8") as f:
        taxonomy = json.load(f)
    alias_map: dict[str, str] = {}
    for skills in taxonomy["categories"].values():
        for entry in skills:
            canonical = entry["name"]
            alias_map[canonical.lower()] = canonical
            for alias in entry.get("aliases", []):
                alias_map[alias.lower()] = canonical
    return alias_map


def _normalize_via_taxonomy(skill: str, alias_map: dict[str, str]) -> str:
    return alias_map.get(skill.lower(), skill)


def deduplicate(skills: list[str]) -> list[str]:
    """
    Remove duplicate skills using:
    1. Taxonomy alias resolution (Python3 → Python)
    2. Edit-distance deduplication for near-identical strings
    """
    alias_map = _load_alias_map()

    # Step 1: resolve to canonical names
    canonical: list[str] = [_normalize_via_taxonomy(s, alias_map) for s in skills]

    # Step 2: deduplicate by exact match (case-insensitive)
    seen: dict[str, str] = {}
    for skill in canonical:
        key = skill.lower()
        if key not in seen:
            seen[key] = skill
    unique = list(seen.values())

    # Step 3: edit-distance dedup (handles "Python Programming" vs "Python")
    final: list[str] = []
    for skill in unique:
        is_dup = False
        for accepted in final:
            ed = levenshtein_distance(skill.lower(), accepted.lower())
            max_len = max(len(skill), len(accepted))
            # Relative edit distance < 20% of the longer string → duplicate
            if ed <= Thresholds.DUPLICATE_EDIT_DISTANCE or ed / max_len < 0.2:
                is_dup = True
                break
        if not is_dup:
            final.append(skill)

    return final
