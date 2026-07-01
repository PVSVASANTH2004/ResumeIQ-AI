from __future__ import annotations

import re

import networkx as nx

from core.models import ParsedResume


class ResumeGraph:
    """
    Directed graph over resume entities.

    Node types  : skill, project, experience, company, education
    Edge types  : demonstrated_in, used_in, worked_at, studied_at, required_by
    """

    def __init__(self):
        self.G: nx.DiGraph = nx.DiGraph()

    def build(self, resume: ParsedResume) -> "ResumeGraph":
        self.G.clear()

        # ── Skills ────────────────────────────────────────────────────────────
        for skill in resume.skills_raw:
            self._add_skill(skill)

        # ── Experience ────────────────────────────────────────────────────────
        for i, exp in enumerate(resume.experience):
            exp_id = f"exp_{i}_{exp.company}"
            self.G.add_node(exp_id, type="experience", role=exp.role,
                            company=exp.company, duration_months=exp.duration_months,
                            start_year=exp.start_year)
            self.G.add_node(exp.company, type="company")
            self.G.add_edge(exp_id, exp.company, relation="worked_at")

            for bullet in exp.bullets:
                for skill in resume.skills_raw:
                    if re.search(re.escape(skill), bullet, re.I):
                        self._add_skill(skill)
                        self.G.add_edge(skill, exp_id, relation="demonstrated_in")

        # ── Projects ──────────────────────────────────────────────────────────
        for i, proj in enumerate(resume.projects):
            proj_id = f"proj_{i}_{proj.title}"
            self.G.add_node(proj_id, type="project", title=proj.title,
                            description=proj.description)
            for tech in proj.technologies:
                self._add_skill(tech)
                self.G.add_edge(tech, proj_id, relation="used_in")

            # Also look for skills mentioned in description
            for skill in resume.skills_raw:
                if re.search(re.escape(skill), proj.description, re.I):
                    self._add_skill(skill)
                    self.G.add_edge(skill, proj_id, relation="used_in")

        # ── Education ─────────────────────────────────────────────────────────
        for edu in resume.education:
            edu_id = f"edu_{edu.institution}"
            self.G.add_node(edu_id, type="education",
                            institution=edu.institution,
                            degree=edu.degree, field=edu.field,
                            graduation_year=edu.graduation_year)

        return self

    def _add_skill(self, skill: str):
        if skill and not self.G.has_node(skill):
            self.G.add_node(skill, type="skill")

    # ── Query helpers ──────────────────────────────────────────────────────────

    def evidence_nodes_for_skill(self, skill: str) -> list[str]:
        """Return all experience/project nodes the skill points to."""
        if not self.G.has_node(skill):
            return []
        return [
            n for n in self.G.successors(skill)
            if self.G.nodes[n].get("type") in ("experience", "project")
        ]

    def skills_by_year(self) -> list[tuple[int, str]]:
        """Return (year, skill) pairs sorted by year — for learning progression."""
        pairs: list[tuple[int, str]] = []
        for node, data in self.G.nodes(data=True):
            if data.get("type") != "skill":
                continue
            for exp_id in self.G.successors(node):
                exp_data = self.G.nodes.get(exp_id, {})
                year = exp_data.get("start_year")
                if year:
                    pairs.append((year, node))
        return sorted(pairs)

    def all_skills(self) -> list[str]:
        return [n for n, d in self.G.nodes(data=True) if d.get("type") == "skill"]

    def all_projects(self) -> list[dict]:
        return [
            {"id": n, **d}
            for n, d in self.G.nodes(data=True)
            if d.get("type") == "project"
        ]

    def skill_project_count(self, skill: str) -> int:
        if not self.G.has_node(skill):
            return 0
        return sum(
            1 for n in self.G.successors(skill)
            if self.G.nodes[n].get("type") == "project"
        )

    def skill_experience_count(self, skill: str) -> int:
        if not self.G.has_node(skill):
            return 0
        return sum(
            1 for n in self.G.successors(skill)
            if self.G.nodes[n].get("type") == "experience"
        )
