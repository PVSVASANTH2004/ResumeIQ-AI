import streamlit as st

from core.models import ExtractedSkills, MissingSkill


def render_skill_badges(skills: list[str], color: str = "#3b82f6", label: str = "Skills"):
    if not skills:
        st.caption(f"No {label.lower()} found.")
        return
    badges = "".join(
        f'<span style="background:{color}22; color:{color}; border:1px solid {color}; '
        f'border-radius:6px; padding:3px 10px; margin:3px; font-size:12px; display:inline-block;">'
        f'{s}</span>'
        for s in skills
    )
    st.markdown(f"<div>{badges}</div>", unsafe_allow_html=True)


def render_extracted_skills(skills: ExtractedSkills):
    categories = {
        "Programming":  (skills.programming, "#3b82f6"),
        "Frameworks":   (skills.frameworks,  "#8b5cf6"),
        "Cloud":        (skills.cloud,       "#06b6d4"),
        "Databases":    (skills.databases,   "#10b981"),
        "AI / ML":      (skills.ai_ml,       "#f59e0b"),
        "DevOps":       (skills.devops,      "#ec4899"),
        "Soft Skills":  (skills.soft_skills, "#64748b"),
    }
    for cat, (skill_list, color) in categories.items():
        if skill_list:
            st.markdown(f"**{cat}**")
            render_skill_badges(skill_list, color=color, label=cat)


def render_missing_skills(missing: list[MissingSkill]):
    if not missing:
        st.success("All required skills matched!")
        return

    for ms in missing:
        importance_color = "#ef4444" if ms.importance.value == "required" else "#f59e0b"
        st.markdown(
            f"""
            <div style="background:#1e293b; border-radius:8px; padding:10px 14px;
                        margin-bottom:8px; border-left:3px solid {importance_color};">
                <span style="color:{importance_color}; font-weight:600;">{ms.skill}</span>
                <span style="color:#64748b; font-size:11px; margin-left:8px;">
                    ({ms.importance.value})
                </span>
                <div style="margin-top:4px; font-size:12px; color:#94a3b8;">
                    Current match: <b style="color:#e2e8f0;">{ms.current_match_pct:.0f}%</b>
                    &nbsp;→&nbsp;
                    With skill: <b style="color:#22c55e;">{ms.estimated_match_pct:.0f}%</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
