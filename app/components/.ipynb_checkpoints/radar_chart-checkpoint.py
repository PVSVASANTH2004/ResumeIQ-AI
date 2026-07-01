import plotly.graph_objects as go

from core.models import ScoreBreakdown


def render_radar_chart(sb: ScoreBreakdown) -> go.Figure:
    categories = [
        "Semantic Match", "Evidence Score", "Experience Quality",
        "Project Complexity", "Skill Confidence", "ATS Compatibility",
        "Readability", "Learning Progression",
    ]
    values = [
        sb.semantic_match, sb.evidence_score, sb.experience_quality,
        sb.project_complexity, sb.skill_confidence, sb.ats_compatibility,
        sb.readability, sb.learning_progression,
    ]
    values_pct = [round(v * 100, 1) for v in values]
    values_pct_closed = values_pct + [values_pct[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_pct_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(59, 130, 246, 0.15)",
        line=dict(color="#3b82f6", width=2),
        name="Resume Score",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#94a3b8")),
            angularaxis=dict(tickfont=dict(color="#e2e8f0")),
            bgcolor="#1e293b",
        ),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        margin=dict(l=60, r=60, t=40, b=40),
        showlegend=False,
    )
    return fig
