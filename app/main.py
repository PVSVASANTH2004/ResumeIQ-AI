import sys
from pathlib import Path

# Ensure project root is on sys.path when running via `streamlit run app/main.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from core.database import init_db
from core.pipeline import ResumeIQPipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeIQ AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light theme CSS
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; color: #1e293b; }
    .stButton > button {
        background: #3b82f6; color: white; border: none;
        border-radius: 8px; font-weight: 600;
    }
    .stButton > button:hover { background: #2563eb; }
    [data-testid="stSidebar"] { background-color: #f1f5f9; border-right: 1px solid #e2e8f0; }
    h1, h2, h3 { color: #0f172a !important; }
    .stTabs [data-baseweb="tab"] { color: #64748b; }
    .stTabs [aria-selected="true"] { color: #3b82f6 !important; }
    hr { border-color: #e2e8f0; }
    [data-testid="stMetricValue"] { color: #0f172a; }
    .stTextArea textarea { background-color: #ffffff; color: #1e293b; }
    .stDataFrame { background-color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ── Initialise DB + pipeline ───────────────────────────────────────────────────
init_db()

@st.cache_resource(show_spinner="Loading AI engines...")
def get_pipeline() -> ResumeIQPipeline:
    return ResumeIQPipeline()

pipeline = get_pipeline()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 ResumeIQ AI")
    st.markdown("*Beyond ATS — Understanding Candidates*")
    st.divider()

    mode = st.radio("Mode", ["Single Resume", "Bulk Ranking"], index=0)

    st.divider()
    st.markdown("**Scoring Weights**")
    st.markdown("""
    | Dimension | Weight |
    |---|---|
    | Required Skill Match | 35% |
    | Experience Quality | 20% |
    | Project Complexity | 15% |
    | Evidence Score | 10% |
    | ATS Compatibility | 10% |
    | Readability | 5% |
    | Learning Progression | 5% |
    """)
    st.divider()
    st.markdown("**Hiring Thresholds**")
    st.markdown("""
    | Recommendation | Score |
    |---|---|
    | Strong Hire | ≥ 80 |
    | Interview | 65 – 79 |
    | Consider | 50 – 64 |
    | Reject | < 50 |
    """)

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("# 🧠 ResumeIQ AI")
st.markdown("##### Explainable Context-Aware Resume Intelligence Platform")
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
def _render_single_mode(pipeline: ResumeIQPipeline):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Upload Resume")
        resume_file = st.file_uploader("Resume PDF", type=["pdf"], key="resume")

    with col2:
        st.markdown("### Job Description")
        jd_text = st.text_area("Paste the job description", height=180, key="jd")

    if not resume_file or not jd_text.strip():
        st.info("Upload a resume PDF and paste a job description to begin analysis.")
        return

    if st.button("Analyze Resume", use_container_width=True):
        with st.spinner("Running ResumeIQ AI engines..."):
            try:
                result = pipeline.run(resume_file.read(), jd_text)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

        _display_results(result)


def _display_results(result):
    from app.components.radar_chart import render_radar_chart
    from app.components.score_card import render_recommendation_badge, render_score_card
    from app.components.shap_plot import render_shap_waterfall
    from app.components.skill_badges import render_extracted_skills, render_missing_skills

    sb = result.explainability.score_breakdown
    e1 = result.resume_intelligence
    e2 = result.semantic_intelligence
    e3 = result.candidate_verification
    e4 = result.candidate_quality
    e5 = result.recruiter_decision
    e6 = result.explainability

    st.divider()

    # ── Top-level scores ──────────────────────────────────────────────────────
    top_col1, top_col2 = st.columns([1, 2])

    with top_col1:
        render_recommendation_badge(e5.hiring_recommendation.value)
        st.metric("Final Score", f"{sb.final_score:.1f} / 100")
        st.metric("Interview Probability", f"{e5.interview_probability * 100:.1f}%")
        predicted = (
            max(e2.role_fit_scores, key=e2.role_fit_scores.get)
            if e2.role_fit_scores else e1.predicted_role
        )
        st.metric("Predicted Role", predicted)

    with top_col2:
        st.markdown("### Score Breakdown")
        fig = render_radar_chart(sb)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Scores", "Role Fit", "Skills", "Projects", "Explainability",
        "Missing Skills", "Suggestions", "AI Review"
    ])

    with tab1:
        # Show active JD weights so the user understands what's being scored
        if sb.active_weights:
            active = {k: v for k, v in sb.active_weights.items() if v > 0}
            weight_labels = {
                "semantic_match": "Required Skill Match",
                "education_score": "Education Match",
                "experience_quality": "Experience Quality",
                "project_complexity": "Project Complexity",
                "evidence_score": "Evidence Score",
                "ats_compatibility": "ATS Compatibility",
                "readability": "Readability",
                "learning_progression": "Learning Progression",
            }
            weight_str = " · ".join(
                f"{weight_labels.get(k, k)}: **{v*100:.0f}%**"
                for k, v in active.items()
            )
            st.caption(f"Active weights for this JD — {weight_str}")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_score_card("Semantic Match",       sb.semantic_match)
            render_score_card("Evidence Score",       sb.evidence_score)
        with c2:
            render_score_card("Experience Quality",   sb.experience_quality)
            render_score_card("Project Complexity",   sb.project_complexity)
        with c3:
            render_score_card("Skill Confidence",     sb.skill_confidence)
            render_score_card("ATS Compatibility",    sb.ats_compatibility)
        with c4:
            render_score_card("Readability",          sb.readability)
            render_score_card("Learning Progression", sb.learning_progression)
        if sb.education_score > 0:
            st.divider()
            render_score_card("Education Match", sb.education_score)

        if e1.ats_issues:
            st.markdown("**ATS Issues Detected**")
            for issue in e1.ats_issues:
                st.warning(issue)

        if e1.keyword_stuffing_detected:
            st.error(f"Keyword stuffing detected (risk: {e1.stuffing_risk * 100:.0f}%)")

    with tab2:
        st.markdown("### Resume → Role Fit")
        st.markdown(
            "How well does this resume match common engineering roles, "
            "independent of the current JD? Recruiter view: **where does this candidate best fit?**"
        )

        role_fit = e2.role_fit_scores
        if role_fit:
            import plotly.graph_objects as _go

            roles = list(role_fit.keys())
            scores = [v * 100 for v in role_fit.values()]
            colors = ["#22c55e" if s >= 60 else "#3b82f6" if s >= 40 else "#f59e0b" for s in scores]

            fig_role = _go.Figure(_go.Bar(
                x=roles, y=scores,
                marker_color=colors,
                text=[f"{s:.0f}%" for s in scores],
                textposition="outside",
            ))
            fig_role.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                font=dict(color="#e2e8f0"), height=320,
                yaxis=dict(range=[0, 110], title="Match %"),
                margin=dict(t=20),
            )
            st.plotly_chart(fig_role, use_container_width=True)

            best_role = max(role_fit, key=role_fit.get)
            best_pct = role_fit[best_role] * 100
            jd_pct = e2.semantic_score * 100
            st.info(
                f"**Best natural fit:** {best_role} ({best_pct:.0f}%)  "
                f"—  **JD match:** {jd_pct:.0f}%  "
                f"({'aligned' if best_pct - jd_pct < 15 else 'gap detected — this JD may not be the best fit for this profile'})"
            )
        else:
            st.info("Role fit scores not available.")

    with tab3:  # Skills
        st.markdown("### Extracted Skills")
        render_extracted_skills(e1.extracted_skills)

        if e3.hallucinated_skills:
            st.divider()
            st.markdown("### Potentially Unsupported Claims")
            st.warning(
                "These skills appear in the resume but lack strong evidence in projects/experience:"
            )
            for skill in e3.hallucinated_skills:
                conf = e3.skill_confidence.get(skill, 0)
                ev = e3.evidence_scores.get(skill, 0)
                st.markdown(
                    f"- **{skill}** — Confidence: {conf*100:.0f}%, Evidence: {ev*100:.0f}%"
                )

    with tab4:
        st.markdown("### Project Analysis")
        if e4.project_scores:
            import plotly.graph_objects as go
            proj_names = [ps.title[:30] for ps in e4.project_scores]
            complexities = [ps.complexity.value for ps in e4.project_scores]
            novelties = [ps.novelty_score * 100 for ps in e4.project_scores]

            complexity_colors = {
                "Beginner": "#64748b", "Intermediate": "#3b82f6",
                "Advanced": "#8b5cf6", "Production": "#22c55e",
            }
            bar_colors = [complexity_colors.get(c, "#64748b") for c in complexities]

            fig = go.Figure(data=[
                go.Bar(name="Novelty Score (%)", x=proj_names, y=novelties,
                       marker_color="#f59e0b", yaxis="y2", opacity=0.6),
            ])
            for i, (name, complexity, color) in enumerate(zip(proj_names, complexities, bar_colors)):
                fig.add_annotation(
                    x=name, y=0, text=complexity, showarrow=False,
                    font=dict(color=color, size=10), yshift=-20,
                )
            fig.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                font=dict(color="#e2e8f0"), height=350,
                yaxis2=dict(title="Novelty %", overlaying="y", side="right", color="#f59e0b"),
            )
            st.plotly_chart(fig, use_container_width=True)

            for ps in e4.project_scores:
                col_a, col_b, col_c = st.columns([3, 1, 1])
                with col_a:
                    st.markdown(f"**{ps.title}**")
                with col_b:
                    st.markdown(f"`{ps.complexity.value}`")
                with col_c:
                    st.markdown(f"Novelty: `{ps.novelty_score * 100:.0f}%`")
        else:
            st.info("No projects found in the resume.")

        lp = e4.learning_progression
        st.divider()
        st.markdown("### Learning Progression")
        st.markdown(f"- **Velocity:** {lp.velocity:.1f} new skills/year")
        st.markdown(f"- **Growth Trend:** `{lp.growth_trend.value.title()}`")

        if lp.timeline:
            import plotly.express as px
            import pandas as pd
            rows = []
            for entry in lp.timeline:
                for skill in entry.get("skills", []):
                    rows.append({"Year": entry["year"], "Skill": skill})
            if rows:
                df = pd.DataFrame(rows)
                fig = px.scatter(
                    df, x="Year", y="Skill", color="Skill",
                    title="Skill Timeline",
                    template="plotly_dark",
                )
                fig.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.markdown("### Explainability — SHAP Feature Contributions")
        fig = render_shap_waterfall(e6.shap_values)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("### Score Composition")
        weight_data = {
            "Dimension": [
                "Semantic Match", "Experience Quality", "Project Complexity",
                "Evidence Score", "ATS Compatibility",
                "Readability", "Learning Progression",
            ],
            "Score": [
                sb.semantic_match, sb.experience_quality, sb.project_complexity,
                sb.evidence_score, sb.ats_compatibility,
                sb.readability, sb.learning_progression,
            ],
            "Weight": [0.35, 0.20, 0.15, 0.10, 0.10, 0.05, 0.05],
        }
        import pandas as pd
        import plotly.express as px
        df = pd.DataFrame(weight_data)
        df["Contribution"] = df["Score"] * df["Weight"] * 100
        fig2 = px.bar(
            df, x="Contribution", y="Dimension", orientation="h",
            color="Contribution",
            color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
            title="Score Contributions to Final Score",
            template="plotly_dark",
        )
        fig2.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    with tab6:
        st.markdown("### Missing Skills Analysis")
        render_missing_skills(e2.missing_skills)

    with tab7:
        st.markdown("### Resume Improvement Suggestions")
        if e6.improvement_suggestions:
            for i, suggestion in enumerate(e6.improvement_suggestions, 1):
                st.markdown(
                    f"""
                    <div style="background:#1e293b; border-radius:8px; padding:12px 16px;
                                margin-bottom:10px; border-left:3px solid #3b82f6;">
                        <span style="color:#64748b; font-size:11px;">#{i}</span><br>
                        <span style="color:#e2e8f0;">{suggestion}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No suggestions generated.")

    with tab8:
        st.markdown("### 🤖 AI Recruiter's Review")
        st.caption("Powered by GPT-4o — language intelligence layered on top of deterministic ML scores.")
        st.divider()

        if e6.recruiter_summary:
            st.markdown(
                f"""<div style="background:#0f2027; border-radius:10px; padding:16px 20px;
                               border-left:4px solid #22c55e; margin-bottom:16px;">
                    <p style="color:#94a3b8; font-size:12px; margin:0 0 6px 0;">RECRUITER SUMMARY</p>
                    <p style="color:#e2e8f0; font-size:15px; margin:0;">{e6.recruiter_summary}</p>
                </div>""",
                unsafe_allow_html=True,
            )

        col_s, col_w = st.columns(2)
        with col_s:
            st.markdown("#### Strengths")
            for s in e6.candidate_strengths:
                st.markdown(
                    f'<div style="background:#052e16; border-radius:6px; padding:8px 12px; '
                    f'margin-bottom:6px; border-left:3px solid #22c55e;">'
                    f'<span style="color:#86efac;">✓ {s}</span></div>',
                    unsafe_allow_html=True,
                )
        with col_w:
            st.markdown("#### Gaps")
            for w in e6.candidate_weaknesses:
                st.markdown(
                    f'<div style="background:#2d0a0a; border-radius:6px; padding:8px 12px; '
                    f'margin-bottom:6px; border-left:3px solid #ef4444;">'
                    f'<span style="color:#fca5a5;">✗ {w}</span></div>',
                    unsafe_allow_html=True,
                )

        if e6.interview_questions:
            st.divider()
            st.markdown("#### Interview Questions")
            for i, q in enumerate(e6.interview_questions, 1):
                st.markdown(
                    f'<div style="background:#1e1b4b; border-radius:6px; padding:10px 14px; '
                    f'margin-bottom:6px; border-left:3px solid #818cf8;">'
                    f'<span style="color:#a5b4fc; font-size:12px;">Q{i}</span> '
                    f'<span style="color:#e2e8f0;">{q}</span></div>',
                    unsafe_allow_html=True,
                )

        if e6.ai_review:
            st.divider()
            st.markdown("#### Full Assessment")
            st.markdown(
                f"""<div style="background:#0f172a; border-radius:10px; padding:20px 24px;
                               border:1px solid #334155; line-height:1.7;">
                    <p style="color:#cbd5e1; margin:0;">{e6.ai_review}</p>
                </div>""",
                unsafe_allow_html=True,
            )


def _render_bulk_mode(pipeline: ResumeIQPipeline):
    st.markdown("### Bulk Candidate Ranking")

    col1, col2 = st.columns(2)
    with col1:
        resume_files = st.file_uploader(
            "Upload multiple resume PDFs", type=["pdf"],
            accept_multiple_files=True, key="bulk_resumes"
        )
    with col2:
        jd_text = st.text_area("Job Description", height=180, key="bulk_jd")

    if not resume_files or not jd_text.strip():
        st.info("Upload at least 2 resume PDFs and paste a job description.")
        return

    if st.button("Rank Candidates", use_container_width=True):
        pdfs = [(f.name, f.read()) for f in resume_files]

        with st.spinner(f"Analyzing {len(pdfs)} candidates..."):
            try:
                ranked = pipeline.run_bulk(pdfs, jd_text)
            except Exception as e:
                st.error(f"Bulk analysis failed: {e}")
                return

        st.markdown(f"### Results — {len(ranked)} Candidates Ranked")
        for rank, (filename, result) in enumerate(ranked, 1):
            sb = result.explainability.score_breakdown
            rec = result.recruiter_decision.hiring_recommendation.value
            rec_colors = {
                "Strong Hire": "#22c55e", "Interview": "#3b82f6",
                "Maybe": "#f59e0b", "Reject": "#ef4444",
            }
            color = rec_colors.get(rec, "#64748b")

            with st.expander(f"#{rank} — {filename}  |  Score: {sb.final_score:.1f}  |  {rec}"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Final Score", f"{sb.final_score:.1f}")
                c2.metric("Semantic Match", f"{sb.semantic_match * 100:.1f}%")
                c3.metric("Evidence Score", f"{sb.evidence_score * 100:.1f}%")
                c4.metric("Recommendation", rec)


# ── Dispatch ──────────────────────────────────────────────────────────────────
if mode == "Single Resume":
    _render_single_mode(pipeline)
else:
    _render_bulk_mode(pipeline)
