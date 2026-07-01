import streamlit as st


def render_score_card(label: str, value: float, max_value: float = 1.0, suffix: str = "%"):
    """Renders a metric card with a progress bar."""
    pct = value / max_value if max_value else 0
    display = f"{value * 100:.1f}{suffix}" if max_value == 1.0 else f"{value:.1f}{suffix}"

    color = "#22c55e" if pct >= 0.75 else "#f59e0b" if pct >= 0.50 else "#ef4444"

    st.markdown(
        f"""
        <div style="
            background: #1e293b;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 12px;
            border-left: 4px solid {color};
        ">
            <div style="color: #94a3b8; font-size: 12px; font-weight: 600; letter-spacing: 0.05em;">{label.upper()}</div>
            <div style="color: #f1f5f9; font-size: 26px; font-weight: 700; margin: 4px 0;">{display}</div>
            <div style="background: #334155; border-radius: 4px; height: 6px; margin-top: 8px;">
                <div style="background: {color}; width: {min(pct * 100, 100):.1f}%; height: 6px; border-radius: 4px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_badge(recommendation: str):
    color_map = {
        "Strong Hire": "#22c55e",
        "Interview":   "#3b82f6",
        "Maybe":       "#f59e0b",
        "Reject":      "#ef4444",
    }
    icon_map = {
        "Strong Hire": "⭐",
        "Interview":   "✅",
        "Maybe":       "🤔",
        "Reject":      "❌",
    }
    color = color_map.get(recommendation, "#64748b")
    icon = icon_map.get(recommendation, "")

    st.markdown(
        f"""
        <div style="
            background: {color}22;
            border: 2px solid {color};
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            margin-bottom: 16px;
        ">
            <div style="font-size: 36px;">{icon}</div>
            <div style="color: {color}; font-size: 20px; font-weight: 700; margin-top: 8px;">
                {recommendation}
            </div>
            <div style="color: #94a3b8; font-size: 12px; margin-top: 4px;">Hiring Recommendation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
