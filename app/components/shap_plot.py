import plotly.graph_objects as go


def render_shap_waterfall(shap_values: dict[str, float]) -> go.Figure:
    sorted_items = sorted(shap_values.items(), key=lambda x: x[1])
    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = ["#ef4444" if v < 0 else "#22c55e" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=11),
    ))
    fig.update_layout(
        title=dict(text="Feature Contributions (SHAP)", font=dict(color="#e2e8f0", size=14)),
        xaxis=dict(title="SHAP Value", color="#94a3b8", zeroline=True,
                   zerolinecolor="#475569", zerolinewidth=1),
        yaxis=dict(color="#e2e8f0"),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        margin=dict(l=160, r=80, t=50, b=40),
        height=380,
    )
    return fig


def render_score_breakdown_bar(shap_values: dict[str, float]) -> go.Figure:
    """Horizontal stacked bar showing positive vs negative contributions."""
    positive = {k: v for k, v in shap_values.items() if v > 0}
    negative = {k: v for k, v in shap_values.items() if v < 0}

    fig = go.Figure()
    for feat, val in positive.items():
        fig.add_trace(go.Bar(name=feat, x=[val], orientation="h",
                             marker_color="#22c55e", showlegend=True))
    for feat, val in negative.items():
        fig.add_trace(go.Bar(name=feat, x=[val], orientation="h",
                             marker_color="#ef4444", showlegend=True))

    fig.update_layout(
        barmode="relative",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        height=200,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig
