"""Charts — Plotly chart builders."""
import plotly.express as px


def county_comparison_chart(df, metric, name_col="county"):
    if name_col not in df.columns or metric not in df.columns:
        return px.bar(title="No data")

    sorted_df = df.sort_values(metric, ascending=True).tail(15)
    fig = px.bar(
        sorted_df, x=metric, y=name_col, orientation="h",
        color=metric, color_continuous_scale="RdYlGn_r",
        labels={metric: metric.replace("_", " ").title(), name_col: ""},
    )
    fig.update_layout(
        height=400, margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        coloraxis_showscale=False,
    )
    return fig
