# ==================================================================================
#  FILE: utils/helpers.py
# ==================================================================================
#  Small reusable helper functions used by app.py: drawing a badge,
#  building a chart, converting a chart to an image. None of this does
#  any statistical analysis, it only displays numbers that were already
#  calculated elsewhere.
# ==================================================================================

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_health_badge(score: float) -> str:
    """
    Turns the Data Health Score (0-100) into a small colored badge, green
    for good, orange for needs review, red for critical. Purely cosmetic,
    the score itself gets calculated in agents/ingestion_agent.py.
    """
    if score >= 80:
        color = "#2e7d32"   # green
        text = "Excellent"
    elif score >= 50:
        color = "#ed6c02"   # orange
        text = "Needs Review"
    else:
        color = "#d32f2f"   # red
        text = "Critical"

    return f'''
    <div style="display: inline-block; padding: 0.3em 0.8em; font-size: 85%; font-weight: 600;
        border-radius: 4px; background-color: {color}; color: white; letter-spacing: 0.5px;">
        {score}/100 - {text}
    </div>
    '''


def get_minimalist_layout():
    """A shared set of Plotly styling options (transparent background,
    clean fonts, muted gridlines) so every chart looks the same. Called by
    the chart functions below and applied with fig.update_layout(**...)."""
    return dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif"),
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=False, zeroline=False, linecolor="rgba(128,128,128,0.2)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.1)", zeroline=False, linecolor="rgba(128,128,128,0.2)"),
        title_font=dict(size=18, family="Inter, sans-serif")
    )


def df_to_plotly_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Builds a correlation heatmap: a grid showing how strongly each pair of
    numeric columns moves together. Values near 1 or -1 mean a strong
    relationship, near 0 means barely any. The correlation numbers come
    from pandas' .corr(), this function just draws them.
    """
    import numpy as np
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty or numeric_df.shape[1] < 2:
        # not enough numeric columns for a real correlation, show an
        # empty chart with a message instead of crashing
        fig = go.Figure()
        fig.update_layout(title="Not enough numeric columns for correlation heatmap.")
        return fig

    corr = numeric_df.corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Purpor",
        title="Feature Correlation Matrix"
    )
    fig.update_layout(**get_minimalist_layout())
    return fig


def df_to_plotly_histogram(df: pd.DataFrame, col: str) -> go.Figure:
    """Builds a histogram with a small box-plot on top for whichever
    column the user picks in the EDA tab."""
    fig = px.histogram(
        df,
        x=col,
        marginal="box",
        title=f"Distribution Analysis: {col}",
        color_discrete_sequence=['#6366f1'],  # indigo
        opacity=0.8
    )
    fig.update_layout(**get_minimalist_layout())
    fig.update_traces(marker_line_width=0.5, marker_line_color="white")
    return fig


def plotly_to_image_bytes(fig: go.Figure) -> bytes:
    """
    Converts an interactive chart into a static PNG so it can be dropped
    into the PDF report. Uses the kaleido rendering engine. Returns None
    if the export fails for any reason, so the caller can just skip that
    image instead of the whole app crashing.
    """
    try:
        return fig.to_image(format="png", engine="kaleido", scale=2)  # scale=2 for retina quality
    except Exception as e:
        print(f"Kaleido export error: {e}")
        return None


def apply_nlp_filter(df: pd.DataFrame, filter_code: str) -> pd.DataFrame:
    """
    Runs a small snippet of pandas code, written by the NLP agent in
    response to the user's question, against the current dataset.

    Worth flagging: eval() runs code, which is normally risky with
    untrusted input. Here it's limited because the code is only ever
    something the AI generated for the current user's own uploaded file
    in their own session, and the only names available to it are `pd`
    and `df`, nothing else on the system is reachable. If anything goes
    wrong the original, unfiltered dataset is returned instead of
    crashing the app.
    """
    if not filter_code:
        return df

    try:
        filtered_df = eval(filter_code, {"pd": pd}, {"df": df})
        if isinstance(filtered_df, pd.DataFrame):
            return filtered_df
    except Exception:
        pass

    return df
