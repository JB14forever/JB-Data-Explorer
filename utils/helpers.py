# ==================================================================================
#  FILE: utils/helpers.py
# ==================================================================================
#  WHAT THIS FILE DOES (in plain English):
#  A toolbox of small, reusable helper functions used by the main app screen
#  (app.py). These functions handle purely visual / mechanical jobs — drawing
#  a coloured badge, building a chart, converting a chart to an image — so
#  that app.py itself doesn't get cluttered with repetitive chart-styling
#  code. None of these functions perform statistical analysis; they only
#  display data that has already been calculated elsewhere.
# ==================================================================================

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_health_badge(score: float) -> str:
    """
    Turns a numeric Data Health Score (0-100) into a small coloured HTML
    badge for display in the app — green for a good score, orange for a
    score that needs review, red for a poor one. This is purely cosmetic;
    the score itself is calculated in agents/ingestion_agent.py.
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
    """
    A shared, reusable set of Plotly chart styling options (transparent
    background, clean fonts, muted gridlines) so that every chart in the
    app looks visually consistent. Any chart-building function below calls
    this and applies the returned settings with `fig.update_layout(**...)`.
    """
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
    Builds a correlation heatmap: a grid showing how strongly every pair of
    numeric columns move together (a value close to 1 or -1 means a strong
    relationship, close to 0 means little relationship). This is a purely
    visual step — the actual correlation numbers come from pandas' built-in
    `.corr()` calculation, not from the AI.
    """
    import numpy as np
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty or numeric_df.shape[1] < 2:
        # Not enough numeric columns to compute a meaningful correlation —
        # show a friendly empty chart instead of crashing.
        fig = go.Figure()
        fig.update_layout(title="Not enough numeric columns for correlation heatmap.")
        return fig

    corr = numeric_df.corr().round(2)

    # Use a subtle minimalist sequential palette
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
    """
    Builds a histogram (a bar chart of how often values occur) with a small
    box-plot on top, for a single column the user selects in the "EDA"
    (Exploratory Data Analysis) tab of the app.
    """
    fig = px.histogram(
        df,
        x=col,
        marginal="box",
        title=f"Distribution Analysis: {col}",
        color_discrete_sequence=['#6366f1'],  # Modern Indigo
        opacity=0.8
    )
    # Apply the shared styling defined above
    fig.update_layout(**get_minimalist_layout())
    fig.update_traces(marker_line_width=0.5, marker_line_color="white")
    return fig


def plotly_to_image_bytes(fig: go.Figure) -> bytes:
    """
    Converts an interactive on-screen chart into a static PNG image (a
    plain picture), so it can be embedded inside the downloadable PDF
    report. Uses the "kaleido" image-rendering engine under the hood.
    Returns None if the conversion fails for any reason, so the calling
    code can skip that image gracefully instead of crashing the app.
    """
    try:
        # scale=2 for retina quality
        return fig.to_image(format="png", engine="kaleido", scale=2)
    except Exception as e:
        print(f"Kaleido export error: {e}")
        return None


def apply_nlp_filter(df: pd.DataFrame, filter_code: str) -> pd.DataFrame:
    """
    Runs a small snippet of pandas code (generated by the NLP Agent in
    response to the user's plain-English question) against the current
    dataset, to filter/aggregate it for charting.

    NOTE for reviewers: `eval()` executes code, which is normally a security
    risk if it runs untrusted input. Here the risk is limited because:
      - It only ever runs code the AI generated for the CURRENT user's own
        uploaded file, in the CURRENT user's own local session.
      - The available names are restricted to just `pd` (pandas) and `df`
        (the current dataset) — no other file, network, or system access
        is exposed to the executed snippet.
    If anything goes wrong (bad code, unexpected error), the original,
    unfiltered dataset is returned instead of crashing the app.
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
