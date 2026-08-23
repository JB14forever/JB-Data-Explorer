# ==================================================================================
#  FILE: app.py
# ==================================================================================
#  This is the main entry point, run with `streamlit run app.py`. It
#  builds the page the user sees and acts as the conductor calling each
#  agent in the right order:
#
#     Upload file -> Ingestion -> Domain (AI) -> Cleaning -> Transformation
#     -> [user-triggered] ML Modelling -> NLP Q&A (AI) -> PDF Report
# ==================================================================================

import streamlit as st
import pandas as pd
import json

from agents.ingestion_agent import IngestionAgent
from agents.cleaning_agent import CleaningAgent
from agents.transformation_agent import TransformationAgent
from agents.ml_agent import MLAgent
from agents.domain_agent import DomainAgent
from agents.nlp_agent import NLPAgent
from agents.graph_describer import GraphDescriber
from agents.report_narrator import ReportNarrator

from utils.pdf_generator import generate_pdf
from utils.helpers import (
    render_health_badge, 
    df_to_plotly_heatmap, 
    df_to_plotly_histogram, 
    apply_nlp_filter,
    plotly_to_image_bytes
)

# ==========================================
# 1. Page Config
#    Basic browser tab / layout settings for the whole app.
# ==========================================
st.set_page_config(
    page_title="JB Data Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. Session State Initialization
#    Streamlit re-runs this whole script from top to bottom every time the
#    user clicks anything. "Session state" is how the app remembers things
#    (like the uploaded data, or results from a previous step) BETWEEN those
#    re-runs, instead of forgetting everything each click. Every key below
#    is set up once, on the user's very first visit, with a starting value.
# ==========================================
STATE_KEYS = {
    'raw_df': None,
    'clean_df': None,
    'transformed_df': None,
    'schema': {},
    'domain_context': {},
    'cleaning_logs': {},
    'health_score': 0.0,
    'ml_results': {},
    'saved_nlp_queries': [],
    'eda_images_bytes': [],
    'pipeline_stage': 'Awaiting Data',
    'pipeline_audit_log': [],
    'file_format': '',
    # LLM description caches
    'eda_desc_cache': {},
    'heatmap_desc_cache': '',
    # Report configuration
    'report_author': '',
    'report_title': '',
    'report_sections': {"profile": True, "cleaning": True, "eda": True, "ml": True, "insights": True, "conclusions": True},
}

for k, v in STATE_KEYS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================================
# 3. Sidebar Configuration
#    The left-hand panel: file upload control, the "run pipeline" button,
#    and a live summary of the current dataset/stage.
# ==========================================
with st.sidebar:
    uploaded_file = st.file_uploader("Upload Dataset (CSV / XLXS)", type=["csv", "xlsx", "xls"])
    run_pipeline_btn = st.button("🚀 Execute Smart Pipeline", use_container_width=True)
    
    st.divider()
    st.subheader("💓 Dataset Vital Signs")
    
    df_ref = st.session_state['raw_df']
    rows = df_ref.shape[0] if df_ref is not None else 0
    cols = df_ref.shape[1] if df_ref is not None else 0
    
    st.metric("Rows", rows)
    st.metric("Columns", cols)
    st.markdown(f"**Stage:** `{st.session_state['pipeline_stage']}`")
    
    if st.session_state['pipeline_stage'] == 'Completed':
        st.divider()
        st.subheader("📥 Export Final Report")
        st.info("Head over to the **Report Architect** tab to configure, generate, and download your consulting-grade PDF report.")

# ==========================================
# Execution Engine Pipeline
#    This block runs ONLY when the user clicks "Execute Smart Pipeline"
#    with a file already uploaded. It calls each agent in strict order -
#    Ingestion -> Domain (AI) -> Cleaning -> Transformation, and appends a
#    human-readable line to `audit` after every step, which becomes the
#    "Full Pipeline Execution Audit" table shown on screen and in the PDF.
#    This audit trail is the platform's core transparency mechanism.
# ==========================================
if run_pipeline_btn and uploaded_file:
    progress_bar = st.progress(0)
    status_text = st.empty()
    audit = []
    try:
        # Detect file format
        fname = uploaded_file.name
        ext = fname.split('.')[-1].lower()
        fmt_label = ext.upper()
        st.session_state['file_format'] = fmt_label
        st.session_state['dataset_name'] = fname
        audit.append({"#": 1, "Step": "File Upload", "Detail": f"'{fname}' received ({fmt_label} format)", "Status": "✅ Success"})

        # Phase 1: Ingestion, read the file, drop useless columns, score health.
        # (calls agents/ingestion_agent.py, see that file for the exact rules)
        status_text.text("Phase 1: Ingesting & Heuristic Checks...")
        progress_bar.progress(10)
        with st.spinner("Agent 1/4: Ingesting & Heuristic Checks..."):
            ingestor = IngestionAgent()
            raw_data = ingestor.load_data(uploaded_file)
            st.session_state['raw_df'] = raw_data
            if ext in ['xlsx', 'xls']:
                audit.append({"#": 2, "Step": "Format Conversion", "Detail": f"Excel ({fmt_label}) parsed into tabular DataFrame ({raw_data.shape[0]} rows × {raw_data.shape[1]} cols)", "Status": "✅ Success"})
            else:
                audit.append({"#": 2, "Step": "Format Parsing", "Detail": f"CSV parsed with auto-delimiter sniffing ({raw_data.shape[0]} rows × {raw_data.shape[1]} cols)", "Status": "✅ Success"})

            # Filter primary keys and Zero variance
            filtered_df, drops_1 = ingestor.filter_primary_features(raw_data)
            dropped_count = len(drops_1)
            audit.append({"#": 3, "Step": "Primary Key & Zero-Variance Filter", "Detail": f"{dropped_count} uninformative column(s) removed (ID fields, zero-variance)", "Status": "✅ Success"})

            st.session_state['health_score'] = ingestor.compute_health_score(filtered_df)
            audit.append({"#": 4, "Step": "Dataset Health Scoring", "Detail": f"Health score computed: {st.session_state['health_score']}/100", "Status": "✅ Success"})
            st.session_state['pipeline_stage'] = 'Ingestion Finished'
        progress_bar.progress(25)

        # Phase 2: Domain Context Identification, ask the AI what industry
        # this data is from and which column is the likely prediction target.
        # (calls agents/domain_agent.py, falls back to simple rules if no AI key is set)
        status_text.text("Phase 2: OpenAI Context Identification...")
        with st.spinner("Agent 2/4: OpenAI Context Identification..."):
            initial_schema = ingestor.infer_schema(filtered_df)
            sample_rows = filtered_df.head(5).to_dict(orient='records')
            audit.append({"#": 5, "Step": "Schema Inference", "Detail": f"Column types, null %, and cardinality computed for {len(initial_schema)} features", "Status": "✅ Success"})

            domain = DomainAgent()
            context = domain.analyze_context(initial_schema, sample_rows)
            st.session_state['domain_context'] = context
            api_used = domain.available
            audit.append({"#": 6, "Step": "Domain Context Identification", "Detail": f"Industry: '{context.get('industry','N/A')}', Target: '{context.get('target_variable','N/A')}', via {'OpenAI GPT-4o-mini' if api_used else 'Heuristic Fallback'}", "Status": "✅ Success"})
            st.session_state['pipeline_stage'] = 'Context Resolved'
        progress_bar.progress(50)

        # Phase 3: Rigorous 12-Step Cleaning, standardise names, fix types,
        # remove duplicates, fill in missing values, cap outliers.
        # (calls agents/cleaning_agent.py, every decision here is rule-based, not AI)
        status_text.text("Phase 3: Deep Cleaning (Missing, Outliers, Bounds)...")
        with st.spinner("Agent 3/4: Deep Cleaning (Missing, Outliers, Bounds)..."):
            cleaner = CleaningAgent()

            # Step-by-step audit entries for cleaning sub-steps
            step_n = 7
            audit.append({"#": step_n, "Step": "Column Name Standardization", "Detail": "All column names lowercased, spaces → underscores, special chars removed", "Status": "✅ Success"})
            step_n += 1
            audit.append({"#": step_n, "Step": "Data Type Fixing & Text Cleaning", "Detail": "String columns stripped, lowercased; date patterns auto-converted to datetime", "Status": "✅ Success"})
            step_n += 1

            clean_df, miss_drops, dups_removed = cleaner.clean(filtered_df, progress_bar, 50, 85)

            audit.append({"#": step_n, "Step": "Duplicate Row Removal", "Detail": f"{dups_removed} duplicate row(s) detected and removed", "Status": "✅ Success"})
            step_n += 1
            imputed = [c for c, r in miss_drops.items() if 'Imputed' in r]
            dropped_miss = [c for c, r in miss_drops.items() if 'Dropped' in r]
            audit.append({"#": step_n, "Step": "Missing Value Imputation", "Detail": f"{len(imputed)} column(s) imputed (Mean/Median/Mode), {len(dropped_miss)} column(s) dropped (>30% missing)", "Status": "✅ Success"})
            step_n += 1
            audit.append({"#": step_n, "Step": "Outlier Winsorization (IQR)", "Detail": "IQR-based clipping applied on all numeric columns; domain-specific ranges enforced (age, salary, price)", "Status": "✅ Success"})
            step_n += 1

            # Aggregate cleaning logs for PDF and UI
            formatted_logs = []
            for col, reason in drops_1.items():
                formatted_logs.append({"Phase": "Ingestion", "Column/Action": f"Drop '{col}'", "Decision Justification": reason})
            for col, reason in miss_drops.items():
                formatted_logs.append({"Phase": "Cleaning", "Column/Action": f"Modify/Drop '{col}'", "Decision Justification": reason})
            if dups_removed > 0:
                formatted_logs.append({"Phase": "Cleaning", "Column/Action": "Remove Duplicates", "Decision Justification": f"Removed {dups_removed} identical duplicate rows to ensure model integrity."})
            st.session_state['cleaning_logs'] = formatted_logs
            st.session_state['clean_df'] = clean_df

            # Re-infer schema after cleaning
            clean_schema = ingestor.infer_schema(clean_df)
            st.session_state['schema'] = clean_schema
            audit.append({"#": step_n, "Step": "Post-Cleaning Schema Re-inference", "Detail": f"Updated schema inferred on cleaned dataset ({clean_df.shape[0]} rows × {clean_df.shape[1]} cols)", "Status": "✅ Success"})
            step_n += 1
            st.session_state['pipeline_stage'] = 'Cleaning Done'
        progress_bar.progress(85)

        # Phase 4: Transformation for ML, convert categories/dates to numbers
        # and scale numeric columns so the data is ready for model training.
        # (calls agents/transformation_agent.py; this is the final automatic
        # step, machine learning modelling itself only starts when the user
        # clicks "Initiate Algorithm Sweeps" in Tab 3, further down this file)
        status_text.text("Phase 4: Transformation for ML...")
        with st.spinner("Agent 4/4: Feature Enc & Scale..."):
            transformer = TransformationAgent()
            st.session_state['transformed_df'] = transformer.transform(st.session_state['clean_df'], clean_schema)

            n_numeric = sum(1 for v in clean_schema.values() if v.get('dtype') == 'numeric')
            n_cat = sum(1 for v in clean_schema.values() if v.get('dtype') == 'categorical')
            audit.append({"#": step_n, "Step": "Categorical Encoding (One-Hot / Label)", "Detail": f"{n_cat} categorical column(s) encoded", "Status": "✅ Success"})
            step_n += 1
            audit.append({"#": step_n, "Step": "Numeric Feature Scaling", "Detail": f"{n_numeric} numeric column(s) standardized (Z-score / MinMax)", "Status": "✅ Success"})
            step_n += 1

            # Re-verify target variable
            target = context.get('target_variable', "")
            clean_cols = st.session_state['transformed_df'].columns
            if target and target.lower() in [x.lower() for x in clean_cols]:
                true_target = [x for x in clean_cols if x.lower() == target.lower()][0]
                st.session_state['domain_context']['target_variable'] = true_target

            audit.append({"#": step_n, "Step": "Dataset Ready for ML", "Detail": f"Final transformed dataset: {st.session_state['transformed_df'].shape[0]} rows × {st.session_state['transformed_df'].shape[1]} features", "Status": "✅ Success"})
            st.session_state['pipeline_stage'] = 'Completed'

        st.session_state['pipeline_audit_log'] = audit
        progress_bar.progress(100)
        status_text.text("Pipeline Execution Completed!")
        st.rerun()

    except Exception as e:
        st.sidebar.error(f"Execution Error: {e}")

# ==========================================
# Main Dashboard UI
#    Everything below only appears once the pipeline above has produced a
#    cleaned dataset. The results are organised into 5 tabs, each covering
#    one stage of the analysis (see the banner comments further down for
#    each tab). Nothing in this section re-runs the pipeline, it purely
#    displays results already stored in st.session_state.
# ==========================================
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>JB Data Explorer</h1>", unsafe_allow_html=True)

if st.session_state['clean_df'] is None:
    if st.session_state['raw_df'] is not None:
        st.warning("⚠️ The pipeline encountered an error during execution. Please check the sidebar logs.")
    else:
        st.info("👈 Upload your dataset and execute the pipeline to begin.")
else:
    t1, t2, t3, t4, t5 = st.tabs(["🚀 Phase 1: Context", "🔍 Phase 2: EDA", "🤖 Phase 3: ML Modeling", "💬 Phase 4: Query Insights", "📑 Report Architect"])
    
    # -------------------------------------
    # TAB 1: Profile & Context
    #   Shows the cleaned data preview, the AI's understanding of the
    #   dataset's industry/target, the step-by-step pipeline audit log, and
    #   the full cleaning decisions log with justifications.
    # -------------------------------------
    with t1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Data Health & Overview")
            st.markdown(render_health_badge(st.session_state['health_score']), unsafe_allow_html=True)
            st.dataframe(st.session_state['clean_df'].head(50), use_container_width=True)
        with c2:
            ctx = st.session_state['domain_context']
            if ctx:
                col_hdr, col_btn = st.columns([3, 1])
                with col_hdr:
                    st.subheader("🧠 Domain Insights")
                with col_btn:
                    st.write("")
                    if st.button("\U0001f504 Refresh", help="Refresh Phase 1 to reflect latest ML sweep results", use_container_width=True):
                        st.rerun()
                st.info(f"**Extrapolated Industry:** {ctx.get('industry', 'N/A')}")
                st.success(f"**Identified Core ML Target:** `{ctx.get('target_variable', 'None')}`")
                mlr_live = st.session_state.get('ml_results', {})
                if mlr_live and mlr_live.get('metric_name'):
                    task_t = mlr_live.get('task_type', '')
                    m_name = mlr_live.get('metric_name', '')
                    m_score = mlr_live.get('best_metric_value', 'N/A')
                    m_score_fmt = f"{m_score:.4f}" if isinstance(m_score, float) else m_score
                    if task_t == 'classification':
                        metric_display = f"{m_name}, F1-Score, ROC-AUC (Classification), Best Score: {m_score_fmt}"
                    else:
                        metric_display = f"{m_name}, MAE, R\u00b2 (Regression), Best Score: {m_score_fmt}"
                    st.warning(f"**Best Metric Strategy:** {metric_display}")
                else:
                    st.warning(f"**Best Metric Strategy:** {ctx.get('evaluation_metric', 'N/A')} *(Run ML Sweep in Phase 3 to populate actual scores)*")
                st.markdown(f"*{ctx.get('business_summary', '')}*")

            st.subheader("📋 Full Pipeline Execution Audit")
            audit_log = st.session_state.get('pipeline_audit_log', [])
            if audit_log:
                # Always patch Best Model Selection row with live ml_results so score is never stale
                mlr_patch = st.session_state.get('ml_results', {})
                if mlr_patch and mlr_patch.get('best_metric_value') is not None:
                    live_score = mlr_patch.get('best_metric_value', 0.0)
                    live_metric = mlr_patch.get('metric_name', '')
                    live_model = mlr_patch.get('best_model_name', 'N/A')
                    for row in audit_log:
                        if row.get('Step') == 'Best Model Selection':
                            row['Detail'] = f"Best model: '{live_model}' | {live_metric}: {live_score:.4f}"
                audit_df = pd.DataFrame(audit_log)
                st.dataframe(audit_df, use_container_width=True, hide_index=True)
            else:
                st.info("Audit log will populate once the pipeline runs.")

            st.subheader("🧹 Pipeline Insights & Decisions")
            if st.session_state['cleaning_logs']:
                logs_df = pd.DataFrame(st.session_state['cleaning_logs'])
                st.dataframe(logs_df, use_container_width=True, hide_index=True)
            else:
                st.write("Dataset passed baseline checks un-scathed.")

    # -------------------------------------
    # TAB 2: Exploratory Data Analysis (EDA)
    #   Lets the user pick any column to see its distribution chart, plus a
    #   correlation heatmap across all numeric columns. Each chart gets an
    #   AI-written plain-English description, and can be added ("captured")
    #   into the final PDF report with one click.
    # -------------------------------------
    with t2:
        st.subheader("Statistical Distributions")
        c3, c4 = st.columns([1, 2])
        with c3:
            s_col = st.selectbox("Analyze Element:", st.session_state['clean_df'].columns)
            st.markdown("All generated graphs using this tool can be appended to your final offline PDF report automatically.")
            
        with c4:
            if s_col:
                fig_hist = df_to_plotly_histogram(st.session_state['clean_df'], s_col)
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Dynamic LLM Description for Histogram
                if s_col not in st.session_state['eda_desc_cache']:
                    with st.spinner("Generating AI description..."):
                        describer = GraphDescriber()
                        desc = describer.describe_distribution(
                            st.session_state['clean_df'], 
                            s_col, 
                            st.session_state.get('domain_context')
                        )
                        st.session_state['eda_desc_cache'][s_col] = desc
                
                if st.session_state['eda_desc_cache'].get(s_col):
                    st.info(f"**AI Analyst Insight:** {st.session_state['eda_desc_cache'][s_col]}")
                
                if st.button(f"📸 Add {s_col} Distribution to Report"):
                    img_bytes = plotly_to_image_bytes(fig_hist)
                    if img_bytes:
                        st.session_state['eda_images_bytes'].append(img_bytes)
                        # Store description keyed by index to align with images list
                        idx = len(st.session_state['eda_images_bytes']) - 1
                        st.session_state[f'eda_desc_queued_{idx}'] = st.session_state['eda_desc_cache'].get(s_col, "")
                        st.success("Graph Captured!")
                        
        st.divider()
        st.subheader("Macro Correlations")
        fig_heat = df_to_plotly_heatmap(st.session_state['clean_df'])
        st.plotly_chart(fig_heat, use_container_width=True)
        
        # Dynamic LLM Description for Heatmap
        if not st.session_state['heatmap_desc_cache']:
            with st.spinner("Analyzing correlations..."):
                describer = GraphDescriber()
                st.session_state['heatmap_desc_cache'] = describer.describe_heatmap(
                    st.session_state['clean_df'], 
                    st.session_state.get('domain_context')
                )
                
        if st.session_state['heatmap_desc_cache']:
            st.info(f"**AI Analyst Insight:** {st.session_state['heatmap_desc_cache']}")
            
        if st.button("📸 Add Heatmap to Report"):
            h_bytes = plotly_to_image_bytes(fig_heat)
            if h_bytes:
                st.session_state['eda_images_bytes'].append(h_bytes)
                idx = len(st.session_state['eda_images_bytes']) - 1
                st.session_state[f'eda_desc_queued_{idx}'] = st.session_state['heatmap_desc_cache']
                st.success("Heatmap Captured!")

    # -------------------------------------
    # TAB 3: ML Algorithm Suite
    #   Lets the user confirm/change the prediction target, then trains and
    #   compares five machine learning algorithms at once via MLAgent
    #   (agents/ml_agent.py). Shows the leaderboard and a feature importance
    #   chart, this is the tab that made it possible to catch the target
    #   leakage issue during testing (see the README).
    # -------------------------------------
    with t3:
        st.subheader("Model Prototyping Leaderboard")
        opt_target = st.session_state['domain_context'].get('target_variable', "")
        all_cols = list(st.session_state['transformed_df'].columns)
        
        # Ensure default target exists
        idx = all_cols.index(opt_target) if opt_target in all_cols else 0
        target_col = st.selectbox("Confirm Target Variable:", all_cols, index=idx)
        
        if st.button("🚂 Initiate Algorithm Sweeps"):
            with st.spinner("Agent Training Models (Multiple)..."):
                ml = MLAgent()
                try:
                    res = ml.train(st.session_state['transformed_df'], target_col)
                    st.session_state['ml_results'] = res

                    # Update Best Metric Strategy in domain_context with actual metric used
                    actual_metric = res.get('metric_name', '')
                    task_type = res.get('task_type', '')
                    if actual_metric:
                        if task_type == 'classification':
                            metric_str = f"{actual_metric}, F1-Score, ROC-AUC (Classification)"
                        else:
                            metric_str = f"{actual_metric}, MAE, R\u00b2 (Regression)"
                        st.session_state['domain_context']['evaluation_metric'] = metric_str

                    # Append ML steps to audit log
                    audit_log = st.session_state.get('pipeline_audit_log', [])
                    # Guard: don't append ML steps more than once
                    existing_steps = [e.get('Step', '') for e in audit_log]
                    if 'Train/Test Split' not in existing_steps:
                        next_n = len(audit_log) + 1
                        leaderboard = res.get('leaderboard', [])
                        models_run = [e.get('Model', '') for e in leaderboard]
                        audit_log.append({"#": next_n, "Step": "Train/Test Split", "Detail": "Dataset split 80% training / 20% test (stratified where applicable)", "Status": "\u2705 Success"})
                        next_n += 1
                        audit_log.append({"#": next_n, "Step": "Algorithm Sweep", "Detail": f"{len(models_run)} model(s) trained: {', '.join(models_run)}", "Status": "\u2705 Success"})
                        next_n += 1
                        audit_log.append({"#": next_n, "Step": "Best Model Selection", "Detail": f"Best model: '{res.get('best_model_name','N/A')}' | {actual_metric}: {res.get('best_metric_value', 0.0):.4f}", "Status": "\u2705 Success"})
                        next_n += 1
                        if res.get('feature_importance'):
                            audit_log.append({"#": next_n, "Step": "Feature Importance Extraction", "Detail": f"{len(res['feature_importance'])} features ranked by contribution to '{target_col}'", "Status": "\u2705 Success"})
                        st.session_state['pipeline_audit_log'] = audit_log

                    # Auto-refresh entire app so Phase 1 reflects updated audit log & metric
                    st.rerun()

                except Exception as e:
                    st.error(f"Algorithm Sweep Failed: {e}")

        mlr = st.session_state['ml_results']
        if mlr:
            st.success(f"Sweep Completed: **{mlr.get('task_type', '').capitalize()}** Problem.")
            st.markdown(f"**Dominant Engine:** {mlr.get('best_model_name')}")
            
            import plotly.express as px
            # Render Leaderboard
            df_board = pd.DataFrame(mlr.get('leaderboard', []))
            st.dataframe(df_board, use_container_width=True)
            
            st.markdown("### Structural Importance Extraction")
            fi = mlr.get('feature_importance', {})
            if fi:
                df_fi = pd.DataFrame(list(fi.items()), columns=['Feature', 'Importance']).sort_values(by='Importance')
                fig_fi = px.bar(df_fi, x='Importance', y='Feature', orientation='h', title="Feature Dependency Map")
                # Add layout properties
                from utils.helpers import get_minimalist_layout
                fig_fi.update_layout(**get_minimalist_layout())
                
                st.plotly_chart(fig_fi, use_container_width=True)
                
                if st.button("📸 Attach ML Dependency Plot"):
                    btn_bytes = plotly_to_image_bytes(fig_fi)
                    if btn_bytes:
                        st.session_state['eda_images_bytes'].append(btn_bytes)
                        st.success("Algorithm Plot Captured!")
                        
    # -------------------------------------
    # TAB 4: NLP Semantic Query API
    #   The "ask a question in plain English" feature. Sends the question to
    #   NLPAgent (agents/nlp_agent.py), which returns a chart type + config;
    #   the block below then does the actual work of building that specific
    #   Plotly chart from the AI's instructions, with several fallback rules
    #   (e.g. switching a bar chart to a histogram) in case the requested
    #   configuration doesn't quite fit the data shape.
    # -------------------------------------
    with t4:
        st.subheader("Intelligent Natural Language Querying")
        st.markdown("Ask questions in plain English. The AI will choose the best chart type, filter the data, and provide data-driven insights.")
        
        sugg = st.session_state['domain_context'].get('suggested_queries', [])
        if sugg:
            st.caption("AI Ideas for this structure:")
            for s in sugg:
                st.markdown(f"- _{s}_")
                
        user_q = st.text_input("What do you want to see? (e.g., 'Show me the revenue breakdown by region')")
        
        if st.button("Generate Insight") and user_q:
            nlp = NLPAgent()
            with st.spinner("Analyzing intent and generating visualization..."):
                response = nlp.query(
                    user_q, 
                    st.session_state['clean_df'], 
                    st.session_state['clean_df'].columns.tolist(),
                    st.session_state.get('domain_context')
                )
                
                if 'error' in response and response['error']:
                    st.error(response['error'])
                else:
                    fc = response.get('filter_code')
                    ct = response.get('chart_type')
                    cfg = response.get('chart_config', {})
                    
                    df_v = apply_nlp_filter(st.session_state['clean_df'], fc)
                    
                    current_fig = None
                    if ct:
                        from plotly import express as pxe
                        from utils.helpers import get_minimalist_layout
                        try:
                            # Use LLM-provided config or fallback to first columns
                            c_x = cfg.get('x') or df_v.columns[0]
                            c_y = cfg.get('y') or (df_v.columns[1] if len(df_v.columns) > 1 else None)
                            c_color = cfg.get('color')
                            
                            kwargs = {}
                            if c_x in df_v.columns: kwargs['x'] = c_x
                            if c_y and c_y in df_v.columns: kwargs['y'] = c_y
                            if c_color and c_color in df_v.columns: kwargs['color'] = c_color
                            
                            kwargs['color_discrete_sequence'] = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4']
                            
                            c_t = ct.lower()
                            
                            # Fallback to prevent wide-form mixed-type error
                            if c_t in ['bar', 'line', 'area'] and 'y' not in kwargs:
                                if c_t == 'bar':
                                    c_t = 'histogram' # Best equivalent for single-variable count
                                else:
                                    # For line/area, try to isolate numeric columns for wide-form
                                    num_cols = df_v.select_dtypes(include='number').columns.tolist()
                                    if c_x in num_cols:
                                        num_cols.remove(c_x)
                                    if num_cols:
                                        kwargs['y'] = num_cols
                                    else:
                                        c_t = 'histogram'
                                        
                            if c_t == 'bar': current_fig = pxe.bar(df_v, **kwargs)
                            elif c_t == 'histogram': current_fig = pxe.histogram(df_v, **kwargs)
                            elif c_t == 'line': current_fig = pxe.line(df_v, **kwargs)
                            elif c_t == 'scatter': current_fig = pxe.scatter(df_v, **kwargs)
                            elif c_t == 'box': current_fig = pxe.box(df_v, **kwargs)
                            elif c_t == 'violin': current_fig = pxe.violin(df_v, box=True, points="all", **kwargs)
                            elif c_t == 'pie':
                                if c_y and c_y in df_v.columns:
                                    current_fig = pxe.pie(df_v, values=c_y, names=c_x, color_discrete_sequence=kwargs.get('color_discrete_sequence'))
                                else:
                                    # Fallback for pie without values
                                    counts = df_v[c_x].value_counts().reset_index()
                                    counts.columns = [c_x, 'count']
                                    current_fig = pxe.pie(counts, values='count', names=c_x, color_discrete_sequence=kwargs.get('color_discrete_sequence'))
                            elif c_t == 'treemap':
                                current_fig = pxe.treemap(df_v, path=[c_x], values=c_y if c_y in df_v.columns else None, color_discrete_sequence=kwargs.get('color_discrete_sequence'))
                            elif c_t == 'area': current_fig = pxe.area(df_v, **kwargs)
                            elif c_t == 'funnel': current_fig = pxe.funnel(df_v, **kwargs)
                            else:
                                # Fallback
                                y_fallback = df_v.select_dtypes(include='number').columns.tolist()
                                if df_v.columns[0] in y_fallback:
                                    y_fallback.remove(df_v.columns[0])
                                if y_fallback:
                                    current_fig = pxe.bar(df_v, x=df_v.columns[0], y=y_fallback[0], color_discrete_sequence=kwargs.get('color_discrete_sequence'))
                                else:
                                    current_fig = pxe.histogram(df_v, x=df_v.columns[0], color_discrete_sequence=kwargs.get('color_discrete_sequence'))
                                
                            if current_fig:
                                # Apply titles and labels from LLM
                                layout_updates = get_minimalist_layout()
                                if cfg.get('title'):
                                    layout_updates['title_text'] = cfg.get('title')
                                if cfg.get('labels'):
                                    current_fig.update_layout(xaxis_title=cfg['labels'].get(c_x, c_x))
                                    if c_y: current_fig.update_layout(yaxis_title=cfg['labels'].get(c_y, c_y))
                                if cfg.get('legend_title'):
                                    current_fig.update_layout(legend_title_text=cfg.get('legend_title'))
                                    
                                current_fig.update_layout(**layout_updates)
                                st.plotly_chart(current_fig, use_container_width=True)
                                
                                # Show AI insights
                                if response.get('figure_description'):
                                    st.caption(f"**Figure Description:** {response['figure_description']}")
                                if response.get('data_narrative'):
                                    st.info(f"**Data Narrative:** {response['data_narrative']}")
                                    
                        except Exception as e:
                            st.warning(f"Could not render {ct} chart cleanly. Showing data table instead. (Error: {e})")
                            st.dataframe(df_v.head(20))
                    else:
                        st.dataframe(df_v.head(20))
                        if response.get('data_narrative'):
                            st.info(f"**Data Narrative:** {response['data_narrative']}")
                    
                    # Store temporally
                    st.session_state['_last_nlp'] = {
                        'question': user_q,
                        'filter_logic': fc,
                        'current_fig': current_fig,
                        'figure_description': response.get('figure_description', ''),
                        'data_narrative': response.get('data_narrative', '')
                    }
                    
        # Check if there's a recent query to pin
        if '_last_nlp' in st.session_state:
            recent = st.session_state['_last_nlp']
            st.divider()
            if st.button("📌 Include This Insight in Final PDF"):
                fig_bytes = None
                if recent.get('current_fig'):
                    fig_bytes = plotly_to_image_bytes(recent['current_fig'])
                
                st.session_state['saved_nlp_queries'].append({
                    'question': recent['question'],
                    'filter_logic': recent['filter_logic'],
                    'image_bytes': fig_bytes,
                    'figure_description': recent.get('figure_description', ''),
                    'data_narrative': recent.get('data_narrative', '')
                })
                # clear the active slot to avoid saving the same insight twice
                del st.session_state['_last_nlp'] 
                st.success("Appended to Report Roster!")

    # -------------------------------------
    # TAB 5: Report Preview Architect
    #   Lets the user choose which sections to include, then triggers
    #   ReportNarrator (agents/report_narrator.py) to write the narrative
    #   text for each enabled section, and finally hands everything to
    #   generate_pdf() (utils/pdf_generator.py) to assemble the downloadable
    #   PDF report.
    # -------------------------------------
    with t5:
        st.subheader("Consulting-Grade Report Architect")
        st.markdown("Configure and generate your final analytical deliverable.")
        
        col_meta, col_opts = st.columns([1, 1])
        with col_meta:
            st.session_state['report_title'] = st.text_input("Report Title (Optional)", value=st.session_state.get('report_title', ''), help="Leave blank for AI-generated title")
            st.session_state['report_author'] = st.text_input("Author Name (Optional)", value=st.session_state.get('report_author', ''))
            
            st.markdown("### Included Content")
            st.write(f"- **EDA Graphics:** {len(st.session_state['eda_images_bytes'])} snapshots")
            st.write(f"- **Pinned NLP Insights:** {len(st.session_state['saved_nlp_queries'])} slots")
            st.write(f"- **ML Models:** {'Loaded ✅' if st.session_state['ml_results'] else 'Not Run ❌'}")
            
        with col_opts:
            st.markdown("### Section Configuration")
            # Toggles for each section
            rs = st.session_state['report_sections']
            rs['profile'] = st.checkbox("1. Data Profile & Health", value=rs['profile'])
            rs['cleaning'] = st.checkbox("2. Cleaning Narrative", value=rs['cleaning'])
            rs['eda'] = st.checkbox("3. Exploratory Data Analysis", value=rs['eda'])
            rs['ml'] = st.checkbox("4. Predictive Modeling", value=rs['ml'], disabled=not bool(st.session_state['ml_results']))
            rs['insights'] = st.checkbox("5. Analyst Insights (NLP)", value=rs['insights'], disabled=len(st.session_state['saved_nlp_queries'])==0)
            rs['conclusions'] = st.checkbox("6. Conclusions & Recommendations", value=rs['conclusions'])
            
        st.divider()
        
        has_content = bool(st.session_state['eda_images_bytes'] or st.session_state['saved_nlp_queries'] or st.session_state.get('ml_results'))
        
        if st.button("📄 Generate & Download Consulting PDF", use_container_width=True, type="primary"):
            if not has_content:
                st.warning("⚠️ Very little content has been generated. The report will mostly contain templates. Generating anyway...")
                
            with st.spinner("AI Agents drafting report narratives... This takes 10-20 seconds."):
                narrator = ReportNarrator()
                ctx = st.session_state.get('domain_context', {})
                dname = st.session_state.get('dataset_name', "Analyzed Dataset" if 'raw_df' in st.session_state else "Dataset")
                
                # Extract queued EDA descriptions
                eda_desc = {}
                for i in range(len(st.session_state['eda_images_bytes'])):
                    k = f'eda_desc_queued_{i}'
                    if k in st.session_state:
                        eda_desc[f'eda_{i}'] = st.session_state[k]
                
                # Only generate text for enabled sections to save time/tokens
                title = st.session_state['report_title'] or narrator.generate_report_title(ctx, dname)
                
                exec_sum = narrator.generate_executive_summary(ctx, st.session_state['ml_results'], st.session_state['cleaning_logs'], dname)
                
                clean_nar = ""
                if rs['cleaning']:
                    clean_nar = narrator.generate_cleaning_narrative(st.session_state['cleaning_logs'])
                    
                ml_interp = ""
                if rs['ml'] and st.session_state['ml_results']:
                    ml_interp = narrator.generate_ml_interpretation(st.session_state['ml_results'], ctx)
                    
                conclusions = ""
                if rs['conclusions']:
                    conclusions = narrator.generate_conclusions(ctx, st.session_state['ml_results'], st.session_state['saved_nlp_queries'])
                
                # Fetch row/col counts
                r_count = st.session_state['raw_df'].shape[0] if st.session_state['raw_df'] is not None else 0
                c_count = st.session_state['clean_df'].shape[0] if st.session_state['clean_df'] is not None else 0
                cc_count = st.session_state['clean_df'].shape[1] if st.session_state['clean_df'] is not None else 0

                pdf_b = generate_pdf(
                    dataset_name=dname,
                    domain_context=ctx,
                    cleaning_logs=st.session_state['cleaning_logs'],
                    ml_results=st.session_state['ml_results'],
                    eda_images=st.session_state['eda_images_bytes'],
                    saved_queries=st.session_state['saved_nlp_queries'],
                    report_title=title,
                    author_name=st.session_state['report_author'],
                    executive_summary=exec_sum,
                    cleaning_narrative=clean_nar,
                    ml_interpretation=ml_interp,
                    conclusions_text=conclusions,
                    eda_descriptions=eda_desc,
                    heatmap_description=st.session_state.get('heatmap_desc_cache', ''),
                    health_score=st.session_state.get('health_score', 0.0),
                    pipeline_audit_log=st.session_state.get('pipeline_audit_log', []),
                    enabled_sections=rs,
                    raw_row_count=r_count,
                    clean_row_count=c_count,
                    clean_col_count=cc_count
                )
                
            st.download_button(
                label="📥 Download Ready",
                data=pdf_b,
                file_name="Automated_Analytics_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="secondary"
            )
