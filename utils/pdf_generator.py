# ==================================================================================
#  FILE: utils/pdf_generator.py
# ==================================================================================
#  WHAT THIS FILE DOES (in plain English):
#  This is the "Report Architect" engine — it takes everything the pipeline
#  has produced so far (data profile, cleaning log, charts, ML leaderboard,
#  saved natural-language insights, and the AI-written narrative text) and
#  lays it all out into a single, consulting-style A4 PDF document, complete
#  with a cover page and a working Table of Contents.
#
#  HOW THE TABLE OF CONTENTS TRICK WORKS (the "two-pass" approach below):
#  A PDF's page numbers can't be known in advance — you only find out which
#  page a section landed on AFTER laying out everything before it. So this
#  function actually builds the ENTIRE report TWICE:
#    - Pass 1 ("silent" run): builds the whole PDF once just to record
#      which page number each section ends up on.
#    - Pass 2 (real run): builds the PDF again, this time already knowing
#      the correct page numbers, so the Table of Contents on page 2 can
#      show accurate numbers for sections that appear later in the document.
#  This is why the function calls itself recursively near the top — that is
#  intentional, not a bug.
# ==================================================================================

import datetime
from fpdf import FPDF
import io

def generate_pdf(
    dataset_name: str,
    domain_context: dict,
    cleaning_logs,
    ml_results: dict,
    eda_images: list,
    saved_queries: list,
    report_title: str = "",
    author_name: str = "",
    executive_summary: str = "",
    cleaning_narrative: str = "",
    ml_interpretation: str = "",
    conclusions_text: str = "",
    eda_descriptions: dict = None,
    heatmap_description: str = "",
    schema_info: dict = None,
    health_score: float = 0.0,
    pipeline_audit_log: list = None,
    enabled_sections: dict = None,
    raw_row_count: int = 0,
    clean_row_count: int = 0,
    clean_col_count: int = 0,
    _toc_data=None,
    _is_second_pass=False,
) -> bytes:
    """
    Generates a professional, template-driven A4 PDF analytics report.

    Most parameters are simply the results already produced by the other
    agents (cleaning_logs, ml_results, eda_images, etc.) plus a handful of
    optional AI-written narrative strings (executive_summary,
    cleaning_narrative, ml_interpretation, conclusions_text).

    The `_toc_data` and `_is_second_pass` parameters are INTERNAL — they
    are never set by app.py. They exist purely to support the "two-pass"
    Table of Contents trick described above; ignore them when calling this
    function from elsewhere.
    """
    if eda_descriptions is None:
        eda_descriptions = {}
    if pipeline_audit_log is None:
        pipeline_audit_log = []
    if enabled_sections is None:
        enabled_sections = {"profile": True, "cleaning": True, "eda": True, "ml": True, "insights": True, "conclusions": True}

    if _toc_data is None and not _is_second_pass:
        # ── PASS 1: silently build the whole report once just to record
        # which page number every section lands on (stored in `dummy_toc`).
        dummy_toc = []
        generate_pdf(
            dataset_name, domain_context, cleaning_logs, ml_results, eda_images, saved_queries,
            report_title, author_name, executive_summary, cleaning_narrative, ml_interpretation,
            conclusions_text, eda_descriptions, heatmap_description, schema_info, health_score,
            pipeline_audit_log, enabled_sections, raw_row_count, clean_row_count, clean_col_count,
            _toc_data=dummy_toc, _is_second_pass=True
        )
        # ── PASS 2: build the report again for real, now that dummy_toc has
        # been filled in with correct page numbers from Pass 1, so the
        # Table of Contents can display them accurately.
        return generate_pdf(
            dataset_name, domain_context, cleaning_logs, ml_results, eda_images, saved_queries,
            report_title, author_name, executive_summary, cleaning_narrative, ml_interpretation,
            conclusions_text, eda_descriptions, heatmap_description, schema_info, health_score,
            pipeline_audit_log, enabled_sections, raw_row_count, clean_row_count, clean_col_count,
            _toc_data=dummy_toc, _is_second_pass=True
        )

    final_title = report_title or f"{dataset_name} — Analytical Report"

    # ── Custom PDF class: adds a small running header/footer to every page ──
    class PDF(FPDF):
        def header(self):
            if self.page_no() > 1:
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(150, 150, 150)
                self.cell(0, 10, "Analytics Report", border=False, align="R")
                self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Small reusable drawing helpers (kept local to this function) ──
    # These just wrap repetitive FPDF styling calls (font, colour, spacing)
    # so the section-building code below stays readable.
    def section_heading(title):
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(180, 12, title, ln=True)
        pdf.set_draw_color(52, 152, 219)
        pdf.set_line_width(0.8)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 180, pdf.get_y())
        pdf.ln(6)

    def body_text(text):
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(60, 60, 60)
        # FPDF's built-in fonts only support the Latin-1 character set, so
        # any unsupported character (e.g. certain typographic quotes or
        # emoji) is safely replaced rather than crashing the PDF export.
        safe = text.encode('latin-1', 'replace').decode('latin-1') if text else ""
        pdf.multi_cell(180, 5.5, safe)
        pdf.ln(3)

    def boilerplate(text):
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(100, 100, 100)
        safe = text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(180, 5.5, safe)
        pdf.ln(3)

    def label_value(label, value):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(50, 7, label, border=0)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(60, 60, 60)
        safe = str(value).encode('latin-1', 'replace').decode('latin-1')
        pdf.cell(130, 7, safe, ln=True)

    def table_header(cols, widths):
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(44, 62, 80)
        pdf.set_text_color(255, 255, 255)
        for c, w in zip(cols, widths):
            pdf.cell(w, 8, c, border=1, fill=True, align="C")
        pdf.ln()
        pdf.set_text_color(60, 60, 60)

    def table_row(vals, widths, aligns=None):
        if aligns is None:
            aligns = ["L"] * len(vals)
        pdf.set_font("Helvetica", "", 9)
        for v, w, a in zip(vals, widths, aligns):
            safe = str(v)[:int(w/2)].encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(w, 7, safe, border=1, align=a)
        pdf.ln()

    def clean_emojis(text):
        # The audit log uses emoji (✅) for on-screen display, but FPDF's
        # built-in fonts can't render them — strip anything non-ASCII
        # before it goes into the PDF appendix.
        return text.encode('ascii', 'ignore').decode('ascii')

    # ═══════════════════════════════════════════
    # COVER PAGE
    # ═══════════════════════════════════════════
    pdf.add_page()

    # Header band
    pdf.set_fill_color(44, 62, 80)
    pdf.rect(0, 0, 210, 45, 'F')
    pdf.set_y(12)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, "JB DATA EXPLORER", align="C", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, "Automated Analytics Platform", align="C", ln=True)

    # Title
    pdf.set_y(70)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(44, 62, 80)
    safe_title = final_title.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 12, safe_title, align="C")
    pdf.ln(15)

    # Metadata block
    left = 50
    pdf.set_x(left)
    label_value("Dataset:", dataset_name)
    if author_name:
        pdf.set_x(left)
        label_value("Author:", author_name)
    if domain_context:
        pdf.set_x(left)
        label_value("Industry:", domain_context.get('industry', 'General'))
        pdf.set_x(left)
        label_value("Target Variable:", domain_context.get('target_variable', 'N/A'))

    pdf.set_x(left)
    label_value("Generated:", datetime.datetime.now().strftime("%d %B %Y"))

    # Executive Summary
    if executive_summary:
        pdf.ln(15)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 10, "Executive Summary", ln=True, align="C")
        pdf.ln(3)
        body_text(executive_summary)

    # ═══════════════════════════════════════════
    # TABLE OF CONTENTS
    # (populated using the page numbers recorded during Pass 1 — see the
    # big comment block at the top of this function)
    # ═══════════════════════════════════════════
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 12, "Table of Contents", ln=True, align="C")
    pdf.set_draw_color(52, 152, 219)
    pdf.set_line_width(0.8)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(6)

    if _toc_data:
        import re
        widths = [15, 135, 30]
        table_header(["Sr. No.", "Section Title", "Page No."], widths)
        for i, item in enumerate(_toc_data, 1):
            clean_title = re.sub(r'^\d+\.\s*', '', item['title'])
            table_row([str(i), clean_title, str(item['page'])], widths, aligns=["C", "L", "C"])
    else:
        # First pass placeholder — keeps this page's content length (and
        # therefore every later page number) identical between passes.
        pdf.cell(0, 10, "Collecting index data...", ln=True)

    # ═══════════════════════════════════════════
    # SECTION 1: DATA PROFILE & HEALTH
    # ═══════════════════════════════════════════
    if enabled_sections.get("profile", True):
        pdf.add_page()
        if _toc_data is not None and not _toc_data:
            _toc_data.append({"title": "1. Data Profile & Health Assessment", "page": pdf.page_no()})

        section_heading("1. Data Profile & Health Assessment")

        boilerplate("This section presents the structural profile of the ingested dataset, including data types, completeness metrics, and an algorithmic health assessment. The health score is computed based on missing values, duplicates, and dominant-value prevalence.")
        pdf.ln(3)

        label_value("Total Raw Rows:", str(raw_row_count))
        label_value("Clean Rows:", str(clean_row_count))
        label_value("Features:", str(clean_col_count))
        label_value("Health Score:", f"{health_score}/100")
        pdf.ln(5)

        if domain_context:
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(44, 62, 80)
            pdf.cell(180, 8, "Domain Context", ln=True)
            body_text(f"Industry: {domain_context.get('industry', 'N/A')}")
            bs = domain_context.get('business_summary', '')
            if bs:
                body_text(bs)

    # ═══════════════════════════════════════════
    # SECTION 2: DATA CLEANING
    # ═══════════════════════════════════════════
    if enabled_sections.get("cleaning", True):
        pdf.add_page()
        if _toc_data is not None and not any(d['title'].startswith("2.") for d in _toc_data):
            _toc_data.append({"title": "2. Data Cleaning Narrative", "page": pdf.page_no()})

        section_heading("2. Data Cleaning Narrative")

        boilerplate("The dataset was subjected to a rigorous multi-step cleaning architecture encompassing column standardization, data type correction, duplicate removal, missing value imputation (mean/median/mode based on distribution skewness), IQR-based outlier winsorization, and domain-specific range validation.")

        if cleaning_narrative:
            pdf.ln(2)
            body_text(cleaning_narrative)

        if cleaning_logs:
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(44, 62, 80)
            pdf.cell(180, 8, "Cleaning Decisions Log", ln=True)

            if isinstance(cleaning_logs, list):
                widths = [60, 120]
                table_header(["Action", "Justification"], widths)
                for log in cleaning_logs:
                    action = str(log.get("Column/Action", ""))
                    reason = str(log.get("Decision Justification", ""))
                    table_row([action, reason], widths)
            elif isinstance(cleaning_logs, dict):
                widths = [60, 120]
                table_header(["Column", "Reason"], widths)
                for col, reason in cleaning_logs.items():
                    table_row([col, reason], widths)

    # ═══════════════════════════════════════════
    # SECTION 3: EXPLORATORY DATA ANALYSIS
    # ═══════════════════════════════════════════
    if enabled_sections.get("eda", True):
        pdf.add_page()
        if _toc_data is not None and not any(d['title'].startswith("3.") for d in _toc_data):
            _toc_data.append({"title": "3. Exploratory Data Analysis", "page": pdf.page_no()})

        section_heading("3. Exploratory Data Analysis")

        boilerplate("The following visualizations capture the statistical distributions and inter-feature relationships within the cleaned dataset. Each chart is accompanied by an AI-generated analytical description highlighting key patterns, central tendencies, and anomalies relevant to the identified domain.")

        if eda_images:
            for i, img_bytes in enumerate(eda_images):
                if img_bytes:
                    if pdf.get_y() > 180:
                        pdf.add_page()
                    try:
                        img_io = io.BytesIO(img_bytes)
                        pdf.image(img_io, w=170)
                    except Exception:
                        pass
                    # Add description if available
                    desc_key = f"eda_{i}"
                    if desc_key in eda_descriptions:
                        pdf.ln(3)
                        body_text(eda_descriptions[desc_key])
                    pdf.ln(8)
        else:
            body_text("No visual distributions were selected for the report.")

        if heatmap_description:
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(180, 8, "Correlation Analysis Interpretation", ln=True)
            body_text(heatmap_description)

    # ═══════════════════════════════════════════
    # SECTION 4: PREDICTIVE MODELING
    # ═══════════════════════════════════════════
    if enabled_sections.get("ml", True):
        pdf.add_page()
        if _toc_data is not None and not any(d['title'].startswith("4.") for d in _toc_data):
            _toc_data.append({"title": "4. Predictive Modeling Results", "page": pdf.page_no()})

        section_heading("4. Predictive Modeling Results")

        boilerplate("An automated machine learning sweep was conducted across multiple algorithm families including tree-based ensembles, linear models, and support vector machines. Each model was evaluated on a stratified 80/20 train-test split with performance measured using industry-standard metrics.")

        if ml_results and ml_results.get('leaderboard'):
            target = domain_context.get('target_variable', 'Target') if domain_context else 'Target'
            task = ml_results.get('task_type', 'N/A').title()
            body_text(f"Task Type: {task}. Target Variable: '{target}'. Best Model: {ml_results.get('best_model_name', 'N/A')} ({ml_results.get('metric_name', '')}: {ml_results.get('best_metric_value', 0):.4f}).")

            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(180, 8, "Model Leaderboard", ln=True)

            widths = [90, 90]
            table_header(["Algorithm", ml_results.get('metric_name', 'Score')], widths)
            for entry in ml_results.get('leaderboard', []):
                model_name = str(entry.get('Model', ''))
                score = entry.get(ml_results.get('metric_name', 'Accuracy'), 0)
                score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
                table_row([model_name, score_str], widths)

            if ml_results.get('feature_importance'):
                pdf.ln(8)
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(180, 8, "Feature Importance Ranking", ln=True)
                widths = [90, 90]
                table_header(["Feature", "Importance"], widths)
                for feat, score in ml_results.get('feature_importance', {}).items():
                    table_row([str(feat)[:40], f"{score:.4f}"], widths)

            if ml_interpretation:
                pdf.ln(5)
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(180, 8, "Model Interpretation", ln=True)
                body_text(ml_interpretation)
        else:
            body_text("No machine learning models were computed for this analysis.")

    # ═══════════════════════════════════════════
    # SECTION 5: ANALYST INSIGHTS
    # ═══════════════════════════════════════════
    if enabled_sections.get("insights", True):
        pdf.add_page()
        if _toc_data is not None and not any(d['title'].startswith("5.") for d in _toc_data):
            _toc_data.append({"title": "5. Natural Language Analyst Insights", "page": pdf.page_no()})

        section_heading("5. Natural Language Analyst Insights")

        boilerplate("This section contains insights derived from natural language queries posed against the dataset. Each query was processed by an AI agent that selected the optimal visualization type, generated appropriate chart configurations, and produced contextual data narratives.")

        if saved_queries:
            for idx, sq in enumerate(saved_queries):
                if pdf.get_y() > 200:
                    pdf.add_page()
                pdf.set_font("Helvetica", "B", 11)
                pdf.set_text_color(44, 62, 80)
                q_text = str(sq.get('question', '')).encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(180, 7, f"Query {idx+1}: {q_text}")

                # Figure description
                fig_desc = sq.get('figure_description', '')
                if fig_desc:
                    pdf.set_x(10)
                    pdf.set_font("Helvetica", "I", 9)
                    pdf.set_text_color(80, 80, 80)
                    safe_desc = fig_desc.encode('latin-1', 'replace').decode('latin-1')
                    pdf.multi_cell(180, 5, safe_desc)

                # Chart image
                q_img = sq.get('image_bytes')
                if q_img:
                    pdf.ln(3)
                    try:
                        pdf.set_x(25)
                        pdf.image(io.BytesIO(q_img), w=160)
                    except Exception:
                        pass

                # Data narrative
                narrative = sq.get('data_narrative', '')
                if narrative:
                    pdf.ln(3)
                    pdf.set_x(10)
                    body_text(narrative)

                pdf.ln(8)
        else:
            body_text("No natural language queries were attached to this report.")

    # ═══════════════════════════════════════════
    # SECTION 6: CONCLUSIONS
    # ═══════════════════════════════════════════
    if enabled_sections.get("conclusions", True):
        pdf.add_page()
        if _toc_data is not None and not any(d['title'].startswith("6.") for d in _toc_data):
            _toc_data.append({"title": "6. Conclusions & Recommendations", "page": pdf.page_no()})

        section_heading("6. Conclusions & Recommendations")

        boilerplate("The following conclusions are synthesized from the complete analytical pipeline — spanning data profiling, cleaning, exploratory analysis, predictive modeling, and ad-hoc query insights.")

        if conclusions_text:
            body_text(conclusions_text)
        else:
            body_text("No automated conclusions were generated. Consider running the full pipeline and adding insights to populate this section.")

    # ═══════════════════════════════════════════
    # APPENDIX: PIPELINE AUDIT LOG
    # ═══════════════════════════════════════════
    if pipeline_audit_log:
        pdf.add_page()
        if _toc_data is not None and not any(d['title'].startswith("Appendix") for d in _toc_data):
            _toc_data.append({"title": "Appendix: Pipeline Execution Audit", "page": pdf.page_no()})

        section_heading("Appendix: Pipeline Execution Audit")

        boilerplate("Complete trace of every step executed by the automated pipeline, from initial file ingestion through to model selection.")

        widths = [12, 55, 90, 23]
        table_header(["#", "Step", "Detail", "Status"], widths)
        for row in pipeline_audit_log:
            step = clean_emojis(str(row.get('Step', '')))
            detail = clean_emojis(str(row.get('Detail', '')))
            status = clean_emojis(str(row.get('Status', '')))
            table_row([str(row.get('#', '')), step, detail, status], widths)

    # Return the finished PDF as raw bytes, ready to be offered as a
    # Streamlit download. fpdf2's newer API returns bytes directly; the
    # `except` branch supports slightly older fpdf2 versions that still
    # use the legacy string-based `.output(dest='S')` interface.
    try:
        return bytes(pdf.output())
    except Exception:
        return pdf.output(dest='S').encode('latin1')
