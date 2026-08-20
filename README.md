<div align="center">
  <h1>📊 JB Data Explorer</h1>
  <p><b>An Agentic AI-Powered Automated Analytics Pipeline for Decision Support</b></p>
  <p><i>DATA 5010 MSc Data Analytics Research Project — Technological University Dublin</i></p>

  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B.svg?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
  [![Plotly](https://img.shields.io/badge/Plotly-Express-3F4F75.svg?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
</div>

<br/>

> **Author:** Jagannath Ravindra Barik (Student No. A00046418) · A00046418@mytudublin.ie
> This repository is the Code & Data submission accompanying the Research
> Paper and Client Report for the same project. See
> [`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md) for how the code
> maps directly onto the findings discussed in those two documents.

## 📖 Overview

**JB Data Explorer** is a multi-agent analytics platform that takes a raw
tabular dataset and automatically carries it all the way from ingestion to
a client-ready PDF report — profiling, cleaning, exploring, modelling, and
narrating — in roughly 90 seconds, with every automated decision logged and
justified along the way.

Instead of writing manual Pandas code or configuring BI dashboards, the
platform delegates each stage to a specialised "agent." Nine agents combine
rigorous, repeatable statistical rules with a Large Language Model (LLM) —
deliberately kept to language and interpretation tasks only, never to the
actual number-crunching — so the platform can explain its results in plain
English without ever risking a fabricated statistic. This design decision
is discussed in full in the accompanying Research Paper (Section 3.4) and
Client Report (Section 2.4), and demonstrated directly in the code — see
[`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md).

The platform was evaluated using the public **IBM Telco Customer Churn**
dataset (included in this repository — see [Data](#-data-sources--access)
below), and the same evaluation run is what the Research Paper and Client
Report both discuss.

---

## ✨ Core Features

### 🧠 1. Multi-Agent Architecture
The platform delegates specialised tasks to a network of agents (all in `agents/`):
- **Ingestion Agent:** Loads the file, automatically removes uninformative columns (IDs, zero-variance fields), and computes a Data Health Score.
- **Domain Agent (AI):** Reads the schema and a small sample of rows to infer the industry, business context, likely prediction target, and problem type.
- **Cleaning Agent:** Runs a rule-based 12-step cleaning sequence — standardising names, fixing types, removing duplicates, imputing missing values (mean/median/mode, chosen by distribution skewness), and IQR-based outlier capping.
- **Transformation Agent:** Encodes categories and scales numeric features so the dataset is machine-learning ready.
- **ML Agent:** Trains and compares five algorithms (Random Forest, XGBoost, Decision Tree, Logistic/Linear Regression, and SVC/SVR) into a single leaderboard, and extracts feature importance from the winner.
- **NLP Agent (AI):** Turns a plain-English question into a Plotly chart specification plus a written data narrative.
- **Graph Describer (AI):** Writes a short analytical description of every EDA chart, grounded in statistics that were already calculated.
- **Report Narrator (AI):** Drafts the executive summary, cleaning narrative, model interpretation, and conclusions for the final PDF.

### 📈 2. Automated Exploratory Data Analysis (EDA)
Interactive distribution charts and a correlation heatmap, each with an AI-written explanation grounded in the real, pre-calculated statistics.

### 🗣️ 3. Natural Language Querying
Ask a question like *"Show me churn by contract type"* — the platform filters/aggregates the real data, chooses an appropriate chart, and writes a narrative explaining what it shows.

### 📄 4. Consulting-Grade PDF Report
One click assembles a full, client-ready PDF — cover page, an accurate two-pass Table of Contents, embedded charts, AI-written narrative sections, and a complete, transparent audit log of every pipeline step.

---

## 🔎 What we found — and how the platform proved its own transparency

During evaluation, the modelling stage produced an unusually strong
result: four of five models scored a perfect 1.0000, well above published
benchmarks for the same dataset. Rather than being a hidden flaw, this is
the platform working exactly as designed: the built-in feature-importance
output (`MLAgent.get_feature_importance()`) made the cause traceable within
minutes — two columns derived from the outcome itself were being used as
predictors. A black-box tool would have reported the perfect score as a
win with no way to question it; this platform's transparency mechanisms
caught it instead.

That finding is now a concrete, already-scoped next step rather than an
open problem: an automatic pre-modelling check that flags target-derived
features before training begins, building directly on functionality that
already exists in this codebase today. Full detail is in
[`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md), the Research Paper
(Sections 5–6), and the Client Report (Section 3.3 and Section 5).

---

## 🛠️ Technology Stack

- **App Framework:** [Streamlit](https://streamlit.io/)
- **Data Processing:** Pandas, NumPy, SciPy
- **Machine Learning:** Scikit-Learn, XGBoost
- **Visualisation:** Plotly Express, Kaleido (static image export for the PDF)
- **PDF Generation:** fpdf2
- **AI / LLM Integration:** OpenAI Python SDK, pointed at the free GitHub Models inference endpoint (an OpenAI API key also works)

---

## 📂 Project Structure

```text
jb-data-explorer/
├── app.py                       # Main Streamlit app — the screen you interact with,
│                                 # and the "conductor" that calls each agent in order
├── requirements.txt              # Exact, pinned Python package versions
├── .env.example                  # Template for your own .env file (see Configuration below)
├── LICENSE                       # MIT license
│
├── agents/                       # One file per specialised pipeline stage
│   ├── ingestion_agent.py        #   Load file, drop useless columns, health score
│   ├── domain_agent.py           #   AI: infer industry, target variable, problem type
│   ├── cleaning_agent.py         #   Rule-based cleaning (missing values, duplicates, outliers)
│   ├── transformation_agent.py   #   Encode categories, scale numbers for ML
│   ├── ml_agent.py               #   Train & compare 5 ML algorithms, feature importance
│   ├── nlp_agent.py              #   AI: plain-English question -> chart specification
│   ├── graph_describer.py        #   AI: writes descriptions of EDA charts
│   └── report_narrator.py        #   AI: writes the PDF report's narrative sections
│
├── utils/                        # Shared, reusable helpers
│   ├── helpers.py                #   Chart-building & styling helpers used by app.py
│   ├── llm_client.py             #   The ONE place the app connects to the AI service
│   └── pdf_generator.py          #   Builds the final downloadable PDF report
│
├── data/                         # Sample evaluation dataset (see Data section below)
│   └── telco_customer_churn.csv
│
├── docs/
│   └── PROJECT_SUMMARY.md        # How this code maps onto the Research Paper / Client Report
│
└── .streamlit/, .devcontainer/   # Editor/deployment configuration (optional, not required to run)
```

Every file above starts with a plain-English `FILE:` comment block
explaining its purpose, and every function has a comment describing what
it does and why — written so that a reader who doesn't code can still
follow the pipeline stage by stage.

---

## 🚀 How to Run the Code

### Prerequisites
- Python 3.10 or higher
- (Optional, for AI features) A free [GitHub Personal Access Token](https://github.com/settings/tokens) *or* an OpenAI API key

### 1. Clone the repository
```bash
git clone https://github.com/JB14forever/jb-data-explorer.git
cd jb-data-explorer
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your AI key (optional but recommended)
Copy the template and fill in your own token:
```bash
cp .env.example .env
```
Then edit `.env`:
```env
# Get a free token from https://github.com/settings/tokens (no billing needed)
GITHUB_TOKEN=your_github_token_here
```
> **The app runs without this step too.** If no key is configured, every
> AI-powered feature automatically falls back to a simpler, rule-based
> description instead of failing — see the `_fallback_*` methods in each
> agent file. Only the AI-written prose (industry summaries, chart
> descriptions, natural-language querying) is affected; all statistics,
> cleaning, and model training work identically either way.

### 5. Launch the app
```bash
streamlit run app.py
```
Your browser will open automatically at `http://localhost:8501`.

### 6. Try it out
1. Upload `data/telco_customer_churn.csv` from the sidebar (or any CSV/Excel file of your own).
2. Click **🚀 Execute Smart Pipeline** and watch the automated stages run.
3. Explore the five tabs: Context, EDA, ML Modelling, Query Insights, and Report Architect.
4. Generate and download your own PDF report from the last tab.

---

## 📦 Dependencies & Libraries

All exact versions are pinned in [`requirements.txt`](requirements.txt) for reproducibility:

| Library | Purpose |
|---|---|
| `streamlit` | The web app framework powering the whole interface |
| `pandas`, `numpy` | Data loading, cleaning, and manipulation |
| `scipy` | Statistical calculations (e.g. skewness, used in missing-value strategy) |
| `scikit-learn` | Machine learning models, scaling, encoding, evaluation metrics |
| `xgboost` | Gradient-boosted trees, one of the five candidate ML algorithms |
| `plotly`, `kaleido` | Interactive charts, and exporting them as static images for the PDF |
| `openai` | Client library used to talk to the AI inference endpoint |
| `python-dotenv` | Loads your `.env` file's configuration automatically |
| `fpdf2` | Builds the downloadable PDF report |
| `openpyxl` | Reads `.xlsx` Excel files |

---

## 🗄️ Data Sources & Access

The dataset used throughout the platform's evaluation (and discussed in
the Research Paper and Client Report) is the **IBM Telco Customer Churn**
dataset — a **publicly available, synthetic** dataset released for
instructional use, containing no records relating to real, identifiable
individuals.

- **Included in this repository:** [`data/telco_customer_churn.csv`](data/telco_customer_churn.csv) (7,043 rows, 21 columns)
- **Original source:** [IBM Telco Customer Churn dataset](https://github.com/IBM/telco-customer-churn-on-icp4d) (also widely mirrored, e.g. on Kaggle as "Telco Customer Churn")
- **License:** Released by IBM for sample/educational use

**The app itself is not limited to this dataset** — it accepts any
structured CSV or Excel file a user uploads, and will run its full
pipeline against it. The Telco dataset is simply the one used to produce
the results discussed in the written outputs, so it's included here to
make those results fully reproducible.

---

## ✅ A few extra notes worth knowing before you dig in

A handful of points that go beyond the strict submission checklist but are
worth flagging for anyone reviewing this repository:

- **Reproducibility:** every ML training run uses a fixed random seed
  (`random_state=42`), so re-running the pipeline on the same dataset
  reproduces the same leaderboard and feature importance results shown in
  the written outputs.
- **No data leaves your machine except to the AI provider, and only what's
  needed:** the AI is only ever sent column names, data types, and small
  statistical summaries — never a full raw dataset — see the "Data Ethics"
  discussion in the Client Report (Section 7.1) and Research Paper (Section 7).
- **Everything still works without an AI key configured** — this was a
  deliberate resilience decision so the platform is never a single point
  of failure; see `_fallback_context()` in `agents/domain_agent.py` for an
  example of how this is implemented.
- **This repository is public and requires no login** to view, clone, or
  run, per the module's submission requirements.
- For the fuller picture of how this code, the Research Paper, and the
  Client Report all tell the same story, see
  [`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md).

---

## 📜 License

Released under the [MIT License](LICENSE) — free to use, modify, and share.

## 🎓 Academic Integrity

This repository is submitted as part of the DATA 5010 MSc Data Analytics
Research Project at Technological University Dublin. All code was authored
by the student named above as part of this project; third-party libraries
are used under their respective open-source licenses and are listed in
[Dependencies & Libraries](#-dependencies--libraries).
