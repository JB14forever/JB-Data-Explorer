<div align="center">
  <h1>📊 JB Data Explorer</h1>
  <p><b>An Agentic AI Analytics Pipeline for Decision Support</b></p>
  <p><i>DATA 5010 MSc Data Analytics Research Project, Technological University Dublin</i></p>

  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B.svg?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
  [![Plotly](https://img.shields.io/badge/Plotly-Express-3F4F75.svg?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com/)
</div>

<br/>

> **Author:** Jagannath Ravindra Barik<br>
> **Student No:** A00046418<br>
> **Email:** A00046418@mytudublin.ie

## 🔗 Live Demo

JB Data Explorer on Streamlit : [https://jb-data-explorer.streamlit.app/](https://jb-data-explorer.streamlit.app/)

## 📖 Overview

The Data Explorer platform takes a raw tabular dataset and runs it through a full
analytics workflow, profiling, cleaning, exploring, modelling, and
narrating, in about 90 seconds, logging and justifying every automated
decision along the way.

Instead of writing manual Pandas code or wiring up a BI dashboard, the
platform splits the work across nine specialised "agents." Most of them
run on plain statistical rules; a few use an AI model, but only for
language tasks (explaining a chart, writing a summary) and never for the
actual number crunching. That split is the main design decision behind
the whole project, and it's discussed in more detail in the Research Paper
(Section 3.4) and Client Report (Section 2.4), plus a code-level view in
[`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md).

The platform was tested on the public IBM Telco Customer Churn dataset,
which is bundled in this repo (see [Data](#-data-sources--access) below).
That same run is what both written outputs discuss.

---

## ✨ Core Features

### 🧠 1. Multi-Agent Pipeline
Each stage of the workflow lives in its own file under `agents/`:
- **Ingestion Agent** loads the file, drops obviously useless columns (IDs, zero-variance fields), and scores the dataset's health.
- **Domain Agent (AI)** reads the schema and a sample of rows to guess the industry, likely target variable, and problem type.
- **Cleaning Agent** runs a rule-based cleaning pass: standardises column names, fixes types, drops duplicates, fills missing values (mean/median/mode depending on skew), and caps outliers with IQR.
- **Transformation Agent** encodes categories and scales numbers so the data is ready for modelling.
- **ML Agent** trains and compares five algorithms into a leaderboard, then pulls out feature importance from the winner.
- **NLP Agent (AI)** turns a plain-English question into a chart spec and a written narrative.
- **Graph Describer (AI)** writes a short description for every EDA chart, based on stats that were already computed.
- **Report Narrator (AI)** drafts the prose sections of the final PDF.

### 📈 2. Exploratory Data Analysis
Distribution charts and a correlation heatmap for the cleaned dataset, each with an AI-written explanation grounded in the real numbers.

### 🗣️ 3. Natural Language Querying
Type a question like *"Show me churn by contract type"* and the app filters the data, picks a chart, and writes a short narrative explaining it.

### 📄 4. PDF Report Builder
One click builds a full client-ready PDF: cover page, table of contents with correct page numbers, embedded charts, AI-written narrative, and a full audit log of every pipeline step.

---

## 🔎 A finding worth calling out

During testing, four out of five models scored a perfect 1.0000, well
above the range reported in published research on the same dataset. That
sounds like a red flag, and it should be treated as one, but the useful
part is how it got caught: the platform's own feature importance output
made the cause traceable within minutes. Two columns turned out to be
derived from the target itself, so they were leaking information a real
prediction task would never have access to.

A tool that hid its reasoning would have just reported the perfect score
as a win. Because the pipeline logs its decisions and shows what drove
each prediction, the issue surfaced on its own. That's a concrete next
step rather than an open question: an automatic check that flags
target-derived features before training starts, building on the
feature-importance code that already exists. Details are in
[`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md), the Research Paper
(Sections 5-6), and the Client Report (Section 3.3 and Section 5).

---

## 🛠️ Technology Stack

- **App Framework:** [Streamlit](https://streamlit.io/)
- **Data Processing:** Pandas, NumPy, SciPy
- **Machine Learning:** Scikit-Learn, XGBoost
- **Visualisation:** Plotly Express, Kaleido (for exporting charts into the PDF)
- **PDF Generation:** fpdf2
- **AI Integration:** OpenAI Python SDK, pointed at the free GitHub Models endpoint (a regular OpenAI key also works)

---

## 📂 Project Structure

```text
jb-data-explorer/
├── app.py                       # Main Streamlit app, the screen you interact with
│                                 # and the thing that calls each agent in order
├── requirements.txt              # Pinned Python package versions
├── .env.example                  # Template for your own .env file
│
├── agents/                       # One file per pipeline stage
│   ├── ingestion_agent.py        #   Load file, drop useless columns, health score
│   ├── domain_agent.py           #   AI: guess industry, target variable, problem type
│   ├── cleaning_agent.py         #   Rule-based cleaning (missing values, duplicates, outliers)
│   ├── transformation_agent.py   #   Encode categories, scale numbers for ML
│   ├── ml_agent.py               #   Train and compare 5 ML algorithms, feature importance
│   ├── nlp_agent.py              #   AI: plain-English question -> chart spec
│   ├── graph_describer.py        #   AI: writes descriptions for EDA charts
│   └── report_narrator.py        #   AI: writes the PDF report's narrative sections
│
├── utils/                        # Shared helpers
│   ├── helpers.py                #   Chart building/styling used by app.py
│   ├── llm_client.py             #   The one place the app talks to the AI service
│   └── pdf_generator.py          #   Builds the final PDF report
│
├── data/                         # Sample dataset used for evaluation
│   └── telco_customer_churn.csv
│
├── docs/
│   └── PROJECT_SUMMARY.md        # How the code maps onto the Research Paper / Client Report
│
└── .streamlit/, .devcontainer/   # Editor/deployment config, not required to run locally
```

Every file has a short comment block at the top explaining what it's for,
and functions are commented as they go, to enhance the code readability.

---

## 🚀 How to Run the Code

### Prerequisites
- Python 3.10 or higher
- Optional, for the AI features: a free [GitHub Personal Access Token](https://github.com/settings/tokens) or an OpenAI API key

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

### 4. Set up an AI key (optional but recommended)
```bash
cp .env.example .env
```
Then edit `.env`:
```env
# Get a free token at https://github.com/settings/tokens, no billing needed
GITHUB_TOKEN=your_github_token_here
```
> Skipping this step is fine too. Without a key, every AI-powered feature
> falls back to a simpler, rule-based description instead of failing, see
> the `_fallback_*` methods in each agent file. Only the AI-written prose
> is affected (summaries, chart descriptions, natural-language querying),
> all statistics, cleaning, and model training still run exactly the same.

### 5. Launch the app
```bash
streamlit run app.py
```
It should open automatically at `http://localhost:8501`.

### 6. Try it out
1. Upload `data/telco_customer_churn.csv` from the sidebar (or any CSV/Excel file).
2. Click **🚀 Execute Smart Pipeline** and watch the pipeline run.
3. Look through the five tabs: Context, EDA, ML Modelling, Query Insights, and Report Architect.
4. Generate and download a PDF report from the last tab.

---

## ☁️ Deploying on Streamlit Community Cloud

The app is set up to deploy as-is on [Streamlit Community Cloud](https://streamlit.io/cloud) (free tier) and below are the steps listed in order to deloy:

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with the GitHub account that owns this repository.
2. Click **New app**, pick this repository and the `main` branch, and set the main file path to `app.py`.
3. Optional, for AI-powered features to work on the deployed app: open **Advanced settings → Secrets** and add:
   ```toml
   GITHUB_TOKEN = "your_github_token_here"
   ```
4. Click **Deploy**. The build installs everything from `requirements.txt` and the app should be live in a couple of minutes.

Once deployed, it should be ready to use in action.

---

## 📦 Dependencies & Libraries

Exact versions are pinned in [`requirements.txt`](requirements.txt) for reproducibility:

| Library | Purpose |
|---|---|
| `streamlit` | Web app framework powering the interface |
| `pandas`, `numpy` | Data loading, cleaning, and manipulation |
| `scipy` | Statistical calculations (e.g. skewness, used for the missing-value strategy) |
| `scikit-learn` | ML models, scaling, encoding, evaluation metrics |
| `xgboost` | Gradient-boosted trees, one of the five candidate ML algorithms |
| `plotly`, `kaleido` | Interactive charts, and exporting them as static images for the PDF |
| `openai` | Client library used to talk to the AI inference endpoint |
| `python-dotenv` | Loads the `.env` file's configuration automatically |
| `fpdf2` | Builds the downloadable PDF report |
| `openpyxl` | Reads `.xlsx` Excel files |

---

## 🗄️ Data Sources & Access

The dataset used throughout evaluation (and discussed in the Research
Paper and Client Report) is the IBM Telco Customer Churn dataset, a
publicly available, synthetic dataset released for instructional use. It
contains no records relating to real, identifiable individuals.

- **Included in this repository:** [`data/telco_customer_churn.csv`](data/telco_customer_churn.csv) (7,043 rows, 21 columns)
- **Original source:** [IBM Telco Customer Churn dataset](https://github.com/IBM/telco-customer-churn-on-icp4d) (also mirrored on Kaggle as "Telco Customer Churn")
- **Data license:** released by IBM for sample and educational use

The app itself isn't limited to this dataset, it takes any structured CSV
or Excel file a user uploads and runs the full pipeline on it. The IBM Telco Customer Churn
dataset is just the one used to produce the results in the written
outputs, so it's included here to keep those results reproducible.

---

## ✅ A few extra notes worth knowing

- **Reproducibility:** every ML run uses a fixed random seed (`random_state=42`), so re-running the pipeline on the same data reproduces the same leaderboard and feature importance shown in the written outputs.
- **What leaves the machine:** the AI is only ever sent column names, types, and small statistical summaries, never a full raw dataset. See the Data Ethics discussion in the Client Report (Section 7.1) and Research Paper (Section 7).
- **Works without an AI key too:** a deliberate choice so the platform isn't dependent on one external service, see `_fallback_context()` in `agents/domain_agent.py` for an example.
- **Public repository, no login required** to view, clone, or run, in line with the module's submission requirements.
- For the full picture of how the code, the Research Paper, and the Client Report tie together, see [`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md).

---

## 🎓 Academic Integrity

Submitted as part of the DATA 5010 MSc Data Analytics Research Project at
Technological University Dublin. All code was written by the author
named above as part of this project. Third-party libraries are used under
their own respective licenses and listed in
[Dependencies & Libraries](#-dependencies--libraries).
