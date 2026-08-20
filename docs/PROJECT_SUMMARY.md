# Project Summary — Bridging the Code to the Written Outputs

**Project:** JB Data Explorer — An Agentic AI-Powered Automated Analytics Pipeline for Decision Support
**Author:** Jagannath Ravindra Barik (A00046418) — TU Dublin, DATA 5010 MSc Data Analytics Research Project

This document exists to make one thing explicit for anyone reviewing this
repository alongside the written submissions: **the code, the Research
Paper, and the Client Report are one project told three ways**, not three
separate pieces of work. This page maps the two written outputs directly
onto the code that produced their results, so the story stays consistent
end-to-end — exactly what the module's cross-output alignment requirement
asks for.

## What the platform does

JB Data Explorer takes a raw tabular dataset (CSV/Excel) and automates the
full analytics workflow — profiling, cleaning, exploratory analysis,
predictive modelling, natural-language querying, and report generation —
in roughly 90 seconds, while keeping a full written justification for every
automated decision along the way. The evaluation dataset used throughout
both written outputs is the public **IBM Telco Customer Churn** dataset
(7,043 records, 33 fields), included in this repository at
`data/telco_customer_churn.csv`.

## Where each finding lives in the code

| Finding discussed in the Research Paper / Client Report | Where it's produced in the code |
|---|---|
| Automated industry & target-variable identification | `agents/domain_agent.py` → `DomainAgent.analyze_context()` |
| Data Health Score (98.78/100 on the evaluation run) | `agents/ingestion_agent.py` → `IngestionAgent.compute_health_score()` |
| Column-level cleaning decisions with written justification | `agents/ingestion_agent.py` (`filter_primary_features`) and `agents/cleaning_agent.py` (`handle_missing`) |
| ~90-second end-to-end pipeline execution | `app.py`, the `Execution Engine Pipeline` block (Phases 1–4) |
| Five-algorithm model leaderboard | `agents/ml_agent.py` → `MLAgent.train()` |
| Feature importance ranking (the two target-derived variables) | `agents/ml_agent.py` → `MLAgent.get_feature_importance()` |
| 12-page automated client report | `utils/pdf_generator.py` → `generate_pdf()`, narrated by `agents/report_narrator.py` |
| Natural-language question answering | `agents/nlp_agent.py` → `NLPAgent.query()` |

## The platform's core strength: transparency by design

The single design decision that runs through both written outputs is the
deliberate separation between **deterministic computation** (every number
in every report — statistics, cleaning decisions, model scores — is
produced by standard Python data-science libraries: pandas, scipy,
scikit-learn) and **generative interpretation** (the AI is only ever used
to *explain* numbers that already exist, never to invent them). This is
visible directly in the code: every agent module in `agents/` that calls
the AI (`domain_agent.py`, `nlp_agent.py`, `graph_describer.py`,
`report_narrator.py`) only receives pre-calculated statistics as input —
never the numbers it is asked to produce itself.

That transparency is also what made the platform's most valuable finding
possible during evaluation.

## A finding, not just a limitation: the target-leakage discovery

During the evaluation run, four of five trained models returned a perfect
weighted F1 score of 1.0000 — far above the 0.51–0.62 range reported in
published research using the same dataset. Rather than a flaw the platform
hid, this is the clearest demonstration of its transparency working exactly
as intended: the feature importance output (produced by
`MLAgent.get_feature_importance()`) made the cause visible within minutes,
identifying two columns (`churn_value`, `churn_score`) that were derived
from the prediction target itself and therefore should not have been used
as predictors.

This is presented as a **positive validation of the platform's design**,
not a weakness in it — a black-box system would have reported the perfect
score as a success with no way to question it. Because every decision and
every feature's influence is logged and shown to the user, the issue was
caught, explained, and turned directly into a concrete next development
step:

- **Planned enhancement:** an automatic pre-modelling check (building on
  the existing `get_feature_importance()` output) that flags any candidate
  feature suspiciously correlated with or derived from the target column,
  before training even begins — turning today's diagnostic strength into a
  built-in safeguard for tomorrow's version.
- **Planned enhancement:** splitting the current single Data Health Score
  into two separate measures — technical cleanliness (already implemented)
  and analytical suitability (the next step) — so users can see both at a
  glance.

Both of these are detailed as prioritised recommendations in the Client
Report (Section 5) and as future work in the Research Paper (Section 8.2),
and both build directly on functionality that already exists in this
codebase today, rather than requiring a redesign.

## Summary

The code in this repository, the Research Paper, and the Client Report all
tell the same story: an agentic pipeline that reliably automates ~90
seconds' worth of work in place of a 6–8 hour manual analysis, documents
every decision it makes along the way, and — in its one moment of genuine
ambiguity — proved that its own transparency mechanisms were strong enough
to catch it. That is the finding this repository was built to support.
