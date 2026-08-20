# ==================================================================================
#  FILE: utils/llm_client.py
# ==================================================================================
#  WHAT THIS FILE DOES (in plain English):
#  This is the ONE place in the whole app that knows how to talk to the AI
#  (language model) service. Every "agent" module that needs AI help (for
#  example, to write a summary or read a natural-language question) imports
#  this file instead of connecting to the AI provider directly.
#
#  WHY THIS MATTERS (design decision, discussed in the accompanying research
#  paper, Section 4.2 "Design Decisions"): centralising the connection here
#  means the whole platform can be pointed at a different AI provider, or
#  disabled entirely, by editing a single file — nothing else needs to change.
#
#  IMPORTANT SAFETY NOTE FOR NON-CODERS:
#  This file NEVER performs any of the actual number-crunching (statistics,
#  cleaning, model training). It is used only for language tasks: writing
#  explanations, summaries, and interpreting free-text questions. All the
#  real calculations happen elsewhere using standard, deterministic Python
#  libraries (pandas, scikit-learn, etc.), so the AI cannot invent numbers.
# ==================================================================================

"""
Centralized LLM Client Factory
===============================
Single source of truth for all LLM interactions across the platform.
Uses the GitHub Models free-tier inference endpoint with a GitHub PAT.
No agent should ever import OpenAI directly — always use this module.
"""

import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from a local ".env" file (if one exists) as
# soon as this module is imported, so the API key is available everywhere.
load_dotenv()

# ── Constants ──────────────────────────────────────────────────────
# The exact AI model requested from the inference endpoint.
LLM_MODEL = "gpt-4o-mini"
# The web address of the free GitHub Models inference service. This is the
# server the app sends requests to whenever it needs the AI's help.
_BASE_URL = "https://models.inference.ai.azure.com"


# ── Private token resolver ─────────────────────────────────────────
# "Private" (the leading underscore) means this helper is only meant to be
# used inside this file, not imported elsewhere.
def _resolve_token() -> str | None:
    """
    Figures out which access key (token) to use to talk to the AI service,
    checking a few possible locations in order until one is found:
      1. Environment variable GITHUB_TOKEN (the recommended option — set in
         your local .env file, see .env.example)
      2. Environment variable OPENAI_API_KEY (a fallback for anyone using a
         standard OpenAI key instead of GitHub Models)
      3. Streamlit's built-in "secrets" store (used automatically when the
         app is deployed on Streamlit Community Cloud)

    Returns None if no key can be found anywhere — the app is designed to
    keep working even then (see the "heuristic fallback" behaviour in the
    agent modules), just without AI-generated text.
    """
    token = os.getenv("GITHUB_TOKEN")
    if token: return token

    token = os.getenv("OPENAI_API_KEY")
    if token: return token

    # Fallback: Streamlit Secrets (for Streamlit Cloud deployments)
    try:
        if "GITHUB_TOKEN" in st.secrets:
            return st.secrets["GITHUB_TOKEN"]
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError, AttributeError):
        # st.secrets raises an error if no secrets file exists at all —
        # that's expected when running locally, so we just ignore it.
        pass

    return None


# ── Public API ─────────────────────────────────────────────────────
# These two functions are what the rest of the app is allowed to call.

def is_llm_available() -> bool:
    """Quick yes/no check: is an AI access key configured right now?
    Other modules use this to decide whether to call the AI or fall back
    to a simpler, rule-based explanation instead."""
    return _resolve_token() is not None


def get_llm_client() -> OpenAI | None:
    """
    Builds and returns a ready-to-use connection ("client") to the AI
    service, already configured with the correct web address and access
    key. Returns None (nothing) if no access key could be found, so
    calling code must always check for that before trying to use it.

    Usage:
        from utils.llm_client import get_llm_client, LLM_MODEL
        client = get_llm_client()
        if client:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[...]
            )
    """
    token = _resolve_token()
    if not token:
        return None

    return OpenAI(
        base_url=_BASE_URL,
        api_key=token
    )
