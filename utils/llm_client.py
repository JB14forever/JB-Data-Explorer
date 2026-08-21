# ==================================================================================
#  FILE: utils/llm_client.py
# ==================================================================================
#  This is the only file that talks to the AI service directly. Every agent
#  that needs AI help imports this instead of connecting to a provider on
#  its own, so if the provider ever needs to change, it only needs to
#  change here.
#
#  Note: this file never does any of the actual number crunching. It only
#  gets used for language tasks, writing summaries, explaining charts,
#  reading free-text questions. All the real calculations happen with
#  normal Python libraries elsewhere (pandas, scikit-learn, etc.).
# ==================================================================================

"""
Centralized LLM Client Factory
===============================
Single source of truth for all LLM interactions across the platform.
Uses the GitHub Models free-tier inference endpoint with a GitHub PAT.
No agent should ever import OpenAI directly, always use this module.
"""

import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# load the .env file (if there is one) as soon as this module gets imported
load_dotenv()

# ── Constants ──────────────────────────────────────────────────────
LLM_MODEL = "gpt-4o-mini"
_BASE_URL = "https://models.inference.ai.azure.com"


# ── Private token resolver ─────────────────────────────────────────
def _resolve_token() -> str | None:
    """
    Figures out which access key to use for the AI service, checking a
    few possible spots in order:
      1. GITHUB_TOKEN env var (the recommended option, set in a local
         .env file, see .env.example)
      2. OPENAI_API_KEY env var (fallback for anyone using a regular
         OpenAI key instead of GitHub Models)
      3. Streamlit's secrets store (used automatically once deployed on
         Streamlit Community Cloud)

    Returns None if nothing is found anywhere. The app is built to keep
    working even then, just without AI-generated text, see the
    "_fallback_*" methods in the agent files.
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
        # st.secrets throws if there's no secrets file at all, which is
        # normal when running locally, so this is fine to ignore
        pass

    return None


# ── Public API ─────────────────────────────────────────────────────
def is_llm_available() -> bool:
    """Quick check: is an AI key configured right now? Other modules use
    this to decide whether to call the AI or fall back to a simpler,
    rule-based description instead."""
    return _resolve_token() is not None


def get_llm_client() -> OpenAI | None:
    """
    Builds a ready-to-use connection to the AI service, pointed at the
    right endpoint with the right key already attached. Returns None if
    no key could be found, so calling code always needs to check for that
    before trying to use it.

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
