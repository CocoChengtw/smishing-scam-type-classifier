"""Load and clean the IMC'25 labeled smishing dataset.

Source: Agarwal et al., "Fishing for Smishing", ACM IMC 2025 (CC BY 4.0).
https://github.com/reportsmishing/Smishing-Dataset-IMC25
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

LABEL_COL = "scam_type"
TEXT_COL = "text"

# Canonical, code-friendly label names.
LABEL_MAP = {
    "banking": "banking",
    "others": "others",
    "delivery": "delivery",
    "government": "government",
    "telecom": "telecom",
    "spam": "spam",
    "wrong number": "wrong_number",
    "hey mum/dad": "family_impersonation",
}

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Light normalization that keeps language-specific characters intact.

    The dataset already masks PII with placeholders such as <URL> and
    <PHONE_NUMBER>; we additionally mask any raw URLs that slipped through,
    lowercase, and collapse whitespace.
    """
    text = _URL_RE.sub("<URL>", str(text))
    text = _WS_RE.sub(" ", text).strip()
    return text.lower()


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Return a frame with columns: text, clean_text, label, language."""
    df = pd.read_csv(path)
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df = df[df[LABEL_COL].isin(LABEL_MAP)].copy()
    df["label"] = df[LABEL_COL].map(LABEL_MAP)
    df["language"] = df["language"].fillna("Unknown").astype(str).str.strip()
    df["clean_text"] = df[TEXT_COL].map(normalize_text)
    df = df[df["clean_text"].str.len() > 0]
    return df[[TEXT_COL, "clean_text", "label", "language"]].reset_index(drop=True)
