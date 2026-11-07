"""Syntactic Pruning Utility: Verbose CoT -> Telegraphic CoT State Transitions.

Literature Positioning (vs. TokenSkip, CCoT, Sketch-of-Thought):
Unlike compressed-CoT methods that treat discrete token pruning as the final inference
architecture, `telegraphic_cot` is strictly a **motivational ablation control** alongside
`verbose_cot` and budget-forced verbose CoT (`compute_matched` / `budget_forced_cot`).
It empirically isolates how much of Chain-of-Thought token volume is English grammatical
scaffolding (64.6%–67.1% of emitted tokens with <1% accuracy contribution), motivating the
transition to pure R^d Continuous Latent Looping (`latent_loop.py`).

Separates natural-language reasoning traces into:
1. Syntactic fluency tokens (discourse markers, determiners/auxiliaries, prepositions/conjunctions)
2. State-transition tokens (numeric literals, operators, state variables, target units)
"""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import BaseModel, Field

TokenCategory = Literal[
    "discourse_marker",
    "determiner_aux",
    "preposition_conj",
    "state_transition",
]

DISCOURSE_MARKERS: frozenset[str] = frozenset(
    {
        "let",
        "let's",
        "lets",
        "think",
        "step",
        "carefully",
        "first",
        "second",
        "third",
        "next",
        "now",
        "therefore",
        "thus",
        "hence",
        "so",
        "wait",
        "okay",
        "well",
        "we",
        "need",
        "want",
        "find",
        "calculate",
        "compute",
        "determine",
        "know",
        "given",
        "means",
        "conclude",
        "final",
        "finally",
        "overall",
        "altogether",
        "together",
        "problem",
        "amount",
        "number",
        "values",
        "answer",
        "get",
        "gives",
        "obtain",
        "notice",
        "observe",
        "see",
    }
)

DETERMINER_AUX: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "such",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "could",
        "should",
        "must",
        "it",
        "its",
        "she",
        "he",
        "they",
        "her",
        "his",
        "their",
        "there",
        "which",
        "who",
        "what",
        "how",
        "many",
        "much",
    }
)

PREPOSITION_CONJ: frozenset[str] = frozenset(
    {
        "of",
        "in",
        "to",
        "for",
        "with",
        "on",
        "at",
        "from",
        "by",
        "into",
        "onto",
        "about",
        "as",
        "after",
        "before",
        "during",
        "between",
        "through",
        "and",
        "or",
        "but",
        "because",
        "since",
        "if",
        "when",
        "while",
        "than",
        "then",
        "both",
        "also",
        "just",
        "only",
    }
)

STATE_OPERATOR_RE = re.compile(
    r"(\d+(?:\.\d+)?|[=+\-*/^%<>]+|->|:=|\\[a-zA-Z]+|\$)"
)


class ClassifiedToken(BaseModel):
    """Token annotated with grammatical vs. state-transition role and pruning threshold."""

    index: int = Field(ge=0)
    text: str
    category: TokenCategory
    is_syntactic: bool
    prune_threshold: float = Field(ge=0.0, le=2.0)


class PrunedTraceSummary(BaseModel):
    """Result of applying syntactic pruning at ratio in [0, 1]."""

    ratio: float = Field(ge=0.0, le=1.0)
    retained_text: str
    original_token_count: int = Field(ge=0)
    retained_token_count: int = Field(ge=0)
    pruned_token_count: int = Field(ge=0)
    token_reduction_ratio: float = Field(ge=0.0, le=1.0)
    tokens: list[ClassifiedToken]


def _deterministic_jitter(token_norm: str, index: int) -> float:
    """Stable pseudo-uniform offset in [0, 1) so tokens fade smoothly across slider [0, 1]."""
    digest = hashlib.sha256(f"{index}:{token_norm}".encode("utf-8")).digest()
    raw = int.from_bytes(digest[:4], byteorder="big")
    return (raw % 10_000) / 10_000.0


def _normalize_word(raw_token: str) -> str:
    return raw_token.strip(".,;:!?()[]{}\"'").lower()


def classify_cot_tokens(text: str) -> list[ClassifiedToken]:
    """Tokenize a verbose CoT trace and assign deterministic syntactic prune thresholds.

    Threshold bands:
    - discourse_marker: [0.04, 0.36] (pruned first: conversational scaffolding)
    - determiner_aux:   [0.37, 0.68] (pruned second: articles, copulas, auxiliaries)
    - preposition_conj: [0.69, 0.99] (pruned third: prepositions & conjunctions)
    - state_transition: 1.50         (never pruned: numbers, operators, state variables)
    """
    raw_parts = [tok for tok in text.strip().split() if tok]
    classified: list[ClassifiedToken] = []

    for idx, tok in enumerate(raw_parts):
        norm = _normalize_word(tok)
        jitter = _deterministic_jitter(norm, idx)

        if STATE_OPERATOR_RE.search(tok) and norm not in DISCOURSE_MARKERS:
            category: TokenCategory = "state_transition"
            is_syntactic = False
            threshold = 1.5
        elif norm in DISCOURSE_MARKERS or norm == "":
            category = "discourse_marker"
            is_syntactic = True
            threshold = round(0.04 + 0.32 * jitter, 4)
        elif norm in DETERMINER_AUX:
            category = "determiner_aux"
            is_syntactic = True
            threshold = round(0.37 + 0.31 * jitter, 4)
        elif norm in PREPOSITION_CONJ:
            category = "preposition_conj"
            is_syntactic = True
            threshold = round(0.69 + 0.30 * jitter, 4)
        else:
            category = "state_transition"
            is_syntactic = False
            threshold = 1.5

        classified.append(
            ClassifiedToken(
                index=idx,
                text=tok,
                category=category,
                is_syntactic=is_syntactic,
                prune_threshold=threshold,
            )
        )

    return classified


def prune_cot_trace(text: str, ratio: float = 1.0) -> PrunedTraceSummary:
    """Strip syntactic glue tokens whose `prune_threshold <= ratio` while preserving state transitions."""
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"Pruning ratio must lie in [0.0, 1.0], got {ratio}")

    tokens = classify_cot_tokens(text)
    if not tokens:
        return PrunedTraceSummary(
            ratio=ratio,
            retained_text="",
            original_token_count=0,
            retained_token_count=0,
            pruned_token_count=0,
            token_reduction_ratio=0.0,
            tokens=[],
        )

    retained = [
        t.text
        for t in tokens
        if (not t.is_syntactic) or (t.prune_threshold > ratio)
    ]
    retained_count = len(retained)
    orig_count = len(tokens)
    pruned_count = orig_count - retained_count
    reduction = pruned_count / orig_count

    return PrunedTraceSummary(
        ratio=ratio,
        retained_text=" ".join(retained),
        original_token_count=orig_count,
        retained_token_count=retained_count,
        pruned_token_count=pruned_count,
        token_reduction_ratio=round(reduction, 4),
        tokens=tokens,
    )
