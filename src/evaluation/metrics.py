"""
Custom evaluation metrics for VClinic medical GraphRAG.

Supplements RAGAS with domain-specific clinical quality signals.

Metrics
-------
entity_recall          — fraction of expected clinical entities found in the response
clinical_code_accuracy — fraction of expected medical codes (SNOMED/RxNorm/ICD-10)
                         present in the response
refusal_flag           — detects responses where the agent could not find data
                         (score 1.0 = refused/no-data, 0.0 = substantive answer)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class MetricResult:
    name: str
    score: float            # 0.0–1.0 (higher is better, except refusal_flag)
    details: dict = field(default_factory=dict)


# ── Individual metrics ────────────────────────────────────────────────────────

def clinical_code_accuracy(response: str, expected_codes: list[str]) -> MetricResult:
    """
    Measures the fraction of expected medical codes (SNOMED, RxNorm, ICD-10)
    that appear verbatim in the response string.

    Returns 1.0 when no expected codes are specified (not applicable).
    """
    if not expected_codes:
        return MetricResult("clinical_code_accuracy", 1.0, {"skipped": True})

    found = [c for c in expected_codes if c in response]
    missing = [c for c in expected_codes if c not in response]
    score = len(found) / len(expected_codes)

    return MetricResult(
        "clinical_code_accuracy",
        score,
        {
            "expected": expected_codes,
            "found": found,
            "missing": missing,
            "total": len(expected_codes),
        },
    )


def entity_recall(response: str, expected_entities: list[str]) -> MetricResult:
    """
    Measures the fraction of expected clinical entities (case-insensitive
    substring match) found in the response.

    Returns 1.0 when no expected entities are specified (not applicable).
    """
    if not expected_entities:
        return MetricResult("entity_recall", 1.0, {"skipped": True})

    resp_lower = response.lower()
    found = [e for e in expected_entities if e.lower() in resp_lower]
    missing = [e for e in expected_entities if e.lower() not in resp_lower]

    return MetricResult(
        "entity_recall",
        len(found) / len(expected_entities),
        {
            "expected": expected_entities,
            "found": found,
            "missing": missing,
            "total": len(expected_entities),
        },
    )


# Patterns indicating the agent failed to retrieve relevant data.
# A high refusal_flag score across the dataset signals retrieval problems.
_REFUSAL_PATTERN = re.compile(
    r"i\s+don'?t\s+(know|have)"
    r"|no\s+(information|records|data|results)\s+(found|available|in\s+the\s+graph)"
    r"|(cannot|can't|unable\s+to)\s+find"
    r"|not\s+found\s+in\s+the\s+(graph|database)"
    r"|the\s+graph\s+doesn'?t\s+have"
    r"|no\s+matching\s+(patient|condition|medication|procedure|record)",
    re.IGNORECASE,
)


def refusal_flag(response: str) -> MetricResult:
    """
    Detects whether the agent returned a substantive answer or refused
    because it could not find data.

    Score interpretation
    --------------------
    1.0 → agent said "I don't know / no records found" (retrieval failure)
    0.0 → agent provided a substantive answer

    A high mean refusal_flag across the dataset indicates the retrieval
    pipeline is not returning relevant graph data for many questions.
    """
    refused = bool(_REFUSAL_PATTERN.search(response))
    return MetricResult(
        "refusal_flag",
        1.0 if refused else 0.0,
        {"refused": refused, "response_length": len(response)},
    )


# ── Composite helper ──────────────────────────────────────────────────────────

def compute_custom_metrics(
    response: str,
    expected_entities: list[str] | None = None,
    expected_codes: list[str] | None = None,
) -> dict[str, MetricResult]:
    """
    Run all custom medical metrics for a single response and return a
    mapping of metric name → MetricResult.
    """
    return {
        "entity_recall": entity_recall(response, expected_entities or []),
        "clinical_code_accuracy": clinical_code_accuracy(response, expected_codes or []),
        "refusal_flag": refusal_flag(response),
    }
