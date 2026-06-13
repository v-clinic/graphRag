"""
VClinic GraphRAG evaluator — RAGAS + custom medical metrics.

Metrics applied
---------------
RAGAS (LLM-as-judge, requires `pip install ragas`):
  faithfulness        — answer is grounded in retrieved contexts (anti-hallucination).
                        Critical for clinical safety.
  answer_relevancy    — answer actually addresses the question asked.

Custom clinical metrics (always available, no extra deps):
  entity_recall       — expected medical entities appear in the answer.
  clinical_code_accuracy — expected SNOMED / RxNorm / ICD-10 codes appear.
  refusal_flag        — 1.0 if the agent said "I don't know / no data",
                        0.0 if a substantive answer was returned.

Quick start
-----------
    from src.evaluation.evaluator import GraphRAGEvaluator

    ev = GraphRAGEvaluator()

    # Run population-level samples only (no patient-name placeholders needed)
    report = ev.run(
        "src/evaluation/data/eval_dataset.json",
        skip_patient_specific=True,
    )
    ev.print_report(report)

    # Save full JSON report
    ev.save_report(report, "src/evaluation/data/results/run_001.json")

    # Evaluate a single ad-hoc question
    result = ev.evaluate_single(
        "Which patients have both diabetes and hypertension?",
        reference="The answer should list patient names who have been diagnosed with both type 2 diabetes (SNOMED 44054006) and hypertension.",
        expected_codes=["44054006"],
    )
    print(result.ragas_scores)
    print(result.custom_scores)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import settings
from src.evaluation.metrics import MetricResult, compute_custom_metrics
from src.evaluation.runner import AgentRunResult, run_with_context

# ── Optional RAGAS import ─────────────────────────────────────────────────────
# RAGAS is an optional dependency.  When it is not installed the evaluator
# falls back to custom metrics only and logs a warning.

_RAGAS_AVAILABLE = False
try:
    from ragas import EvaluationDataset, evaluate  # type: ignore[import]
    from ragas.dataset_schema import SingleTurnSample  # type: ignore[import]
    from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore[import]
    from ragas.llms import LangchainLLMWrapper  # type: ignore[import]
    from ragas.metrics import AnswerRelevancy, Faithfulness  # type: ignore[import]

    _RAGAS_AVAILABLE = True
except ImportError:
    pass


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class SampleEvalResult:
    sample_id: str
    category: str
    difficulty: str
    question: str
    answer: str
    contexts: list[str]
    tool_names: list[str]
    ragas_scores: dict[str, float] = field(default_factory=dict)
    custom_scores: dict[str, MetricResult] = field(default_factory=dict)
    error: Optional[str] = None
    latency_s: float = 0.0


@dataclass
class EvaluationReport:
    samples: list[SampleEvalResult] = field(default_factory=list)
    aggregate: dict[str, float] = field(default_factory=dict)
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)
    total_samples: int = 0
    failed_samples: int = 0


# ── Evaluator ─────────────────────────────────────────────────────────────────

class GraphRAGEvaluator:
    """
    End-to-end evaluator for the VClinic GraphRAG pipeline.

    Parameters
    ----------
    judge_model :
        OpenAI model used for RAGAS LLM-as-judge scoring.
        Defaults to ``settings.openai_model``.
    use_ragas :
        Whether to run RAGAS metrics.  Automatically disabled when the
        ``ragas`` package is not installed.
    """

    def __init__(
        self,
        judge_model: str | None = None,
        use_ragas: bool = True,
    ) -> None:
        self._judge_model = judge_model or settings.openai_model
        self._use_ragas = use_ragas and _RAGAS_AVAILABLE
        self._ragas_metrics: list | None = None

        if not _RAGAS_AVAILABLE and use_ragas:
            import warnings
            warnings.warn(
                "ragas package not found — RAGAS metrics are disabled. "
                "Install it with: pip install ragas",
                stacklevel=2,
            )

        if self._use_ragas:
            self._ragas_metrics = self._init_ragas_metrics()

    # ── Public interface ──────────────────────────────────────────────────────

    def run(
        self,
        dataset_path: str | Path,
        categories: list[str] | None = None,
        max_samples: int | None = None,
        skip_patient_specific: bool = True,
    ) -> EvaluationReport:
        """
        Load an evaluation dataset, run the agent on each question, and
        compute all metrics.

        Parameters
        ----------
        dataset_path :
            Path to ``eval_dataset.json``.
        categories :
            Optional filter — only evaluate samples in these categories
            (e.g. ``["population_query", "condition_query"]``).
        max_samples :
            Cap the number of samples evaluated (useful for smoke tests).
        skip_patient_specific :
            When ``True`` (default) skip samples tagged
            ``"requires_patient_name": true``.  Set to ``False`` only after
            replacing ``{patient_name}`` placeholders with real names.
        """
        dataset_path = Path(dataset_path)
        with dataset_path.open(encoding="utf-8") as fh:
            dataset = json.load(fh)

        samples: list[dict] = dataset["samples"]

        if skip_patient_specific:
            samples = [s for s in samples if not s.get("requires_patient_name", False)]
        if categories:
            samples = [s for s in samples if s["category"] in categories]
        if max_samples is not None:
            samples = samples[:max_samples]

        results = [self._evaluate_sample(s) for s in samples]
        return self._aggregate(results)

    def evaluate_single(
        self,
        question: str,
        reference: str = "",
        expected_entities: list[str] | None = None,
        expected_codes: list[str] | None = None,
    ) -> SampleEvalResult:
        """Evaluate a single ad-hoc question without a dataset file."""
        t0 = time.perf_counter()
        run_result = run_with_context(question)
        latency = time.perf_counter() - t0

        ragas_scores = (
            self._run_ragas(run_result, reference)
            if self._use_ragas
            else {}
        )
        custom_scores = compute_custom_metrics(
            run_result.answer,
            expected_entities=expected_entities,
            expected_codes=expected_codes,
        )

        return SampleEvalResult(
            sample_id="adhoc",
            category="adhoc",
            difficulty="N/A",
            question=question,
            answer=run_result.answer,
            contexts=run_result.contexts,
            tool_names=run_result.tool_names,
            ragas_scores=ragas_scores,
            custom_scores=custom_scores,
            error=run_result.error,
            latency_s=latency,
        )

    def save_report(self, report: EvaluationReport, output_path: str | Path) -> None:
        """Persist an EvaluationReport to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "total_samples": report.total_samples,
            "failed_samples": report.failed_samples,
            "aggregate": report.aggregate,
            "by_category": report.by_category,
            "by_difficulty": report.by_difficulty,
            "samples": [
                {
                    "id": r.sample_id,
                    "category": r.category,
                    "difficulty": r.difficulty,
                    "question": r.question,
                    "answer": r.answer,
                    "contexts": r.contexts,
                    "tool_names": r.tool_names,
                    "ragas_scores": r.ragas_scores,
                    "custom_scores": {
                        k: {"score": v.score, "details": v.details}
                        for k, v in r.custom_scores.items()
                    },
                    "latency_s": round(r.latency_s, 3),
                    "error": r.error,
                }
                for r in report.samples
            ],
        }
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    def print_report(self, report: EvaluationReport) -> None:
        """Print a human-readable summary using Rich (falls back to plain text)."""
        try:
            from rich.console import Console  # noqa: PLC0415
            from rich.table import Table       # noqa: PLC0415
            _rich_print(report)
        except ImportError:
            _plain_print(report)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _init_ragas_metrics(self) -> list:
        llm = ChatOpenAI(
            model=self._judge_model,
            temperature=0,
            openai_api_key=settings.openai_api_key,
        )
        embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        wrapped_llm = LangchainLLMWrapper(llm)
        wrapped_emb = LangchainEmbeddingsWrapper(embeddings)
        return [
            Faithfulness(llm=wrapped_llm),
            AnswerRelevancy(llm=wrapped_llm, embeddings=wrapped_emb),
        ]

    def _evaluate_sample(self, sample: dict) -> SampleEvalResult:
        t0 = time.perf_counter()
        run_result = run_with_context(sample["question"])
        latency = time.perf_counter() - t0

        ragas_scores = (
            self._run_ragas(run_result, sample.get("ground_truth", ""))
            if self._use_ragas
            else {}
        )
        custom_scores = compute_custom_metrics(
            run_result.answer,
            expected_entities=sample.get("expected_entities", []),
            expected_codes=sample.get("expected_codes", []),
        )

        return SampleEvalResult(
            sample_id=sample["id"],
            category=sample["category"],
            difficulty=sample["difficulty"],
            question=sample["question"],
            answer=run_result.answer,
            contexts=run_result.contexts,
            tool_names=run_result.tool_names,
            ragas_scores=ragas_scores,
            custom_scores=custom_scores,
            error=run_result.error,
            latency_s=latency,
        )

    def _run_ragas(self, run: AgentRunResult, reference: str) -> dict[str, float]:
        """Run RAGAS metrics on a single AgentRunResult."""
        try:
            sample = SingleTurnSample(
                user_input=run.question,
                retrieved_contexts=run.contexts if run.contexts else [""],
                response=run.answer or "",
                reference=reference or run.answer or "",
            )
            dataset = EvaluationDataset(samples=[sample])
            result = evaluate(dataset=dataset, metrics=self._ragas_metrics)

            scores: dict[str, float] = {}
            for metric in self._ragas_metrics:
                metric_name: str = metric.name  # "faithfulness", "answer_relevancy", …
                raw = result.get(metric_name)
                if raw is None:
                    continue
                # RAGAS returns a list of per-sample scores; take index 0 for
                # single-sample evaluation.
                if hasattr(raw, "__iter__") and not isinstance(raw, (str, float, int)):
                    raw_list = list(raw)
                    scores[metric_name] = float(raw_list[0]) if raw_list else 0.0
                else:
                    scores[metric_name] = float(raw)

            return scores

        except Exception as exc:  # noqa: BLE001
            return {"ragas_error": str(exc)[:300]}

    # ── Aggregation ───────────────────────────────────────────────────────────

    @staticmethod
    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _aggregate(self, results: list[SampleEvalResult]) -> EvaluationReport:
        report = EvaluationReport(
            samples=results,
            total_samples=len(results),
            failed_samples=sum(1 for r in results if r.error),
        )

        def _collect(subset: list[SampleEvalResult]) -> dict[str, float]:
            ragas_buckets: dict[str, list[float]] = {}
            custom_buckets: dict[str, list[float]] = {}
            for r in subset:
                for k, v in r.ragas_scores.items():
                    if isinstance(v, float):
                        ragas_buckets.setdefault(k, []).append(v)
                for k, mr in r.custom_scores.items():
                    custom_buckets.setdefault(k, []).append(mr.score)
            return {
                **{k: self._avg(v) for k, v in ragas_buckets.items()},
                **{k: self._avg(v) for k, v in custom_buckets.items()},
            }

        report.aggregate = _collect(results)

        for cat in {r.category for r in results}:
            report.by_category[cat] = _collect([r for r in results if r.category == cat])

        for diff in {r.difficulty for r in results}:
            report.by_difficulty[diff] = _collect([r for r in results if r.difficulty == diff])

        return report


# ── Report printers ───────────────────────────────────────────────────────────

def _rich_print(report: EvaluationReport) -> None:
    from rich.console import Console  # noqa: PLC0415
    from rich.table import Table       # noqa: PLC0415

    console = Console()
    console.rule("[bold cyan]VClinic GraphRAG Evaluation Report[/bold cyan]")

    # Aggregate table
    agg = Table(title="Aggregate Scores", header_style="bold magenta")
    agg.add_column("Metric", style="cyan")
    agg.add_column("Score", justify="right")
    for metric, score in sorted(report.aggregate.items()):
        colour = "green" if score >= 0.7 else ("yellow" if score >= 0.4 else "red")
        agg.add_row(metric, f"[{colour}]{score:.3f}[/{colour}]")
    console.print(agg)

    # By-category table
    all_metrics = sorted({m for scores in report.by_category.values() for m in scores})
    cat = Table(title="Scores by Category", header_style="bold magenta")
    cat.add_column("Category", style="cyan")
    for m in all_metrics:
        cat.add_column(m, justify="right")
    for category, scores in sorted(report.by_category.items()):
        row = [category] + [f"{scores.get(m, 0.0):.3f}" for m in all_metrics]
        cat.add_row(*row)
    console.print(cat)

    console.print(
        f"\n[dim]Total: {report.total_samples} samples  |  "
        f"Failed: {report.failed_samples}[/dim]"
    )


def _plain_print(report: EvaluationReport) -> None:
    print("\n=== VClinic GraphRAG Evaluation Report ===")
    print("\nAggregate Scores:")
    for k, v in sorted(report.aggregate.items()):
        print(f"  {k}: {v:.3f}")
    print("\nBy Category:")
    for cat, scores in sorted(report.by_category.items()):
        line = "  ".join(f"{k}={v:.3f}" for k, v in sorted(scores.items()))
        print(f"  {cat}: {line}")
    print(f"\nTotal: {report.total_samples}  |  Failed: {report.failed_samples}")
