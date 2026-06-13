# VClinic GraphRAG — Evaluation Framework

## Folder structure

```
src/evaluation/
├── __init__.py
├── metrics.py          # Custom medical metrics (entity recall, code accuracy, refusal flag)
├── runner.py           # Runs the LangGraph agent and captures tool-call outputs as contexts
├── evaluator.py        # Main evaluator — RAGAS + custom metrics orchestration
├── requirements.txt    # Evaluation-specific dependencies
└── data/
    └── eval_dataset.json   # 30 curated QA evaluation samples
```

---

## Research summary

### RAG evaluation metrics

For a medical GraphRAG, evaluation splits into three layers:

| Layer | Metric | Why it matters for clinical data |
|---|---|---|
| Anti-hallucination | **Faithfulness** (RAGAS) | A hallucinated drug name or SNOMED code is a patient-safety risk |
| Query quality | **Answer Relevancy** (RAGAS) | Ensures the agent doesn't return tangentially related clinical data |
| Clinical accuracy | **Entity Recall** (custom) | Checks that expected drugs/conditions appear in the answer |
| Clinical accuracy | **Clinical Code Accuracy** (custom) | Verifies SNOMED, RxNorm, ICD-10 codes are reproduced correctly |
| Retrieval health | **Refusal Flag** (custom) | Detects when the graph returned no data — a retrieval failure signal |

### Libraries evaluated

| Library | Notes |
|---|---|
| **RAGAS** ✅ (chosen) | De-facto standard for RAG pipelines; LLM-as-judge; works with or without reference answers |
| **DeepEval** | Strong for medical use cases via custom `GEval` criteria; noted in `requirements.txt` |
| **TruLens** | RAG triad (answer relevance, context relevance, groundedness); noted as alternative |
| **ARES** | Stanford; automated evaluation dataset generation; noted as alternative |

---

## Evaluation dataset (`data/eval_dataset.json`)

30 curated samples covering all major query types against Synthea-format clinical data.

| Category | Count | Style |
|---|---|---|
| `population_query` | 12 | Concrete — no patient name needed, run immediately |
| `condition_query` | 2 | Template — requires `{patient_name}` replacement |
| `medication_query` | 4 | Template |
| `procedure_query` | 1 | Template |
| `allergy_query` | 1 | Template |
| `observation_query` | 2 | Template |
| `encounter_query` | 1 | Template |
| `immunization_query` | 1 | Template |
| `provider_query` | 1 | Template |
| `multi_hop` | 3 | Hard, multi-tool, cross-relationship |
| `clinical_reasoning` | 2 | Hard, requires clinical synthesis |

**Patient-specific templates** contain a `{patient_name}` placeholder.  
Replace with real names from your Neo4j database before running:

```cypher
MATCH (p:Patient) RETURN p.first_name + ' ' + p.last_name AS name LIMIT 20
```

Or use `skip_patient_specific=True` (the default) to run only the 12 population queries immediately.

### Sample schema

```json
{
  "id": "eval_001",
  "category": "population_query",
  "difficulty": "easy",
  "requires_patient_name": false,
  "question": "How many patients are currently in the VClinic database?",
  "ground_truth": "The answer must state a specific integer...",
  "expected_entities": [],
  "expected_codes": [],
  "tool_calls_expected": ["text_to_cypher"],
  "notes": "Verifies the agent can execute a simple COUNT query."
}
```

---

## Running evaluation

### 1. Install dependencies

```bash
pip install -r src/evaluation/requirements.txt
```

### 2. Run population-level samples (no patient names required)

```python
from src.evaluation.evaluator import GraphRAGEvaluator

ev = GraphRAGEvaluator()
report = ev.run(
    "src/evaluation/data/eval_dataset.json",
    skip_patient_specific=True,
)
ev.print_report(report)
ev.save_report(report, "src/evaluation/data/results/run_001.json")
```

### 3. Run all samples (after populating patient names)

```python
report = ev.run(
    "src/evaluation/data/eval_dataset.json",
    skip_patient_specific=False,
)
```

### 4. Filter by category or difficulty

```python
report = ev.run(
    "src/evaluation/data/eval_dataset.json",
    categories=["medication_query", "condition_query"],
    max_samples=10,
)
```

### 5. Evaluate a single ad-hoc question

```python
result = ev.evaluate_single(
    "Which patients have both diabetes and hypertension?",
    reference="The answer should list patients diagnosed with both SNOMED 44054006 and SNOMED 59621000.",
    expected_codes=["44054006", "59621000"],
)
print(result.ragas_scores)   # {"faithfulness": 0.92, "answer_relevancy": 0.88}
print(result.custom_scores)  # {"entity_recall": ..., "clinical_code_accuracy": ..., "refusal_flag": ...}
```

---

## Metric interpretation

| Metric | Range | Target | Notes |
|---|---|---|---|
| `faithfulness` | 0–1 | ≥ 0.85 | Critical — low score means hallucinated clinical data |
| `answer_relevancy` | 0–1 | ≥ 0.80 | Low score means the answer is off-topic |
| `entity_recall` | 0–1 | ≥ 0.75 | Fraction of expected medical entities mentioned |
| `clinical_code_accuracy` | 0–1 | ≥ 0.80 | Fraction of expected SNOMED/RxNorm codes present |
| `refusal_flag` | 0 or 1 | ≤ 0.15 mean | Mean > 0.15 across dataset signals retrieval failures |
