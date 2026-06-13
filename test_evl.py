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

print("-------------------------")
# Evaluate a single ad-hoc question
result = ev.evaluate_single(
    "Which patients have both diabetes and hypertension?",
    reference="The answer should list patient names who have been diagnosed with both type 2 diabetes (SNOMED 44054006) and hypertension.",
    expected_codes=["44054006"],
)
print(result.ragas_scores)
print(result.custom_scores)
