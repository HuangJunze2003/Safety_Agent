# Graduation Project Optimization TODO

## Goal
Address four common thesis deduction risks:
- Lack of quantitative metrics
- Insufficient baseline comparison
- Missing failure case analysis
- Vague engineering details

## Task Plan

- [x] Build a fixed evaluation set (50-100 samples), and define gold fields: hazard point, legal clause, case ID.
- [x] Implement an offline evaluation script and export CSV metrics: Recall@1/3/5, P50/P95 latency, and manual scoring fields.
- [x] Complete comparison experiments: no-RAG baseline, legal-only RAG, case-only RAG, dual-retrieval + Agent; include at least one ablation.
- [x] Build a failure case library (>=10 cases), classified by retrieval failure / visual misjudgment / hallucination / pipeline failure, with root cause and fix.
- [x] Add engineering documentation diagrams: system architecture, core sequence flow, and deployment topology.
- [x] Document key APIs (hazard analysis / legal retrieval / case retrieval): params, response schema, error codes, and JSON examples.
- [x] Summarize experiment results and write thesis Chapter 5: metric definitions, comparison, ablation, and failure analysis.
- [x] Improve defense materials: experiment highlights, failure cases, engineering implementation, and demo script.

## Suggested Milestones (2 Weeks)
<!--  -->
- Day 1-2: evaluation set and annotation rules
- Day 3-4: offline evaluator and first metric run
- Day 5-6: baselines + ablation
- Day 7: failure case library
- Day 8-9: diagrams + API docs
- Day 10-11: Chapter 5 writing
- Day 12: defense slides
- Day 13-14: rehearsal and final polish

## Execution Tracker

| ID | Task | Owner | Due | Deliverable | Done Criteria | Status |
| --- | --- | --- | --- | --- | --- | --- |
| T1 | Build fixed evaluation set (50-100) with gold labels | Me | Day 2 | `data/eval/eval_set.csv` and `data/eval/label_guideline.md` | >=50 valid samples; each sample includes hazard point, legal clause, case ID | Done |
| T2 | Implement offline evaluator and export metrics | Me | Day 4 | `scripts/evaluate_pipeline.py` and `results/eval_metrics_round1.csv` | Script runs end-to-end and outputs Recall@1/3/5, P50/P95, manual scoring columns | Done |
| T3 | Run baseline and ablation experiments | Me | Day 6 | `results/exp_comparison.csv` and `results/ablation.csv` | Includes 4 setups (no-RAG, legal-only, case-only, dual+Agent) + >=1 ablation | Done |
| T4 | Build failure-case library (>=10) | Me | Day 7 | `docs/experiments/failure_cases.md` | >=10 cases with expected vs actual, type, root cause, and fix idea | Done |
| T5 | Add architecture/sequence/deployment diagrams | Me | Day 9 | `docs/thesis/figures/architecture.png`, `docs/thesis/figures/sequence.png`, `docs/thesis/figures/deployment.png` | Three diagrams can fully explain data flow and runtime components | Done |
| T6 | Document key APIs with examples | Me | Day 9 | `docs/engineering/api_spec.md` | Covers 3 core APIs with params, response schema, error codes, JSON examples | Done |
| T7 | Write thesis Chapter 5 (experiments) | Me | Day 11 | `docs/thesis/chapter5_experiments.md` | Includes metric definitions, comparison tables, ablation, and failure analysis | Done |
| T8 | Finalize defense deck and demo script | Me | Day 12 | `docs/defense/defense_outline.md` and `docs/demo/demo_script.md` | Slides include metrics, baselines, failure analysis, and clear demo path | Done |

## Weekly Checkpoint

- End of Week 1 (Day 7): T1-T4 completed, all experiment data ready.
- End of Week 2 (Day 14): T5-T8 completed, thesis and defense package finalized.
