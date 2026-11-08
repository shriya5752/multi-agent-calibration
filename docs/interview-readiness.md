# Interview readiness review

## Assessment

A promising ML/AI evaluation project with a focused question and inspectable experiment loop. Currently a small pilot, rather than a validated research conclusion or production system. Stronger for applied ML / evaluation interviews than for demonstrating distributed systems engineering. The differentiator is measuring reliability and belief changes; adding more agent frameworks alone would add little evidence.

## Highest-value additions, in order

1. Add compute-matched controls: independent ensemble, no-peer self-revision, and debate. Reuse identical initial responses between arms and match generation budgets. Report aggregate accuracy at every round, cost, latency and accuracy per dollar. This answers whether gains come from extra inference rather than communication.
2. Run a documented benchmark sample (for example an appropriately licensed MCQ benchmark), with fixed sample IDs, answer-key validation, source attribution and multiple repeated runs. Aim for hundreds of questions subject to budget; choose size from the smallest effect you want to detect. Resample questions as clusters and report paired effects with uncertainty.
3. Add confidence-hidden and neutral-persona ablations. Randomize peer ordering, test explicit own-answer history, and distinguish correct-to-wrong transitions from mere agreement. Confidence-shaped personas confound interpretation of calibration.
4. Add resumable checkpoints, explicit request timeouts/retries, per-call tokens/latency, provider response metadata and invalid-response counts. Preserve failed attempts before retrying. Current fail-fast behavior protects metrics but cannot resume a partial run.
5. Create a short recorded demo: audit stored results offline, inspect one helpful flip and one harmful flip, then show baseline and uncertainty tables. A small read-only dashboard is optional after the evaluation is sound.

## Suggested interview explanation

'I built a controlled testbed to examine whether multi-agent debate improves reliability, measuring accuracy, calibration and answer transitions. In a 30-question pilot, individual accuracy fell 3.33 percentage points, but the question-level uncertainty interval crossed zero. I found that vote aggregation and debate need separate baselines, and that malformed responses and tie policies can bias metrics. I added explicit validation, reproducible mock runs, run manifests, offline reanalysis and regression tests. The next experiment compares debate with a compute-matched no-peer control.'

Use only claims you can explain and reproduce. Do not claim statistical significance, production readiness, independent models, or a proven causal social-pressure effect. No new live-model benchmark was run during this review.

## Refinement scope

Reviewed source, question data, notebook, saved CSVs and documentation at upstream commit ee0b2e8f6ed7c2bdb304eb0f3392dc9e01a2c1a0. Changes are prepared on a GitHub branch for review. Historical CSVs and headline metrics are preserved. Plot files in this delivery are regenerated from historical logs. New code changes parsing and tie policy, so future results should not be silently compared under the old policy.
