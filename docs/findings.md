# Pilot findings (audited from saved logs)

The 180 saved rows reproduce the original metrics: individual accuracy fell from 75.56% to 72.22%, ECE rose from 0.1489 to 0.1672, and 20 of 90 revision opportunities changed answers. Five flips helped, eight hurt and seven changed one wrong answer to another.

Confidence decreased for every persona: skeptic 85.07 -> 79.50, confident solver 96.33 -> 95.50, advocate 89.93 -> 86.50. The original suggestion that debate inflated average confidence is contradicted by these logs. Worse calibration can coexist with lower confidence when accuracy also changes.

Under explicit tie abstention (counted incorrect), majority accuracy moves 76.67% -> 80.00% and weighted voting 80.00% -> 83.33%. The final weighted-versus-majority difference is one question. Aggregation gains cannot be attributed exclusively to independent sampling or debate from these comparisons alone. A matched no-peer revision control is needed.

The paired question bootstrap interval for the individual accuracy change is [-8.89, +2.22] percentage points (5,000 samples, seed 42). All three agents within a question stay together in resampling. This is exploratory uncertainty on this sample, not a substitute for repeated runs or a representative benchmark.

The historical documentation reports Groq / llama-3.1-8b-instant and a run date of 2026-08-09. No historical manifest or raw provider response archive is present, so that attribution is not independently verifiable. The saved results are treated as a pilot artifact, not newly generated evidence.

Run `python -m src.audit_results` for the numeric audit. Plots can be regenerated from saved logs; no model calls are needed.
