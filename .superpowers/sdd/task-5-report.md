# Task 5 Report: Operational States and Emotion-Transition Analysis

## Scope

- Base commit: `88447b5c51b49bc2ce0210eab8577d6e8a7ec417`
- Branch: `feature/core-pipeline`
- Focused commit: `feat: calculate customer emotion transitions`
- No provider, network, credential, model, or key calls were made.

Task 5 adds public operational-state mapping, customer-turn median smoothing, role
proposals, adjacent turn-transition construction, and call-level summaries. The
existing deterministic fake emotion provider now delegates to the public mapper so
Task 4 and Task 5 use one state-mapping implementation.

## Contract Decisions

### Operational state

- The mapper requires exactly the canonical seven labels, finite values in `[0, 1]`,
  normalization within the domain model's `1e-6` tolerance, and an independent finite
  confidence in `[0, 1]`.
- Surprise tied with or above the largest other single emotion yields `uncertain`.
- Confidence below `0.55` yields `uncertain`; exactly `0.55` is accepted.
- Business score ties follow the fixed specification order: satisfied, stable,
  dissatisfied, escalated. A `1e-12` absolute tolerance prevents representation-only
  differences such as `0.3` versus `0.1 + 0.1 + 0.1` from changing that order.

### Smoothing and transitions

- `median_smooth` treats the supplied result list as the caller-established customer
  chronology. It does not sort by ID. Edge windows use two values; interior windows
  use three. It returns deep model copies with raw intensity and provenance preserved.
- Duplicate emotion IDs and invalid raw intensities are rejected before smoothing.
- Transition construction deterministically sorts utterances and evaluates only
  adjacent, confirmed `customer -> agent -> customer` triples. It does not infer roles
  or skip intervening turns.
- Missing smoothed intensity makes a triple ineligible. Raw intensity is never used as
  a fallback. Non-finite or out-of-range smoothed values are rejected.
- Deltas are rounded to four decimals before inclusive `-0.20` / `+0.20` classification.

### Role proposal and summary

- Role suggestion requires exactly two speakers across the entire call, while cue
  inspection is limited to utterances starting before 30 seconds.
- Each literal cue scores at most once per speaker. The self-introduction cue is
  deliberately narrow: only a speaker's first opening turn and an explicit
  `저는 ...입니다` or `제 이름은 ...입니다` ending count.
- Suggestions are proposals only. Input `Utterance.role` values are never changed.
- Summary emotions are aligned by unique customer utterance ID and then ordered by the
  deterministic utterance chronology. Only finite, present smoothed intensities count
  toward coverage and supply the peak and overall delta.
- At least three valid results and coverage of at least `0.50` are required. Empty
  customer input raises `InsufficientEmotionCoverage` without division.
- Equal peaks resolve to the earliest valid customer turn. Recovery and worsening
  counts are calculated directly from the supplied transitions.

## TDD Evidence

### RED

All four broad analysis test modules were created before production modules:

```text
cd backend && uv run pytest tests/analysis -v
collected 0 items / 4 errors
ModuleNotFoundError: No module named 'voxdelta.analysis.emotions'
ModuleNotFoundError: No module named 'voxdelta.analysis.roles'
ModuleNotFoundError: No module named 'voxdelta.analysis.summary'
ModuleNotFoundError: No module named 'voxdelta.analysis.transitions'
```

The first implementation run also caught a true grouped-tie defect:

```text
62 collected; 1 failed, 61 passed
expected stable, got dissatisfied
```

This was caused by the binary-float difference between `0.3` and three summed `0.1`
values. The mapper was corrected to retain specification order for numerically equal
scores.

### GREEN

```text
cd backend && uv run pytest tests/analysis -v
62 passed in 0.06s
```

The 62 focused tests cover validation, confidence and threshold boundaries, surprise
ties, grouped-score ties, smoothing copies/order/windows, duplicate and non-finite
data, deterministic timeline sorting, adjacency and role confirmation, missing
smoothed values, distinct role cues, the 30-second boundary, speaker cardinality,
self-introduction narrowing, summary alignment, exact 50% coverage, minimum count,
zero customers, deterministic peak ties, and transition counts.

## Verification

```text
cd backend && uv run pytest -v
279 passed in 3.02s

cd backend && uv run ruff check .
All checks passed!

cd backend && uv run ruff format --check .
37 files already formatted

cd backend && uv run mypy src
Success: no issues found in 23 source files

git diff --check
exit code 0
```
