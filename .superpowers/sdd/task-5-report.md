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
  deliberately bounded to a speaker's first opening turn. A conservative heuristic
  recognizes explicit `저는 ...입니다` / `제 이름은 ...입니다` forms and greeted
  introductions with a 2–4 Hangul final name-shaped token. Optional preceding text in
  greeted forms must include a recognized service/title marker. Status and policy
  fragments and morphology override the name shape. This is proposal evidence, not
  proof of speaker identity or role.
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

## Review Follow-up: Opening Introductions and Unique Transition Evidence

The Task 5 review identified two contract gaps in commit `f7979f4`:

- The opening self-introduction matcher required `저는` or `제 이름은`, so ordinary
  first-turn forms such as `안녕하세요, 김민수입니다.` were not scored.
- `build_transitions` rejected duplicate emotion IDs but did not reject duplicate
  utterance IDs, allowing ambiguous customer, agent, or unknown-turn evidence.

### Follow-up RED

Regression tests were added before either production change:

```text
cd backend && uv run pytest tests/analysis/test_roles.py \
  tests/analysis/test_transitions.py -v \
  -k 'common_greeted_name_introductions or policy_or_status_sentences or duplicate_utterance_ids'
9 selected: 5 failed, 4 passed
```

Both greeted-name introductions returned `None`, and duplicate customer, agent, and
unknown utterance IDs all failed to raise. The four negative policy/status examples
already passed, confirming that the required broadening could remain narrow.

### Follow-up GREEN

- The first follow-up required an opening greeting plus a three-syllable Hangul name
  immediately before `입니다`. Explicit-marker forms remained supported. The later
  re-review below supersedes that fixed-length proxy with a bounded heuristic.
- Transition construction now validates global utterance-ID uniqueness at function
  entry, before emotion lookup or chronological sorting, regardless of the duplicate
  turns' roles.

```text
cd backend && uv run pytest tests/analysis/test_roles.py -v
13 passed in 0.03s

cd backend && uv run pytest tests/analysis/test_transitions.py -v
20 passed in 0.03s

cd backend && uv run pytest tests/analysis -v
71 passed in 0.05s
```

The follow-up adds positive coverage for `안녕하세요, 김민수입니다.` and
`안녕하십니까? 홍길동입니다`, negative coverage for generic status/policy and
greeted non-name nouns, and duplicate-ID coverage for customer, agent, and unknown
roles.

### Follow-up Verification

```text
cd backend && uv run pytest -v
288 passed in 3.00s

cd backend && uv run ruff check .
All checks passed!

cd backend && uv run ruff format --check .
37 files already formatted

cd backend && uv run mypy src
Success: no issues found in 23 source files

git diff --check
exit code 0
```

## Re-review Follow-up: Bounded Introduction Heuristic

Re-review of commit `36279e2` showed that the fixed three-syllable proxy was not a
defensible final rule: it rejected valid two- and four-syllable name tokens such as
`허준` and `남궁민수`, while accepting three-syllable status morphology such as
`처리중`.

### Heuristic RED

Before changing the matcher, tests added the four reviewer-style examples, separate
2/3/4-syllable name cases, preserved explicit forms, and 2/3/4-syllable non-name and
status cases:

```text
cd backend && uv run pytest tests/analysis/test_roles.py -v \
  -k 'reviewer_opening or two_to_four or preserves_explicit'
27 selected: 9 failed, 18 passed
```

Failures covered two- and four-syllable names, an optional service/title prefix, and
the false-positive statuses `처리중`, `점검중`, `확인중`, and `진행중`.

### Heuristic GREEN

The replacement extracts the final token from an explicit self-introduction or an
opening greeting. It then:

- requires 2–4 Hangul characters;
- permits preceding greeted text only when it contains a recognized service/title
  marker such as `서비스`, `상담사`, `담당자`, or `매니저`;
- rejects status/policy fragments including `처리`, `점검`, `정책`, `규정`, `약관`,
  `정상`, `오류`, `완료`, `예정`, `불가`, and `가능`;
- rejects operational suffixes such as `중`, `완료`, `예정`, `불가`, and `가능`.

This remains a conservative proposal heuristic. It does not identify a person or
confirm a role, and the caller must still complete role confirmation.

```text
cd backend && uv run pytest tests/analysis/test_roles.py -v \
  -k 'reviewer_opening or two_to_four or preserves_explicit'
27 passed, 13 deselected

cd backend && uv run pytest tests/analysis/test_roles.py -v
40 passed in 0.04s

cd backend && uv run pytest tests/analysis -v
98 passed in 0.06s
```

### Heuristic Follow-up Verification

```text
cd backend && uv run pytest -v
315 passed in 3.01s

cd backend && uv run ruff check .
All checks passed!

cd backend && uv run ruff format --check .
37 files already formatted

cd backend && uv run mypy src
Success: no issues found in 23 source files

git diff --check
exit code 0
```
