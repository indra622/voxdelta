from __future__ import annotations

from pathlib import Path

import pytest

from voxdelta_runpod.config import CANONICAL_LABELS, load_experiment_config
from voxdelta_runpod.gates import (
    AggregateReport,
    GateError,
    build_authorized_final_package,
    consume_final_capability,
    decide_final_comparison,
    freeze_candidate,
    full_validation_eligible,
    pilot_eligible,
    publish_frozen_candidate,
    select_pilot_winner,
)

CONFIG = Path(__file__).parents[1] / "config" / "experiment.toml"


def _report(
    *,
    provider: str = "wav2vec-xls-r",
    count: int = 350,
    macro_f1: float = 0.2,
    ece: float = 0.1,
    predicted_classes: int = 7,
    positive_labels: int = 7,
    opened_test_count: int = 0,
    integrity: bool = True,
) -> AggregateReport:
    per_label = {
        label: 0.2 if index < positive_labels else 0.0
        for index, label in enumerate(CANONICAL_LABELS)
    }
    matrix = tuple(
        (count, *([0] * (len(CANONICAL_LABELS) - 1)))
        if index == 0
        else tuple(0 for _ in CANONICAL_LABELS)
        for index in range(len(CANONICAL_LABELS))
    )
    return AggregateReport.model_validate(
        {
            "provider": provider,
            "item_count": count,
            "completed_count": count,
            "macro_f1": macro_f1,
            "per_label_f1": per_label,
            "confusion_matrix": matrix,
            "expected_calibration_error": ece,
            "predicted_class_count": predicted_classes,
            "latency_ms": 1.0,
            "elapsed_seconds": 2.0,
            "peak_cpu_rss_mb": 3.0,
            "peak_cuda_allocated_mb": 4.0 if provider == "wav2vec-xls-r" else None,
            "peak_cuda_reserved_mb": 5.0 if provider == "wav2vec-xls-r" else None,
            "checkpoint_sha256": "a" * 64,
            "report_input_sha256": "b" * 64,
            "finite_training_state": integrity,
            "provider_reload_verified": integrity,
            "provenance_verified": integrity,
            "permissions_private": integrity,
            "privacy_verified": integrity,
            "opened_test_count": opened_test_count,
        }
    )


def test_pilot_gate_uses_strict_thresholds_and_all_integrity_conditions() -> None:
    config = load_experiment_config(CONFIG.resolve())

    assert pilot_eligible(_report(macro_f1=0.1500001), config)
    assert not pilot_eligible(_report(macro_f1=0.15), config)
    assert not pilot_eligible(_report(predicted_classes=5), config)
    assert not pilot_eligible(_report(positive_labels=4), config)
    assert not pilot_eligible(_report(opened_test_count=1), config)
    assert not pilot_eligible(_report(integrity=False), config)


def test_pilot_winner_selection_uses_macro_ece_then_pilot_b() -> None:
    config = load_experiment_config(CONFIG.resolve())

    assert (
        select_pilot_winner(_report(macro_f1=0.30), _report(macro_f1=0.20), config).winner
        == "pilot-a"
    )
    ece = select_pilot_winner(
        _report(macro_f1=0.205, ece=0.05),
        _report(macro_f1=0.20, ece=0.10),
        config,
    )
    assert (ece.winner, ece.reason) == ("pilot-a", "ece")
    tie = select_pilot_winner(
        _report(macro_f1=0.205, ece=0.095),
        _report(macro_f1=0.20, ece=0.10),
        config,
    )
    assert (tie.winner, tie.reason) == ("pilot-b", "pilot-b-tiebreak")
    neither = select_pilot_winner(_report(macro_f1=0.15), _report(macro_f1=0.15), config)
    assert neither.winner is None and neither.reason == "no-eligible-pilot"


def test_full_gate_is_strict_and_candidate_token_is_tamper_evident_one_time(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(CONFIG.resolve())
    passing = _report(count=3_569, macro_f1=0.2400353)

    assert full_validation_eligible(passing, config)
    assert not full_validation_eligible(_report(count=3_569, macro_f1=0.2400352), config)
    candidate = freeze_candidate(
        passing,
        config,
        baseline_checkpoint_sha256="c" * 64,
        baseline_config_sha256="d" * 64,
        metric_schema_sha256="e" * 64,
        decision_rule_sha256="f" * 64,
        holdout_identity_sha256="0" * 64,
    )
    assert candidate.verify_token()
    candidate_path = (tmp_path / "candidate.json").resolve()
    marker = (tmp_path / "final-consumed.json").resolve()
    publish_frozen_candidate(candidate_path, candidate)

    consumed = consume_final_capability(candidate_path, marker, candidate.capability_token)
    assert consumed == candidate
    assert marker.stat().st_mode & 0o077 == 0
    with pytest.raises(GateError, match="^final_capability_consumed$"):
        consume_final_capability(candidate_path, marker, candidate.capability_token)
    with pytest.raises(GateError, match="^invalid_final_capability$"):
        consume_final_capability(
            candidate_path,
            (tmp_path / "other-marker.json").resolve(),
            "9" * 64,
        )


def test_failed_full_gate_never_builds_capability() -> None:
    config = load_experiment_config(CONFIG.resolve())
    with pytest.raises(GateError, match="^full_validation_gate_failed$"):
        freeze_candidate(
            _report(count=3_569, macro_f1=0.2400352),
            config,
            baseline_checkpoint_sha256="c" * 64,
            baseline_config_sha256="d" * 64,
            metric_schema_sha256="e" * 64,
            decision_rule_sha256="f" * 64,
            holdout_identity_sha256="0" * 64,
        )


def test_final_package_cannot_open_manifest_without_private_authorization_marker(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(CONFIG.resolve())
    candidate = freeze_candidate(
        _report(count=3_569, macro_f1=0.3),
        config,
        baseline_checkpoint_sha256="c" * 64,
        baseline_config_sha256="d" * 64,
        metric_schema_sha256="e" * 64,
        decision_rule_sha256="f" * 64,
        holdout_identity_sha256="0" * 64,
    )
    private_manifest = (tmp_path / "must-not-be-opened.jsonl").resolve()

    with pytest.raises(GateError, match="^final_package_not_authorized$"):
        build_authorized_final_package(
            private_manifest,
            private_manifest,
            (tmp_path / "final").resolve(),
            (tmp_path / "missing-authorization-marker").resolve(),
            candidate,
            candidate.capability_token,
            config,
        )
    assert not private_manifest.exists()


def test_final_comparison_requires_exact_dual_completion_and_positive_xls_r_labels() -> None:
    config = load_experiment_config(CONFIG.resolve())
    xls_r = _report(
        count=3_585,
        macro_f1=0.4,
        opened_test_count=3_585,
    )
    baseline = _report(
        provider="emotion2vec-plus",
        count=3_585,
        macro_f1=0.3,
        opened_test_count=3_585,
    )

    decision = decide_final_comparison(xls_r, baseline, config)
    assert decision.promote_xls_r is True
    assert decision.selected_provider == "wav2vec-xls-r"
    failed = decide_final_comparison(
        _report(count=3_585, macro_f1=0.4, positive_labels=6, opened_test_count=3_585),
        baseline,
        config,
    )
    assert failed.promote_xls_r is False
    assert failed.selected_provider == "emotion2vec-plus"
