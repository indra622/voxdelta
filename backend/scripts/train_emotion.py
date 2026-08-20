"""Train one local seven-emotion checkpoint without printing private input details."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Never, cast

from voxdelta.evaluation.emotion_training import (
    Emotion2VecTrainingProfile,
    TrainingError,
    TrainingProfile,
    Wav2VecTrainingProfile,
    default_training_backend,
    train_from_manifest,
)


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise TrainingError("invalid_training_profile")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description="Train a local seven-emotion provider")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--architecture",
        choices=("wav2vec-xls-r", "emotion2vec-plus"),
        default="wav2vec-xls-r",
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-model-path", type=Path)
    parser.add_argument("--micro-batch-size", type=int, choices=(1, 2, 4, 8))
    parser.add_argument(
        "--adaptation-strategy",
        choices=("full", "partial-last4"),
        default="full",
    )
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=622)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        profile: TrainingProfile
        if arguments.architecture == "emotion2vec-plus" and arguments.adaptation_strategy != "full":
            raise TrainingError("invalid_training_profile")
        if arguments.architecture == "wav2vec-xls-r":
            if (
                arguments.base_model != "facebook/wav2vec2-xls-r-300m"
                or arguments.base_model_path is None
                or arguments.micro_batch_size is None
                or (
                    arguments.adaptation_strategy == "partial-last4"
                    and arguments.micro_batch_size != 2
                )
                or arguments.freeze_encoder
                or arguments.seed != 622
            ):
                raise TrainingError("invalid_training_profile")
            accumulation_by_batch = {8: 2, 4: 4, 2: 8, 1: 16}
            profile = Wav2VecTrainingProfile(
                adaptation_strategy=arguments.adaptation_strategy,
                seed=arguments.seed,
                base_model_path=arguments.base_model_path,
                train_batch_size=arguments.micro_batch_size,
                eval_batch_size=arguments.micro_batch_size,
                gradient_accumulation_steps=cast(
                    Literal[2, 4, 8, 16],
                    accumulation_by_batch[arguments.micro_batch_size],
                ),
            )
        else:
            if (
                arguments.base_model != "iic/emotion2vec_plus_large"
                or arguments.base_model_path is not None
                or arguments.micro_batch_size is not None
                or not arguments.freeze_encoder
                or arguments.seed != 622
            ):
                raise TrainingError("invalid_training_profile")
            profile = Emotion2VecTrainingProfile(seed=arguments.seed, freeze_encoder=True)
        train_from_manifest(
            arguments.manifest,
            arguments.output,
            profile=profile,
            backend=default_training_backend,
        )
    except TrainingError as error:
        print(f"training_error: {error.code}", file=sys.stderr)
        return 2
    except Exception:
        print("training_error: training_failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
