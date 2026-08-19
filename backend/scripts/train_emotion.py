"""Train one local seven-emotion checkpoint without printing private input details."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Never

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
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=622)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        profile: TrainingProfile
        if arguments.architecture == "wav2vec-xls-r":
            if (
                arguments.base_model != "facebook/wav2vec2-xls-r-300m"
                or arguments.freeze_encoder
                or arguments.seed != 622
            ):
                raise TrainingError("invalid_training_profile")
            profile = Wav2VecTrainingProfile(seed=arguments.seed)
        else:
            if (
                arguments.base_model != "iic/emotion2vec_plus_large"
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
