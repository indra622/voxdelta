"""Prove an immutable XLS-R release bundle still infers offline from itself alone."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from voxdelta.providers._emotion_runtime import Device

from voxdelta_runpod.smoke import SmokeError, run_release_smoke


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "auto", "cuda", "mps"), default="cpu")
    arguments = parser.parse_args(argv)
    try:
        result = run_release_smoke(
            arguments.bundle.absolute(),
            device=cast(Device, arguments.device),
        )
    except SmokeError as error:
        print(error.code)
        return 2
    except Exception:
        print("release_smoke_error")
        return 2
    print("release_smoke_ok")
    print(f"release_id {result.release_id}")
    print(f"candidate_checkpoint_sha256 {result.candidate_checkpoint_sha256}")
    print(f"device {result.device}")
    print(f"top_label {result.top_label}")
    print(f"confidence {result.confidence:.6f}")
    print(f"probability_sum {result.probability_sum:.9f}")
    print(f"negative_intensity {result.negative_intensity:.6f}")
    print(f"latency_ms {result.latency_ms:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
