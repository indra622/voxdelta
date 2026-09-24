"""Create one reference-aligned local XLS-R emotion overlay candidate.

The script emits only safe counts and a relative artifact name; it never prints speech.
"""

from __future__ import annotations

import argparse

from voxdelta.annotation.emotion_candidates import generate_local_candidate
from voxdelta.api.dependencies import build_dependencies
from voxdelta.config import load_settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("conversation_id")
    args = parser.parse_args()
    settings = load_settings()
    dependencies = build_dependencies(settings)
    path, summary = generate_local_candidate(
        annotation_root=dependencies.annotation_root,
        audio_root=dependencies.annotation_audio_root,
        conversation_id=args.conversation_id,
        provider=dependencies.runner.emotion_provider,
    )
    print(
        {
            "artifact": path.name,
            "turn_count": summary["turn_count"],
            "emotion_histogram": summary["emotion_histogram"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
