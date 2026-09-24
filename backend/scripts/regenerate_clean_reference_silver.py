"""Publish one complete KCSC-reference Silver replacement without exposing transcripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from voxdelta.annotation.clean_reference_silver import (
    CleanReferenceSilverError,
    regenerate_from_reference,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--conversation-id", required=True)
    args = parser.parse_args()
    try:
        receipt = regenerate_from_reference(
            args.annotation_root,
            args.reference_root,
            conversation_id=args.conversation_id,
        )
    except CleanReferenceSilverError as error:
        print(json.dumps({"status": "rejected", "reason": str(error)}))
        return 2
    print(
        json.dumps(
            {
                "status": "published",
                "conversation_id": receipt.conversation_id,
                "turn_count": receipt.turn_count,
                "archived_content_sha256": receipt.archived_content_sha256,
                "content_sha256": receipt.content_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
