"""Report whether VoxDelta credentials are configured without revealing values."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import cast

from voxdelta.credentials import CredentialProfile, UnsafeEnvFilePermissions, check_credentials


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("local", "comparison"),
        default="local",
        help="credential requirements to check (default: local)",
    )
    profile = cast(CredentialProfile, parser.parse_args(argv).profile)

    try:
        check = check_credentials(profile=profile)
    except UnsafeEnvFilePermissions as error:
        print(str(error), file=sys.stderr)
        return 2

    for key, status in check.statuses.items():
        print(f"{key} {status}")
    return check.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
