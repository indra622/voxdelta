"""Report whether VoxDelta credentials are configured without revealing values."""

from __future__ import annotations

import sys

from voxdelta.credentials import UnsafeEnvFilePermissions, check_credentials


def main() -> int:
    try:
        check = check_credentials()
    except UnsafeEnvFilePermissions as error:
        print(str(error), file=sys.stderr)
        return 2

    for key, status in check.statuses.items():
        print(f"{key} {status}")
    return check.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
