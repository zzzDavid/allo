# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Budget actual build/evidence, extraction and profile-copy filesystems.

The per-launch reserve covers all retained raw/normalized reads plus build
scratch; charge it once per distinct filesystem (conservatively in full on
each if split). The installed runtime separately checks the fixed /tmp
session log and transient per-read requirements immediately before reading.
"""

import os
from pathlib import Path
import shutil
import sys


def main():
    launches, per_launch = map(int, sys.argv[1:3])
    if launches < 1 or per_launch < 1 or len(sys.argv) != 6:
        raise ValueError(
            "expected positive launches, bytes per launch and three storage paths"
        )
    needed = launches * per_launch
    seen = set()
    for arg in sys.argv[3:]:
        directory = Path(arg).resolve(strict=True)
        if not directory.is_dir() or not os.access(directory, os.W_OK):
            raise ValueError(f"storage is not a writable directory: {directory}")
        device = directory.stat().st_dev
        print(f"actual storage: {directory}, filesystem {device}")
        if device in seen:
            continue
        seen.add(device)
        free = shutil.disk_usage(directory).free
        print(
            f"planned launches {launches} x {per_launch} bytes = {needed} needed; {free} available in {directory}"
        )
        if free < needed:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
