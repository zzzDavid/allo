# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refuse a campaign with residual owned reserve files; never sweep them.

SIGKILL/SIGTERM can bypass NamedTemporaryFile cleanup. A filename alone does
not prove that its owner is dead. Preserve the file and require an operator
to check live use before archiving/removing it.
"""

import os
from pathlib import Path
import shutil
import stat
import sys


def main(directory=Path("/tmp")):
    minimum = int(os.environ.get("G1_RESIDUAL_MIN_FREE_BYTES", "0"))
    if minimum < 0:
        raise ValueError("G1_RESIDUAL_MIN_FREE_BYTES must be nonnegative")
    free = shutil.disk_usage(directory).free
    if free < minimum:
        print(f"{directory}: {free} bytes free, policy requires {minimum}")
        return 1
    found = []
    for path in directory.glob("tenon-ledag-reserve-*"):
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if info.st_uid == os.getuid() and stat.S_ISREG(info.st_mode):
            found.append(str(path))
    if found:
        print("Owned session-log reserves remain; check live use before removal:")
        print("\n".join(sorted(found)))
        return 1
    print(f"No owned session-log reserve files in {directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp")))
