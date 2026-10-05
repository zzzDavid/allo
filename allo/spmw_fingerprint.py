"""Canonical-JSON content digests shared by the SPMW/PIM compiler.

These are identities and cache keys: they are recorded or compared as
dictionary keys, and no compile path refuses to proceed because of a value.
"""

from __future__ import annotations

import hashlib
import json
import os


def canonical_json(value, *, allow_nan: bool = True) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=allow_nan,
    )


def json_digest(value, *, allow_nan: bool = True) -> str:
    return hashlib.sha256(
        canonical_json(value, allow_nan=allow_nan).encode("ascii")
    ).hexdigest()


def bytes_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["bytes_digest", "canonical_json", "file_digest", "json_digest"]
