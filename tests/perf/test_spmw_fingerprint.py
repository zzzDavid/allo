# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shared canonical-JSON digest helper equals every inline copy it replaced."""

import hashlib
import json
import math

import pytest

from allo.spmw_fingerprint import bytes_digest, canonical_json, file_digest, json_digest

PAYLOADS = (
    {},
    [],
    "text",
    0,
    -17,
    1.5,
    {"b": [1, 2.25, {"z": "y", "a": None}], "a": True, "nested": {"k": [False, -0.5]}},
    [[1, [2, [3, {"deep": "value"}]]], "x", 3.0e-12],
    {"unicode": "café λ", "escaped": "quote\" backslash\\"},
)


def _strict_inline(value):
    # upmem_physical_search, apu_g2_runtime, apu_v1_vector_runtime,
    # apu_g2_composed_contraction
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _matcher_inline(value):
    # spmw_autoschedule._matcher_digest and spmw_plan buffer-metric fingerprint
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _cost_inline(value):
    # perf/cost._canonical_json, then .encode()
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        .encode()
    ).hexdigest()


def _recipe_inline(value):
    # apu_g2_recipe._structural_digest and upmem_calibration report fingerprint
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
    ).hexdigest()


def test_json_digest_matches_every_replaced_inline_digest():
    for payload in PAYLOADS:
        assert json_digest(payload, allow_nan=False) == _strict_inline(payload)
        assert json_digest(payload) == _matcher_inline(payload)
        assert json_digest(payload) == _cost_inline(payload)
        assert json_digest(payload) == _recipe_inline(payload)


def test_canonical_json_matches_replaced_text():
    for payload in PAYLOADS:
        assert canonical_json(payload) == json.dumps(
            payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
        assert canonical_json(payload, allow_nan=False) == json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )


def test_nan_payload_is_hashable_by_default_and_rejected_when_strict():
    payload = {"value": math.nan, "inf": [math.inf]}
    assert json_digest(payload) == _matcher_inline(payload)
    with pytest.raises(ValueError):
        json_digest(payload, allow_nan=False)
    with pytest.raises(ValueError):
        canonical_json(payload, allow_nan=False)


def test_bytes_and_file_digest_match_hashlib(tmp_path):
    data = bytes(range(256)) * 9000
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert bytes_digest(data) == expected
    assert file_digest(path) == expected
