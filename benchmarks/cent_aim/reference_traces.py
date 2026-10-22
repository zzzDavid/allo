# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SHA-256 of the typed-route AiM trace for each CENT case (data, not code).

Generator: ``CompiledAimProgram`` (``allo.compile(build_case(case_id),
build_aim_target())``, ``manifest["trace"]["sha256"]``) at ``tenon@ebb90a6``,
default ``DecodeSpec`` and default AiM target geometry. Generated 2026-10-05
before task 019 deleted the typed route; the matcher path must reproduce
each trace byte for byte.

Command::

    python -c "import allo; from allo.pim.targets import build_aim_target; \
from benchmarks.cent_aim.workloads import build_case; \
print(allo.compile(build_case('<case_id>'), build_aim_target()).manifest['trace']['sha256'])"
"""

from types import MappingProxyType

TYPED_TRACE_PROVENANCE = "CompiledAimProgram @ tenon@ebb90a6, 2026-10-05"

TYPED_TRACE_SHA256 = MappingProxyType(
    {
        "norm_residual_l128": "f846600a17b0e091a682e49d6e5cd8a663b83a16750c1bf33fef5ddd43872523",
        "qkvo_rope_l128": "fd645370fa04c1fa0750e9b7d8de025b11573a99aa611abcb3daf56db93159cb",
        "attention_l128": "29f3190aad6608c8ee3d4fd890344a9020ef51d72b8d016eab46e9c3462142ae",
        "attention_l512": "112c7bad9914c09d5ac0bfa684ef9ce6f0431f2b76ef5250f3ac86588dc35fa7",
        "attention_l4096": "103edc8376c5e4f8035a68182b0b04ab7cf0c2f6171041005968d3f14a5964b1",
        "softmax_pim_l128": "5bbf4172209a12337e05aef4fe494c0c60dbdd1be7cc6d71a13d9c2e0bad9c5a",
        "softmax_pim_l512": "a6f1c35d91e2071c551b06c293078a7c14af77b3ff5fe8cdea32ef8a91bb16b2",
        "softmax_pim_l4096": "f40acc441db399336a907ad4fced33fff232a8f6dba37ccb5e0fdb5fb05e42a9",
        "ffn_fc_l128": "833f14bacd24b628679d78bde7097cf0607c7eb65e4ba71998fce2a4241e4777",
        "ffn_activation_l128": "e30bbfbe179cce55d33cefa7fe0a5c7db88c5519e403f6661c58defdd9b25cea",
        "ffn_complete_l128": "73af1bad404b66e1604e9af0f261dff5a899698954765e50a6be0929915a33ce",
        "full_block_l128": "9c78eeb0b8fa1733f779257582399599fc0808718b98a549a96e9793838a17d6",
        "full_block_l512": "4d6a2a027bc13c3791df21d68bcefe44a6c0ce6d0a15dffcc3791895142ae732",
        "full_block_l4096": "1aa92ccf6f1db5a80204f91d8c91c86e86a81641aecfee1e9106def558fa0589",
    }
)

# Total trace lines including the trailing EOC, for diagnostics.
TYPED_TRACE_LINES = MappingProxyType(
    {
        "norm_residual_l128": 10271,
        "qkvo_rope_l128": 9747,
        "attention_l128": 2249,
        "attention_l512": 2549,
        "attention_l4096": 5649,
        "softmax_pim_l128": 6149,
        "softmax_pim_l512": 24581,
        "softmax_pim_l4096": 196619,
        "ffn_fc_l128": 3312,
        "ffn_activation_l128": 11276,
        "ffn_complete_l128": 14587,
        "full_block_l128": 42999,
        "full_block_l512": 61731,
        "full_block_l4096": 236869,
    }
)

__all__ = ["TYPED_TRACE_PROVENANCE", "TYPED_TRACE_SHA256", "TYPED_TRACE_LINES"]
