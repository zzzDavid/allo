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

# Round 25 (2026-11-04, release gate G10) corrected the decode region's RoPE
# kernels to cover the D2-element vector the host scatters (the typed-route
# program iterated D elements).  Round 71 (2026-11-05) realizes a replicated
# elementwise grid once over the device's 32 channels instead of once per
# contraction replica (`allo/spmw_aim.py`, ISSUES X-32): its operands and
# result are written once and the EWMUL / EWADD op sizes are the single
# copy's.  Round 73 (2026-11-05) gives every elementwise kernel group a bank
# row region of its own and copies a value into each reader's region
# (physical liveness, X-32: no scatter or launch of one group can overwrite a
# value another group still has to read); the regions ahead of the K/V
# caches and the weights are the round 25 ones again (attention_l* and
# ffn_fc_l128 reproduce their typed-route pins byte for byte), and the rows
# with an RMSNorm or FFN activation chain carry the copies into the next
# group's region.  Every pin below is the matcher path's trace at the
# revision that introduced the change (`allo.compile(case.region,
# build_aim_target(), aim_cost, host_moves=case.host_program,
# reduction_order="reassociate")`), simulated natively with ramulator2
# (`docs/release/paper_reproduction/inventory.csv`).
CORRECTED_TRACE_PROVENANCE = (
    "matcher path @ tenon round 73, 2026-11-05 (one bank-row region per elementwise group, X-32); "
    "the round 71 pins are superseded for every case whose rows moved"
)

CORRECTED_TRACE_SHA256 = MappingProxyType(
    {
        "attention_l128": "29f3190aad6608c8ee3d4fd890344a9020ef51d72b8d016eab46e9c3462142ae",
        "attention_l4096": "103edc8376c5e4f8035a68182b0b04ab7cf0c2f6171041005968d3f14a5964b1",
        "attention_l512": "112c7bad9914c09d5ac0bfa684ef9ce6f0431f2b76ef5250f3ac86588dc35fa7",
        "ffn_activation_l128": "7fca6f33363da25e333ba79da9cda881d170da0a619b8a5a9e4dd591e0da77e3",
        "ffn_complete_l128": "0c715e6896ba24803ae8aa5b308c40bad999d679d093b0b9f250eacb434ad6ca",
        "ffn_fc_l128": "833f14bacd24b628679d78bde7097cf0607c7eb65e4ba71998fce2a4241e4777",
        "full_block_l128": "2d1d9f80a00feb6a82573af2e315ccce761783afa01808013219bb4344efe529",
        "full_block_l4096": "79b1a3f60e4ad7fd32825e86c7ed99dbadddbdc259562ff4baa69fecb9938d7d",
        "full_block_l512": "c6243af8f755559b33d6b52f15fcb7e90e0d232f0690553c86802e1db9047f64",
        "norm_residual_l128": "06634de4e228c88de0c540ccf49f1b8e2b2679ce8b0c107323083d8fdf27b08c",
        "qkvo_rope_l128": "2145cdc689bbf2bb09eb4271ac305800bdab412e65bbc6b9449827c0d98d4ae9",
        "softmax_pim_l128": "2fc1c5d5412d7b8d80d305a1a5a6cfbd6703809f9c415969338d0d98bf4354d7",
        "softmax_pim_l4096": "4e19e5f0de62c11c07fe4007b8eaacc475f81c5535ce8483289def6930b8afee",
        "softmax_pim_l512": "007795b1e905d15754f5f19b2cbb5e661de8ba9910c935bdc4b9f2ebb159792b",
    }
)


def reference_trace_sha256(case_id: str) -> str:
    """The pinned trace hash the matcher path must reproduce for ``case_id``:
    the corrected pin where the benchmark program was corrected, else the
    typed-route pin."""
    return CORRECTED_TRACE_SHA256.get(case_id, TYPED_TRACE_SHA256[case_id])


__all__ = [
    "CORRECTED_TRACE_PROVENANCE",
    "CORRECTED_TRACE_SHA256",
    "TYPED_TRACE_PROVENANCE",
    "TYPED_TRACE_SHA256",
    "TYPED_TRACE_LINES",
    "reference_trace_sha256",
]
