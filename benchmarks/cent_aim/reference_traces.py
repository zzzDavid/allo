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
# contraction replica (`allo/spmw_aim.py`, ISSUES X-19): its operands and
# result are written once, the EWMUL/EWADD op sizes are the single copy's,
# and the rows the smaller footprints free shift the matrix rows allocated
# after them (attention_l*: the K/V cache rows move from 11 to 8, every
# command otherwise identical, cycles unchanged).  Every case's pin below is
# the matcher path's trace at the revision that introduced the change
# (`allo.compile(case.region, build_aim_target(), aim_cost,
# host_moves=case.host_program, reduction_order="reassociate")`), simulated
# natively with ramulator2 (`docs/release/paper_reproduction/inventory.csv`).
# The four cases whose cycles are unchanged (`attention_l*`, `ffn_fc_l128`)
# differ from their earlier pins only by that shift: adding 3 to the row
# argument of every W MEM, WR_ABK and MAC_ABK line of the new trace
# reproduces the earlier pin byte for byte.
CORRECTED_TRACE_PROVENANCE = (
    "matcher path @ tenon round 71, 2026-11-05 (replicated elementwise grids realized once over the device, X-19); "
    "the round 25 RoPE D2 pins and the typed-route pins are superseded for every case whose default program changed"
)

CORRECTED_TRACE_SHA256 = MappingProxyType(
    {
        "attention_l128": "b26e089bfbf3697bb214cb486a80bbfea96298263ec98f289de3238c643d2272",
        "attention_l4096": "37cd5413e640ce099fd6cbec0fcda4a6a23354b1dae6c3690fc41532905c5b5b",
        "attention_l512": "13ede34e870d7b3d4345a3210c3ff4e0c9c93a82ca0eb30c62d9e76496f433b0",
        "ffn_activation_l128": "991d9321f7beb7f767f5c7dbb874ce8ffd8329937be4518d1dcbebbb6ace051d",
        "ffn_complete_l128": "2f1a131c2eba30ac320217f1dc123cf87aa3db529e4060a838157c2a774db706",
        "ffn_fc_l128": "db2789ab08e3b37eabb62af7e00bdd3744546b8dba49977bcf496fee918c139e",
        "full_block_l128": "2d6d1c49f013443ec0cabc5bcd50d1a512e7bd59e664bec9f2af549b509798d8",
        "full_block_l4096": "27ed3b547c6e96d4f8109f39f7fd0dea26bb54e376adf1613d09ef59fb42781f",
        "full_block_l512": "32dcab49ccce7056f5b669bad734c118defadebd0192451ee4221d02a9a4f211",
        "norm_residual_l128": "2e728babe7c8d757d58ff7cd8fc5d92a1ff21d33ee741e9268cbaeea5e547805",
        "qkvo_rope_l128": "fb745711ff46789775de9f95c9a170bca70d38407a3329716b3c0b5ce04051a8",
        "softmax_pim_l128": "f94905736375a9351dc6632f89c109b2329f8dd21d02950f91e61fb0645fef43",
        "softmax_pim_l4096": "76c5302bf97c3a687303c283e6b9790994e02bfd350a3ad72a218fcb0c8ce16f",
        "softmax_pim_l512": "ba77059ade1efa172e530b7fb02121b6c47f2aa84508b4786b6043771a05b3d0",
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
