# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen physical-work contracts extracted from the CENT evidence bundle.

The values below are data, not lowering rules.  They come from the first
``generator_run`` in each archived ``trace-metadata.json`` under
``experiments/results/cent-aim-baseline-2026-07-22/cases``.  The independent
second generation has the same SHA-256 and opcode inventory in every case.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


CENT_SOURCE_COMMIT = "3b0f874aa2d0501b85e69164c5112106a40a941c"


@dataclass(frozen=True)
class VendorTraceContract:
    """Identity and physical opcode inventory of one archived CENT trace."""

    trace_sha256: str
    opcode_counts: Mapping[str, int]

    def __post_init__(self):
        if len(self.trace_sha256) != 64:
            raise ValueError("trace_sha256 must contain 64 hexadecimal digits")
        object.__setattr__(
            self,
            "opcode_counts",
            MappingProxyType(dict(self.opcode_counts)),
        )


def _contract(trace_sha256: str, **opcode_counts: int) -> VendorTraceContract:
    return VendorTraceContract(trace_sha256, opcode_counts)


VENDOR_TRACE_CONTRACTS = MappingProxyType(
    {
        "norm_residual_l128": _contract(
            "c0faa3a34b85f54330142c87fc45647369acfd7d6f91f1ed9e1b26b8d8d590e8",
            AiM_COPY_BKGB=8,
            AiM_COPY_GBBK=8,
            AiM_EOC=1,
            AiM_EWADD=2,
            AiM_EWMUL=4,
            AiM_MAC_ABK=2,
            AiM_RD_MAC=2,
            AiM_SYNC=2,
            AiM_WR_BIAS=2,
            R_MEM=2048,
            W_MEM=8192,
        ),
        "qkvo_rope_l128": _contract(
            "9496f340492b8bced6a1c351254bcc5018f5c2fe50455f95169d51904eebdb42",
            AiM_EOC=1,
            AiM_EWMUL=2,
            AiM_MAC_ABK=512,
            AiM_RD_MAC=512,
            AiM_WR_BIAS=512,
            AiM_WR_GB=16,
            W_MEM=8192,
        ),
        "attention_l128": _contract(
            "0376b4f769ab76e803140ef3f417acff0843340862d99977949a915717350cc4",
            AiM_EOC=1,
            AiM_MAC_ABK=64,
            AiM_RD_MAC=64,
            AiM_WR_ABK=256,
            AiM_WR_BIAS=64,
            AiM_WR_GB=8,
            W_MEM=1024,
        ),
        "attention_l512": _contract(
            "eb77b05cfa5974954c6916801a81b9d0d9977a4f3d69077b3d2af22f22b9dee6",
            AiM_EOC=1,
            AiM_MAC_ABK=160,
            AiM_RD_MAC=160,
            AiM_WR_ABK=256,
            AiM_WR_BIAS=160,
            AiM_WR_GB=20,
            W_MEM=1024,
        ),
        "attention_l4096": _contract(
            "50614e90f8aa7181542c3f515b706662ffe58519aec8fc42ca5674da2da9d146",
            AiM_EOC=1,
            AiM_MAC_ABK=1152,
            AiM_RD_MAC=1152,
            AiM_WR_ABK=256,
            AiM_WR_BIAS=1152,
            AiM_WR_GB=144,
            W_MEM=1024,
        ),
        "softmax_pim_l128": _contract(
            "d851760dea768f42a1bdcbc0adeee0ca2209a056df2f37c0ca1e21e565f9888a",
            AiM_EOC=1,
            AiM_EWMUL=2,
            AiM_SYNC=2,
            R_MEM=2048,
            W_MEM=4096,
        ),
        "softmax_pim_l512": _contract(
            "dfac8c1ae10050ea5bff07ea9610966ecfecd3c6ea4ec28a951ef83d0e6b593a",
            AiM_EOC=1,
            AiM_EWMUL=2,
            AiM_SYNC=2,
            R_MEM=8192,
            W_MEM=16384,
        ),
        "softmax_pim_l4096": _contract(
            "541dda508ee3b4c49c3791f2e0cfbdcb58ae52a3eb34e5d209863901d9b6aaa9",
            AiM_EOC=1,
            AiM_EWMUL=8,
            AiM_SYNC=2,
            R_MEM=65536,
            W_MEM=131072,
        ),
        "ffn_fc_l128": _contract(
            "01511216cbb1e2bcf6d29c5cd6d01a6aed2240881aaf2a73f16d684a91e6106d",
            AiM_AF=86,
            AiM_EOC=1,
            AiM_MAC_ABK=1040,
            AiM_RD_AF=86,
            AiM_RD_MAC=1040,
            AiM_WR_BIAS=1040,
            AiM_WR_GB=19,
        ),
        "ffn_activation_l128": _contract(
            "8a00aa90b848142e980471cbd9f3eaaa7898f11a77bd95d31cb399d32ab48c54",
            AiM_COPY_BKGB=4,
            AiM_COPY_GBBK=4,
            AiM_EOC=1,
            AiM_EWMUL=2,
            AiM_SYNC=1,
            R_MEM=2816,
            W_MEM=8448,
        ),
        "ffn_complete_l128": _contract(
            "4ac5a890f5cc6206732415e05d1ea2d7c2fcdf630d3a60083b584239c1843d1c",
            AiM_AF=86,
            AiM_COPY_BKGB=4,
            AiM_COPY_GBBK=4,
            AiM_EOC=1,
            AiM_EWMUL=2,
            AiM_MAC_ABK=1040,
            AiM_RD_AF=86,
            AiM_RD_MAC=1040,
            AiM_SYNC=1,
            AiM_WR_BIAS=1040,
            AiM_WR_GB=19,
            R_MEM=2816,
            W_MEM=8448,
        ),
        "full_block_l128": _contract(
            "3f22f4d6ee6cd6d3dd8aceab69ad469ebcf5111c82da854611485ccc1b4a6fe7",
            AiM_AF=86,
            AiM_COPY_BKGB=12,
            AiM_COPY_GBBK=12,
            AiM_EOC=1,
            AiM_EWADD=2,
            AiM_EWMUL=10,
            AiM_MAC_ABK=1618,
            AiM_RD_AF=86,
            AiM_RD_MAC=1618,
            AiM_SYNC=5,
            AiM_WR_ABK=256,
            AiM_WR_BIAS=1618,
            AiM_WR_GB=43,
            R_MEM=6912,
            W_MEM=29952,
        ),
        "full_block_l512": _contract(
            "5cbd7a6b21866a151c2e8e7f6eb8962a692d4288b93fb475dd21959bdde0210e",
            AiM_AF=86,
            AiM_COPY_BKGB=12,
            AiM_COPY_GBBK=12,
            AiM_EOC=1,
            AiM_EWADD=2,
            AiM_EWMUL=10,
            AiM_MAC_ABK=1714,
            AiM_RD_AF=86,
            AiM_RD_MAC=1714,
            AiM_SYNC=5,
            AiM_WR_ABK=256,
            AiM_WR_BIAS=1714,
            AiM_WR_GB=55,
            R_MEM=13056,
            W_MEM=42240,
        ),
        "full_block_l4096": _contract(
            "fd961aca0d3385eca057b7d64c1177c56ac1252e66bebfcc006627974c557f86",
            AiM_AF=86,
            AiM_COPY_BKGB=12,
            AiM_COPY_GBBK=12,
            AiM_EOC=1,
            AiM_EWADD=2,
            AiM_EWMUL=16,
            AiM_MAC_ABK=2706,
            AiM_RD_AF=86,
            AiM_RD_MAC=2706,
            AiM_SYNC=5,
            AiM_WR_ABK=256,
            AiM_WR_BIAS=2706,
            AiM_WR_GB=179,
            R_MEM=70400,
            W_MEM=156928,
        ),
    }
)


__all__ = [
    "CENT_SOURCE_COMMIT",
    "VendorTraceContract",
    "VENDOR_TRACE_CONTRACTS",
]
