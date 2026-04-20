"""Python ISR-level functional model of SK-Hynix AiM.

The real aim_simulator is trace/timing only — it doesn't compute MAC values,
AF LUT outputs, or EWMUL results. This shadow executes the same ISR stream
the DSL emits and tracks real tensor state, so we can verify numerically
against a numpy reference.

State tracked
-------------
* banks[(channel, bank, row)] -> fp16[16]
    Each AiM burst row holds 16 fp16 elements (256 bits).
* gb[channel] -> fp16[opsize, 16]
    Per-channel global buffer. WR_GB loads `opsize` burst rows.
* mac_reg[(channel, bank)] -> fp16[16]
    Per-bank MAC accumulator (one burst's worth).
* af_reg[(channel, bank)] -> fp16[16]
    Per-bank AF (activation) register, written by the AF ISR.
* gpr[i] -> fp16[16]
    Host-facing GPR file.
* cfr = {"broadcast": 0|1, "ewmul_bg": 0|1, "afm": int}
    Configuration register snapshot used to disambiguate MAC_ABK and AF modes.

ISR coverage
------------
  WR_SBK, WR_ABK, WR_GB, WR_BIAS, RD_MAC, RD_AF, RD_SBK,
  COPY_BKGB, COPY_GBBK, MAC_SBK, MAC_ABK, AF, EWMUL, EWADD, SYNC, EOC.
Each ISR mutates exactly the state its documented semantics mutate (see
aim_simulator/README.md). The shadow is not cycle-accurate and makes no
claims about timing — cross-check cycles with the real ramulator2.

Channel mask
------------
`channel_mask` is a bitmask of the 32 channels. Example: mask=15 hits chans
[0..3]; mask=1<<c hits a single channel c. Broadcast ops (e.g., MAC_ABK with
CFR[broadcast]=1) use GB per-channel and touch all 16 banks.
"""
from __future__ import annotations

import numpy as np

LANES = 16     # fp16 per burst / per MAC_ABK lane
BANKS = 16     # banks per channel
CHANS = 32     # total channels in the modeled AiM part


class AiMShadow:
    def __init__(self):
        self.banks = {}                # (ch,bk,row) -> fp16[16]
        self.gb = {}                   # ch -> fp16[opsize,16]
        self.mac_reg = {}              # (ch,bk) -> fp16[16]
        self.af_reg = {}               # (ch,bk) -> fp16[16]
        self.gpr = {}                  # i -> fp16[16]
        self.cfr = {"broadcast": 0, "ewmul_bg": 0, "afm": 0}
        self.pending_gpr_data = {}     # i -> fp16[16] staged by set_gpr() before W GPR

    # -- host-side staging -----------------------------------------------------

    def set_gpr(self, i: int, data: np.ndarray):
        """Stage data that the next `W GPR i` on the trace will consume."""
        assert data.shape == (LANES,), f"gpr data must be shape ({LANES},), got {data.shape}"
        self.pending_gpr_data[i] = data.astype(np.float32)

    def read_gpr(self, i: int) -> np.ndarray:
        return self.gpr.get(i, np.zeros(LANES, dtype=np.float32)).copy()

    def read_bank(self, ch: int, bk: int, row: int) -> np.ndarray:
        return self.banks.get((ch, bk, row), np.zeros(LANES, dtype=np.float32)).copy()

    # -- channel mask iteration ------------------------------------------------

    @staticmethod
    def _chans(mask: int):
        return [c for c in range(CHANS) if (mask >> c) & 1]

    # -- ISR dispatch ----------------------------------------------------------

    def step(self, line: str):
        """Execute one trace line. Comments and `W CFR / GPR / MEM` headers
        that don't carry data are handled here; data-carrying GPR writes come
        via staged set_gpr() before the matching W GPR line."""
        s = line.strip()
        if not s or s.startswith("#"):
            return
        toks = s.split()

        if toks[0] == "W" and toks[1] == "GPR":
            i = int(toks[2])
            if i in self.pending_gpr_data:
                self.gpr[i] = self.pending_gpr_data.pop(i)
            else:
                self.gpr.setdefault(i, np.zeros(LANES, dtype=np.float32))
            return

        if toks[0] == "W" and toks[1] == "CFR":
            cid, data = int(toks[2]), int(toks[3])
            self.cfr[{0: "broadcast", 1: "ewmul_bg", 2: "afm"}[cid]] = data
            return

        if toks[0] in ("W", "R") and toks[1] == "MEM":
            return   # host DRAM accesses: timing only, no functional effect

        assert toks[0] == "AiM", f"unknown trace line: {line!r}"
        op = toks[1]
        args = toks[2:]

        getattr(self, f"_op_{op.lower()}")(args)

    # -- individual ISRs -------------------------------------------------------

    def _op_wr_sbk(self, a):
        gpr, mask, bank, row = int(a[0]), int(a[1]), int(a[2]), int(a[3])
        v = self.read_gpr(gpr)
        for ch in self._chans(mask):
            self.banks[(ch, bank, row)] = v.copy()

    def _op_wr_abk(self, a):
        gpr, mask, row = int(a[0]), int(a[1]), int(a[2])
        v = self.read_gpr(gpr)
        for ch in self._chans(mask):
            for bk in range(BANKS):
                self.banks[(ch, bk, row)] = v.copy()

    def _op_rd_sbk(self, a):
        gpr, mask, bank, row = int(a[0]), int(a[1]), int(a[2]), int(a[3])
        ch = self._chans(mask)[0]   # single-channel by spec
        self.gpr[gpr] = self.read_bank(ch, bank, row)

    def _op_wr_gb(self, a):
        opsize, gpr, mask = int(a[0]), int(a[1]), int(a[2])
        block = np.stack([self.read_gpr(gpr + i) for i in range(opsize)], axis=0)
        for ch in self._chans(mask):
            self.gb[ch] = block.copy()

    def _op_wr_bias(self, a):
        gpr, mask = int(a[0]), int(a[1])
        v = self.read_gpr(gpr)
        for ch in self._chans(mask):
            for bk in range(BANKS):
                self.mac_reg[(ch, bk)] = v.copy()

    def _op_rd_mac(self, a):
        gpr, mask = int(a[0]), int(a[1])
        # real AiM packs 16 banks into one 256-bit GPR by taking the first
        # element of each bank's MAC. We match that: one GPR = 16 banks' worth.
        # For multiple channels, we write consecutive GPRs (one per channel).
        for idx, ch in enumerate(self._chans(mask)):
            packed = np.array([self.mac_reg.get((ch, bk),
                               np.zeros(LANES, dtype=np.float32))[0]
                               for bk in range(BANKS)],
                              dtype=np.float32)
            self.gpr[gpr + idx] = packed

    def _op_rd_af(self, a):
        gpr, mask = int(a[0]), int(a[1])
        for idx, ch in enumerate(self._chans(mask)):
            packed = np.array([self.af_reg.get((ch, bk),
                               np.zeros(LANES, dtype=np.float32))[0]
                               for bk in range(BANKS)],
                              dtype=np.float32)
            self.gpr[gpr + idx] = packed

    def _op_mac_abk(self, a):
        """MAC across all 16 banks in selected channels.

        Broadcast mode (CFR[0]=1): GB is the other operand. For opsize o, each
        bank accumulates sum over o burst-rows: MAC[ch,bk] += sum_r GB[ch,r] *
        bank[ch,bk,row+r].

        Non-broadcast mode (CFR[0]=0): bankgroup-pair mode — bank pairs each
        other. We only need broadcast mode for our emitted patterns, so the
        other branch is a partial placeholder.
        """
        opsize, mask, row = int(a[0]), int(a[1]), int(a[2])
        if self.cfr["broadcast"] == 1:
            for ch in self._chans(mask):
                gb = self.gb.get(ch)
                assert gb is not None, f"MAC_ABK broadcast on ch{ch} with no GB data"
                assert gb.shape[0] >= opsize, f"GB has {gb.shape[0]} rows, need {opsize}"
                for bk in range(BANKS):
                    acc = self.mac_reg.setdefault(
                        (ch, bk), np.zeros(LANES, dtype=np.float32)).copy()
                    for r in range(opsize):
                        bank_v = self.read_bank(ch, bk, row + r)
                        # element-wise MAC: 16 lanes, per-lane multiply-add
                        acc = acc + gb[r] * bank_v
                    self.mac_reg[(ch, bk)] = acc
        else:
            # next-bank mode, not used by current emitters; implement on demand
            pass

    def _op_mac_sbk(self, a):
        opsize, mask, bank, row = int(a[0]), int(a[1]), int(a[2]), int(a[3])
        for ch in self._chans(mask):
            gb = self.gb.get(ch)
            assert gb is not None
            acc = self.mac_reg.setdefault(
                (ch, bank), np.zeros(LANES, dtype=np.float32)).copy()
            for r in range(opsize):
                acc = acc + gb[r] * self.read_bank(ch, bank, row + r)
            self.mac_reg[(ch, bank)] = acc

    def _op_af(self, a):
        """Activation. `afm` CFR selects the function; AiM paper exposes
        sigmoid, tanh, GELU, ReLU. We model identity=0 / relu=1 / sigmoid=2 /
        tanh=3 / gelu=4 — enough for verification."""
        mask = int(a[0])
        fn = {0: lambda x: x,
              1: lambda x: np.maximum(x, 0),
              2: lambda x: 1.0 / (1.0 + np.exp(-x)),
              3: np.tanh,
              4: lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
              }[self.cfr["afm"]]
        for ch in self._chans(mask):
            for bk in range(BANKS):
                self.af_reg[(ch, bk)] = fn(self.mac_reg.get(
                    (ch, bk), np.zeros(LANES, dtype=np.float32)))

    def _op_ewmul(self, a):
        """Element-wise multiply between two bank rows — writes to the same
        row in one of the banks per the AiM spec. We model it as: for each
        channel, multiply bank[0]'s row by bank[1]'s row into bank[0]."""
        opsize, mask, row = int(a[0]), int(a[1]), int(a[2])
        for ch in self._chans(mask):
            for r in range(opsize):
                a0 = self.read_bank(ch, 0, row + r)
                a1 = self.read_bank(ch, 1, row + r)
                self.banks[(ch, 0, row + r)] = a0 * a1

    def _op_ewadd(self, a):
        """GPR+GPR add, opsize burst rows. gpr1 <- gpr0 + gpr1 (simplified)."""
        opsize, g0, g1 = int(a[0]), int(a[1]), int(a[2])
        for i in range(opsize):
            v0 = self.read_gpr(g0 + i)
            v1 = self.read_gpr(g1 + i)
            self.gpr[g1 + i] = v0 + v1

    def _op_copy_bkgb(self, a):
        opsize, mask, bank, row = int(a[0]), int(a[1]), int(a[2]), int(a[3])
        for ch in self._chans(mask):
            self.gb[ch] = np.stack(
                [self.read_bank(ch, bank, row + r) for r in range(opsize)], axis=0)

    def _op_copy_gbbk(self, a):
        opsize, mask, bank, row = int(a[0]), int(a[1]), int(a[2]), int(a[3])
        for ch in self._chans(mask):
            gb = self.gb.get(ch)
            assert gb is not None
            for r in range(opsize):
                self.banks[(ch, bank, row + r)] = gb[r].copy()

    def _op_sync(self, a):  pass
    def _op_eoc(self, a):   pass


def run_trace(trace_lines, staging=None):
    """Execute a trace. `staging` is a list of `(gpr_id, data)` calls to run
    before each trace line (a dict keyed by line-index works too).

    Returns the final AiMShadow state so callers can read GPR/bank values."""
    sh = AiMShadow()
    staging = staging or {}
    for idx, line in enumerate(trace_lines):
        if idx in staging:
            for (g, d) in staging[idx]:
                sh.set_gpr(g, d)
        sh.step(line)
    return sh
