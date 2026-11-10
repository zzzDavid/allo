"""Negative exercise of the installed-artifact probes (round 108): run a probe
with `allo.compile` replaced by a fake that returns a "virtual" result with
zero cycles and computes nothing (the reviewer's injection).  Every probe must
fail on its own contract assertions, so that the failure reaches the stage
harness and `milestone_finish`.

    python fake_compile_runner.py <probe.py>
"""

import runpy
import sys
import types

import allo


class _Fake:
    def __init__(self):
        self.compiled = types.SimpleNamespace(
            cmds=[],
            layout_ctx=types.SimpleNamespace(
                programs=[types.SimpleNamespace(kind="REDUCE")]
            ),
        )
        self.placement_ranking = types.SimpleNamespace(
            kernels=[
                types.SimpleNamespace(bucket_count=1, scores=[], chosen_position=0)
            ]
        )

    def estimate(self):
        return types.SimpleNamespace(cycles=0)

    def __call__(self, *arrays, **kwargs):
        return types.SimpleNamespace(
            kind="virtual",
            backend="virtual",
            cycles=0,
            outputs={},
            extra={"conductor": {"pimcmdgen_calls": 0}, "sdk_build": None},
        )


def _fake_compile(*args, **kwargs):
    return _Fake()


allo.compile = _fake_compile
print(
    "NEGATIVE: allo.compile replaced by a fake returning kind=virtual, cycles=0, no outputs"
)
sys.argv = [sys.argv[1]]
runpy.run_path(sys.argv[0], run_name="__main__")
print("NEGATIVE: the probe did not fail (this line must never be reached)")
