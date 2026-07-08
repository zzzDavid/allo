# vector_binary on APU v1

Target-neutral packed XNOR/popcount milestone: `M=32`, `N=1024`, `K=8`,
16-bit packed words, one APUC.

```python
result[row, column] += allo.popcount(
    ~(left[row, depth] ^ right[depth, column])
)
```

| analytical cycles | measured CRUN | outputs checked | mismatches |
|---:|---:|---:|---:|
| 131,286 | 248,424 | 32,768 | 0 |

- correctness: **bit-exact PASS** for all 32,768 `int16` outputs.
- device: `apu_v1_device@zhang-capra-xcel.ece.cornell.edu`.
- runtime: GSI 13.7.1.
- measured wall time: 9.314730167 seconds.
- timestamp: `2026-07-07T17:47:00-04:00`.
- source revision before the working-tree milestone:
  `6bc7474a09fa7d2d64007e6b929b6f68660729d3`.

Reproduction:

```bash
LD_PRELOAD=/opt/intel/oneapi/intelpython/python3.9/pkgs/\
intel-extension-for-tensorflow-1.2.0-py3.9_gpu_0/share/\
intel_extension_for_tensorflow/libstdc/libstdc++.so.6.0.30 \
PYTHONPATH=.:tests/pim pytest -q \
tests/pim/apu_v1/vector_binary/test_vector_binary_apu_v1.py \
-m apu_v1_device
```

The DSL intrinsic retains `math.ctpop` in MLIR. The APU analyzer recognizes
the XOR, all-ones inversion, and popcount dataflow structurally, then lowers
raw popcount semantics to GVML XOR/NOT/POPCOUNT/ADD. It does not insert the
separate MICRO bipolar transformation `2*popcount-word_bits`.
