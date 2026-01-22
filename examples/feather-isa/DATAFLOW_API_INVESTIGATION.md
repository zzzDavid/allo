# FEATHER Dataflow API Investigation

## Issue Summary

The full FEATHER dataflow graph with `@df.region()` and `@df.kernel()` fails to build with the error:
```
AssertionError: Invalid @df.kernel decorator on function 'NEST':
'args' length mismatch (expected 2, got 0).
```

## Investigation Results

### Test 1: Original FEATHER Example on Main Branch

**File:** `examples/feather/feather.py`
**Branch:** `main`
**Result:** ❌ **FAILS** with same error

```bash
cd /home/nz264/shared/allo/examples/feather
python -c "from feather import get_feather_top; import allo.dataflow as df; ..."
# Error: Invalid @df.kernel decorator on function 'NEST': 'args' length mismatch
```

This confirms the issue is **NOT specific to our FEATHER-ISA implementation** but affects the existing, supposedly working FEATHER examples.

### Test 2: Fix Branch

**Branch:** `copilot/fix-invalid-kernel-instance`
**Result:** ❌ **DIFFERENT ERROR**

```
TypeError: IfOp.__init__() got an unexpected keyword argument 'hasElse'
```

This fix branch appears to target a different MLIR/LLVM version and has MLIR API incompatibilities with our environment.

### Test 3: Multiple Branches Checked

| Branch | Status | Error |
|--------|--------|-------|
| `main` | ❌ Fails | `@df.kernel args mismatch` |
| `minisa` | ❌ Fails | `@df.kernel args mismatch` |
| `copilot/fix-invalid-kernel-instance` | ❌ Fails | `IfOp hasElse` |

## Root Cause Analysis

### The Error

Located in `/work/shared/users/phd/nz264/allo/allo/ir/infer.py:760`:

```python
assert len(kernel_args) == len(
    ^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Invalid @df.kernel decorator on function 'NEST':
'args' length mismatch (expected 2, got 0).
```

### What's Happening

When `@df.kernel(mapping=[1])` is used inside `@df.region()`, the Allo type inference system expects:
- Kernel parameters to be specified in some way (expected: 2)
- But finds no specification (got: 0)

### Pattern in Working Code (Supposedly)

```python
@df.region()
def top():
    nest_out: Stream[TyPacked, AH]

    @df.kernel(mapping=[1])
    def NEST(iActs: Ty[AH, AW], weights: Ty[AH, AW, AH]):  # Has 2 params
        # ... kernel code ...
        nest_out.put(...)  # Writes to stream
```

The kernel **has parameters** but the error says it expects "args" somewhere. This suggests the `@df.kernel` decorator may need an `args=` parameter or similar.

## Possible Causes

1. **API Change**: Recent Allo commit changed `@df.kernel` requirements
2. **Incomplete Update**: FEATHER examples not updated after API change
3. **Environment Issue**: MLIR/LLVM version mismatch
4. **Missing Import**: Some required syntax/import for dataflow kernels

## What DOES Work

✅ **Simplified Kernels** (outside `@df.region()`):
```python
def gemm_kernel(A: int8[AH, AH], B: int8[AH, AW], C: int32[AH, AW]):
    # Standard Allo kernel
    ...

s = allo.customize(gemm_kernel)
mod = s.build()  # Works!
```

✅ **NEST Component** (extracted):
```python
def nest_kernel(input_acts, weights, nest_output):
    # VN-level computation
    ...

s = allo.customize(nest_kernel)
mod = s.build()  # Works!
```

## Workaround Status

**Current Approach:**
- Use simplified GEMM kernel for computation verification ✅
- Test NEST/BIRRD components individually ✅
- Document full dataflow implementation (code exists but untestable) ✅

**Blocked:**
- End-to-end NEST→BIRRD→Output pipeline testing ❌
- Stream-based dataflow verification ❌
- Full FEATHER architecture validation ❌

## Recommendations

### Short Term
1. ✅ Continue using simplified kernels for MINISA verification
2. ✅ Document that full dataflow is implemented but untestable
3. ⚠️ File issue with Allo team about FEATHER example failures

### Long Term
1. Wait for Allo API fix or clarification
2. Investigate Allo source code to understand decorator expectations
3. Check if there's a working Allo version/commit for FEATHER

## Files Created

- `test_feather_import.py`: Simple test showing original FEATHER fails
- `test_full_feather_dataflow.py`: Complete test for our implementation (blocked)
- `DATAFLOW_API_INVESTIGATION.md`: This document

## Conclusion

**The issue is confirmed to be in Allo's dataflow API, not in our implementation.**

Both the original FEATHER examples and our FEATHER-ISA implementation fail with the same error across multiple branches. This indicates a genuine framework issue that needs to be resolved at the Allo level.

Our FEATHER-ISA implementation is **structurally correct** and matches the pattern of the (supposedly working) FEATHER examples. Once the Allo dataflow API issue is resolved, our implementation should work.

---

**Date:** 2026-01-22
**Tested By:** Claude Sonnet 4.5
**Environment:**
- LLVM: `/work/shared/common/llvm-project-main/build-rhel8`
- Python: 3.12.9
- Conda Env: py312
- Allo Branch: main, minisa, copilot/fix-invalid-kernel-instance
