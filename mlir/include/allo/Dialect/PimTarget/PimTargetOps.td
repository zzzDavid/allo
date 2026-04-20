//===- PimTargetOps.td - PimTarget dialect ops --*- tablegen -*-===//
//
// SPDX-License-Identifier: Apache-2.0
//
// Phase-1 op set: describe / memory / op. The three ops carry a symbol
// name plus a free attribute dictionary. Cost, capability flags and cross
// references live in the attribute dict; structured attribute types
// (e.g. CostAttr) are deferred to phase 2.
//
//===----------------------------------------------------------------------===//

#ifndef PIM_TARGET_OPS
#define PIM_TARGET_OPS

include "allo/Dialect/PimTarget/PimTargetDialect.td"
include "mlir/IR/SymbolInterfaces.td"

def PimTarget_DescribeOp : PimTarget_Op<"describe", [Symbol]> {
  let summary = "Declare a PIM target.";
  let description = [{
    Declares a PIM target by name. Extra properties (parallel_units,
    capability flags, ...) live in the attribute dictionary.

    Example:
    ```mlir
    pim_target.describe @samsung {parallel_units = 1024 : i64}
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let assemblyFormat = "$sym_name attr-dict";
}

def PimTarget_MemoryOp : PimTarget_Op<"memory", [Symbol]> {
  let summary = "Declare a memory region under a target.";
  let description = [{
    Declares a memory region of a parent target. All structural fields
    (parent, capacity_bytes, scope, ...) are free-form attributes.

    Example:
    ```mlir
    pim_target.memory @grf {capacity_bytes = 32 : i64, parent = @samsung}
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let assemblyFormat = "$sym_name attr-dict";
}

def PimTarget_OpDeclOp : PimTarget_Op<"op", [Symbol]> {
  let summary = "Declare a target instruction (cost-bearing).";
  let description = [{
    Declares one target instruction. For phase 1, cost is a plain
    attribute. The `cost` structured attribute is deferred to phase 2.

    Example:
    ```mlir
    pim_target.op @mac {lanes = 16 : i32, latency = 4 : i32, parent = @samsung}
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let assemblyFormat = "$sym_name attr-dict";
}

#endif // PIM_TARGET_OPS
