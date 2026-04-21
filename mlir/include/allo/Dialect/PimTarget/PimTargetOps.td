//===- PimTargetOps.td - PimTarget dialect ops --*- tablegen -*-===//
//
// SPDX-License-Identifier: Apache-2.0
//
// Phase-1 op set: describe / memory / op (flat symbol declarations with
// free-form attribute dictionaries). Phase 2 extends the set with the
// structural target-description ops that mirror the `@tn.unit /
// tn.memory / tn.op / tn.stream / tn.cost` Python surface described in
// Report 11 §3:
//
//   pim_target.unit    — one level of the target grid (symbol, region)
//   pim_target.stream  — typed physical link between sibling units
//   pim_target.cost    — per-level cost attributes (attribute dict carrier)
//
// Cost is still carried as flat attributes on the op's attribute
// dictionary; the structured `CostAttr` and the lambda-valued cost
// functions described in Report 11 §3.6 / §8.4 are phase 2b.
//
//===----------------------------------------------------------------------===//

#ifndef PIM_TARGET_OPS
#define PIM_TARGET_OPS

include "allo/Dialect/PimTarget/PimTargetDialect.td"
include "allo/Dialect/PimTarget/PimTargetInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/CommonAttrConstraints.td"

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

def PimTarget_OpDeclOp : PimTarget_Op<"op", [
    Symbol,
    CostAnalysisOpInterface
  ]> {
  let summary = "Declare a target instruction (cost-bearing).";
  let description = [{
    Declares one target instruction. For phase 1 / 2, cost is a plain
    attribute — latency / throughput / energy_pJ read off the attribute
    dictionary via `CostAnalysisOpInterface`. Structured `CostAttr` is
    phase 2b.

    Example:
    ```mlir
    pim_target.op @mac {lanes = 16 : i32, latency = 4 : i32,
                        throughput = 1.0 : f32, energy_pJ = 0.5 : f32,
                        parent = @samsung}
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let assemblyFormat = "$sym_name attr-dict";
}

//===----------------------------------------------------------------------===//
// Phase 2 structural ops
//===----------------------------------------------------------------------===//

def PimTarget_UnitOp : PimTarget_Op<"unit", [
    Symbol,
    SymbolTable,
    IsolatedFromAbove,
    NoRegionArguments,
    CostAnalysisOpInterface
  ] # GraphRegionNoTerminator.traits> {
  let summary = "One level of the target grid.";
  let description = [{
    A `pim_target.unit` op declares one level of a target grid: an axis
    name (= `sym_name`), an extent (`mapping` = dense array of i64), an
    execution mode (`mode` ∈ {`"mimd"`, `"simd"`, `"simd-lockstep"`,
    `"simt-barrier"`, `"host"`}), and an optional parent symbol
    reference. The body region holds nested `pim_target.unit`,
    `pim_target.memory`, `pim_target.op`, `pim_target.stream` and
    `pim_target.cost` ops.

    This mirrors the `@tn.unit(mapping=[..], mode=..)` Python decorator
    surface from Report 11 §3.1.

    Example:
    ```mlir
    pim_target.unit @channel mapping [16] mode "mimd" {
      pim_target.unit @lane mapping [16] mode "simd" {
        pim_target.memory @grf {capacity_bytes = 32 : i64}
        pim_target.op @add {lanes = 16 : i32, latency = 2 : i32}
      }
    }
    ```
  }];
  let arguments = (ins
      SymbolNameAttr:$sym_name,
      DenseI64ArrayAttr:$mapping,
      StrAttr:$mode,
      OptionalAttr<FlatSymbolRefAttr>:$parent);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    $sym_name `mapping` $mapping `mode` $mode
    (`parent` $parent^)?
    attr-dict-with-keyword
    $body
  }];
}

def PimTarget_StreamOp : PimTarget_Op<"stream", [Symbol]> {
  let summary = "Typed physical link between sibling unit instances.";
  let description = [{
    Declares a typed physical link between sibling instances of the
    enclosing `pim_target.unit` level. Carries an element type, a
    capacity N, a topology descriptor, and an optional parent unit
    symbol reference. Forbidden inside `simd` / `simd-lockstep` units
    (Rule R4); the Python front-end enforces this — the MLIR verifier
    is phase 2b.

    Example:
    ```mlir
    pim_target.stream @neighbor_link element_type i32 N 16
      topology "neighbor" parent @dpu
    ```
  }];
  let arguments = (ins
      SymbolNameAttr:$sym_name,
      TypeAttr:$element_type,
      I64Attr:$N,
      StrAttr:$topology,
      OptionalAttr<FlatSymbolRefAttr>:$parent);
  let assemblyFormat = [{
    $sym_name `element_type` $element_type `N` $N
    `topology` $topology
    (`parent` $parent^)?
    attr-dict
  }];
}

def PimTarget_CostOp : PimTarget_Op<"cost"> {
  let summary = "Per-level cost attributes.";
  let description = [{
    Carries per-level cost attributes (`traverse`, `broadcast`,
    `reduce`, ...) for the enclosing `pim_target.unit`. The attribute
    dictionary holds the cost values verbatim. The lambda-valued cost
    functions described in Report 11 §3.6 are phase 2b; this op exists
    so that flat numeric costs can be round-tripped today.

    Example:
    ```mlir
    pim_target.cost {traverse = 3 : i64, broadcast = 8 : i64}
    ```
  }];
  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

#endif // PIM_TARGET_OPS
