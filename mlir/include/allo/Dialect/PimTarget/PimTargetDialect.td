//===- PimTargetDialect.td - PimTarget dialect --*- tablegen -*-===//
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef PIM_TARGET_DIALECT
#define PIM_TARGET_DIALECT

include "mlir/IR/OpBase.td"

def PimTarget_Dialect : Dialect {
  let name = "pim_target";
  let summary = "Target-description and cost IR for PIM backends.";
  let description = [{
    Phase-1 skeleton of a first-class MLIR dialect for describing
    processing-in-memory (PIM) targets: the accelerator capabilities,
    its memory hierarchy, and its cost-bearing instructions.

    The ops carried here are structural: `pim_target.describe` declares a
    target, `pim_target.memory` declares a memory region under that target,
    and `pim_target.op` declares a target instruction with plain-text cost.
  }];
  let cppNamespace = "::mlir::pim_target";
}

class PimTarget_Op<string mnemonic, list<Trait> traits = []>
    : Op<PimTarget_Dialect, mnemonic, traits>;

#endif // PIM_TARGET_DIALECT
