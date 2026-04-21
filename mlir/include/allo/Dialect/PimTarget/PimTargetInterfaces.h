//===- PimTargetInterfaces.h - PimTarget dialect interfaces -*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Declares the `CostAnalysisOpInterface` C++ interface implemented by
// `pim_target.unit` and `pim_target.op`. The interface is trivial: it
// surfaces latency / throughput / energy numbers carried as attributes
// on the op, returning defaults when an attribute is absent.
//
// Phase 2b (structured `CostAttr` + cost-function DSL) replaces the
// attribute-dict readers with real typed getters; until then the
// interface is a thin convenience for C++ passes that want to treat a
// `pim_target.op` as cost-bearing without reinventing the attribute
// lookup.
//
//===----------------------------------------------------------------------===//

#ifndef ALLO_DIALECT_PIMTARGET_PIMTARGETINTERFACES_H
#define ALLO_DIALECT_PIMTARGET_PIMTARGETINTERFACES_H

#include "mlir/IR/OpDefinition.h"

#include "allo/Dialect/PimTarget/PimTargetInterfaces.h.inc"

#endif // ALLO_DIALECT_PIMTARGET_PIMTARGETINTERFACES_H
