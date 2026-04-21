//===- PimTargetOps.cpp - PimTarget dialect ops -----------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "allo/Dialect/PimTarget/PimTargetOps.h"

using namespace mlir;
using namespace mlir::pim_target;

// Interface method definitions (all provided via `defaultImplementation`
// in PimTargetInterfaces.td).
#include "allo/Dialect/PimTarget/PimTargetInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "allo/Dialect/PimTarget/PimTargetOps.cpp.inc"
