//===- PimTargetDialect.cpp - PimTarget dialect -----------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "allo/Dialect/PimTarget/PimTargetDialect.h"
#include "allo/Dialect/PimTarget/PimTargetOps.h"

using namespace mlir;
using namespace mlir::pim_target;

#include "allo/Dialect/PimTarget/PimTargetDialect.cpp.inc"

void PimTargetDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "allo/Dialect/PimTarget/PimTargetOps.cpp.inc"
      >();
}
