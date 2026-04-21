//===- PimTargetOps.h - PimTarget dialect ops -------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef ALLO_DIALECT_PIMTARGET_PIMTARGETOPS_H
#define ALLO_DIALECT_PIMTARGET_PIMTARGETOPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "allo/Dialect/PimTarget/PimTargetDialect.h"
#include "allo/Dialect/PimTarget/PimTargetInterfaces.h"

#define GET_OP_CLASSES
#include "allo/Dialect/PimTarget/PimTargetOps.h.inc"

#endif // ALLO_DIALECT_PIMTARGET_PIMTARGETOPS_H
