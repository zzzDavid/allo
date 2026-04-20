//===- pim-opt.cpp - PIM opt driver -----------------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0
//
// Minimal MLIR optimizer driver that registers the builtin MLIR dialects
// and the pim_target dialect, so IR using pim_target ops can be parsed,
// round-tripped and (later) transformed.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "allo/Dialect/PimTarget/PimTargetDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllPasses();
  registry.insert<mlir::pim_target::PimTargetDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "PIM opt driver\n", registry));
}
