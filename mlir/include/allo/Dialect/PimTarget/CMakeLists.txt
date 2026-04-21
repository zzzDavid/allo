# SPDX-License-Identifier: Apache-2.0

set(LLVM_TARGET_DEFINITIONS PimTargetInterfaces.td)
mlir_tablegen(PimTargetInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(PimTargetInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRPimTargetInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS PimTargetOps.td)
mlir_tablegen(PimTargetOps.h.inc -gen-op-decls)
mlir_tablegen(PimTargetOps.cpp.inc -gen-op-defs)
mlir_tablegen(PimTargetDialect.h.inc -gen-dialect-decls -dialect=pim_target)
mlir_tablegen(PimTargetDialect.cpp.inc -gen-dialect-defs -dialect=pim_target)
add_public_tablegen_target(MLIRPimTargetOpsIncGen)
add_dependencies(MLIRPimTargetOpsIncGen MLIRPimTargetInterfacesIncGen)
