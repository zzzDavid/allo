# SPDX-License-Identifier: Apache-2.0

add_mlir_dialect_library(MLIRPimTarget
        PimTargetDialect.cpp
        PimTargetOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/allo/Dialect/PimTarget

        DEPENDS
        MLIRPimTargetOpsIncGen
        MLIRPimTargetInterfacesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        )
