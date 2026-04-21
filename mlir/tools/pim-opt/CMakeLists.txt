# SPDX-License-Identifier: Apache-2.0

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}
        MLIRIR
        MLIROptLib
        MLIRParser
        MLIRPass
        MLIRSupport
        MLIRTransforms
        MLIRPimTarget
        )

add_llvm_executable(pim-opt pim-opt.cpp)
llvm_update_compile_flags(pim-opt)
target_link_libraries(pim-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(pim-opt)
