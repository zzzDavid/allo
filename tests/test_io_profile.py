# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import numpy as np


def run_io_profile_demo():
    mlir = r"""
module {
  llvm.mlir.global internal constant @A("memref_A\00")
  llvm.mlir.global internal constant @B("memref_B\00")

  func.func @test_profile() {
    %c0 = llvm.mlir.constant(0 : index) : i64
    %bytes4 = arith.constant 4 : i64
    %bytes8 = arith.constant 8 : i64
    %bytes16 = arith.constant 16 : i64
    %bytes32 = arith.constant 32 : i64

    %gA = llvm.mlir.addressof @A : !llvm.ptr
    %pA = llvm.getelementptr %gA[%c0, %c0] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<2 x i8>

    
    %gB = llvm.mlir.addressof @B : !llvm.ptr
    %pB = llvm.getelementptr %gB[%c0, %c0] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<2 x i8>

    func.call @countLoadBytes(%pA, %bytes4) : (!llvm.ptr, i64) -> ()
    func.call @countLoadBytes(%pA, %bytes4) : (!llvm.ptr, i64) -> ()
    func.call @countStoreBytes(%pA, %bytes8) : (!llvm.ptr, i64) -> ()

    func.call @countLoadBytes(%pB, %bytes16) : (!llvm.ptr, i64) -> ()
    func.call @countStoreBytes(%pB, %bytes16) : (!llvm.ptr, i64) -> ()
    func.call @countStoreBytes(%pB, %bytes32) : (!llvm.ptr, i64) -> ()

    func.call @reportMemrefIO() : () -> ()
    return
    }
  func.func private @countLoadBytes(!llvm.ptr, i64)
  func.func private @countStoreBytes(!llvm.ptr, i64)
  func.func private @reportMemrefIO()
}
"""

    llvm_mod = allo.LLVMModule(mlir, "test_profile", io_profile=True)
    llvm_mod()


def run_insert_pass_demo():
    from allo._mlir.ir import Context, Location, Module
    from allo._mlir.dialects import allo as allo_d, arith as arith_d, func as func_d, memref as memref_d, llvm as llvm_d, affine as affine_d

    mlir = r"""
module {
  func.func @kernel(%A: memref<4xf32>, %B: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %v = memref.load %A[%c0] : memref<4xf32>
    memref.store %v, %B[%c0] : memref<4xf32>
    return
  }
}
"""

    with Context() as ctx, Location.unknown():
        allo_d.register_dialect(ctx)
        mod = Module.parse(mlir, ctx)
        allo_d.insert_io_profiling(mod)
        print(mod)


def run_end_to_end_io_profile_demo():
    mlir = r"""
module {
  func.func @kernel(%A: memref<4xf32>, %B: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %v = memref.load %A[%c0] : memref<4xf32>
    memref.store %v, %B[%c0] : memref<4xf32>
    return
  }
}
"""

    llvm_mod = allo.LLVMModule(mlir, "kernel", io_profile=True)
    A = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    B = np.zeros((4,), dtype=np.float32)
    llvm_mod(A, B)


if __name__ == "__main__":
    # run_io_profile_demo()
    # run_insert_pass_demo()
    run_end_to_end_io_profile_demo()
