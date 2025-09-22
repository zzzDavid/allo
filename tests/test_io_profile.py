# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo


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


if __name__ == "__main__":
    run_io_profile_demo()
