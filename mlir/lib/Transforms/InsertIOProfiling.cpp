/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Transforms/Passes.h"
#include "PassDetail.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::allo;

// Create or fetch a global C string and return an opaque !llvm.ptr to its first char at the current insertion point.
static Value getOrCreateGlobalCString(Location loc, OpBuilder &builder,
                                      ModuleOp module, StringRef symName,
                                      StringRef literalWithNull) {
  LLVM::GlobalOp global;
  if (!(global = module.lookupSymbol<LLVM::GlobalOp>(symName))) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto i8Ty = IntegerType::get(builder.getContext(), 8);
    auto arrayTy = LLVM::LLVMArrayType::get(i8Ty, literalWithNull.size());
    global = builder.create<LLVM::GlobalOp>(loc, arrayTy,
                                            /*isConstant=*/true,
                                            LLVM::Linkage::Internal, symName,
                                            builder.getStringAttr(literalWithNull),
                                            /*alignment=*/0);
  }
  // Build the address and GEP at the caller's insertion point.
  Value addr = builder.create<LLVM::AddressOfOp>(loc, global);
  Value c0_64 = builder.create<LLVM::ConstantOp>(
      loc, IntegerType::get(builder.getContext(), 64),
      builder.getIntegerAttr(builder.getIndexType(), 0));
  auto opaquePtrTy = LLVM::LLVMPointerType::get(builder.getContext());
  return builder.create<LLVM::GEPOp>(loc, opaquePtrTy, global.getType(), addr,
                                     ArrayRef<Value>({c0_64, c0_64}));
}

struct InsertIOProfilingPass
    : public InsertIOProfilingBase<InsertIOProfilingPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    OpBuilder moduleBuilder(module.getContext());

    // Ensure declarations for the runtime functions exist at module scope.
    auto i64Ty = moduleBuilder.getI64Type();
    auto voidTy = moduleBuilder.getNoneType();

    // We declare C-like functions using func.func with opaque pointer type.
    auto opaquePtrTy = LLVM::LLVMPointerType::get(module.getContext());

    auto ensureDecl = [&](StringRef name, TypeRange args, Type res) -> func::FuncOp {
      if (auto f = module.lookupSymbol<func::FuncOp>(name)) return f;
      auto fnType = moduleBuilder.getFunctionType(args, res.isa<NoneType>() ? TypeRange{} : TypeRange{res});
      OpBuilder::InsertionGuard g(moduleBuilder);
      moduleBuilder.setInsertionPointToStart(module.getBody());
      auto fn = moduleBuilder.create<func::FuncOp>(module.getLoc(), name, fnType);
      fn.setPrivate();
      return fn;
    };

    func::FuncOp countLoad = ensureDecl("countLoadBytes", TypeRange{opaquePtrTy, i64Ty}, voidTy);
    func::FuncOp countStore = ensureDecl("countStoreBytes", TypeRange{opaquePtrTy, i64Ty}, voidTy);
    func::FuncOp reportFn = ensureDecl("reportMemrefIO", TypeRange{}, voidTy);

    // Collect top-level function arguments that are memrefs.
    SmallVector<func::FuncOp> funcs;
    module.walk([&](func::FuncOp f) { funcs.push_back(f); });
    bool anyTop = llvm::any_of(funcs, [](func::FuncOp f) { return f->hasAttr("top"); });

    for (func::FuncOp func : funcs) {
      auto funcLoc = func.getLoc();

      auto buildNamePtrForArgAt = [&](OpBuilder &b, unsigned idx) -> Value {
        std::string base = (Twine("memref_") + func.getName() + Twine("_arg") + Twine(idx)).str();
        std::string literal = base + std::string("\0", 1);
        std::string symName = (Twine("__allo_prof_name_") + func.getName() + Twine("_arg") + Twine(idx)).str();
        return getOrCreateGlobalCString(funcLoc, b, module, symName, StringRef(literal));
      };

      // Identify which block arguments are memrefs.
      DenseSet<unsigned> memrefArgIdxs;
      for (auto [idx, arg] : llvm::enumerate(func.getArguments())) {
        if (arg.getType().isa<MemRefType>()) {
          memrefArgIdxs.insert(idx);
        }
      }

      if (memrefArgIdxs.empty()) continue;

      func.walk([&](Operation *op) {
        Location loc = op->getLoc();
        auto insertCounter = [&](BlockArgument memrefArg, bool isLoad, int64_t bytes) {
          OpBuilder b(op);
          unsigned idx = memrefArg.getArgNumber();
          Value namePtr = buildNamePtrForArgAt(b, idx);
          Value numBytes = b.create<arith::ConstantIntOp>(loc, bytes, 64);
          SmallVector<Value> args{namePtr, numBytes};
          b.create<func::CallOp>(loc, isLoad ? countLoad : countStore, args);
          return success();
        };

        if (auto load = dyn_cast<memref::LoadOp>(op)) {
          Value src = load.getMemref();
          if (auto arg = src.dyn_cast<BlockArgument>()) {
            if (arg.getOwner()->isEntryBlock()) {
              auto mrTy = arg.getType().cast<MemRefType>();
              int64_t elemBytes = (mrTy.getElementTypeBitWidth() + 7) / 8;
              (void)insertCounter(arg, /*isLoad=*/true, elemBytes);
            }
          }
        } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
          Value dst = store.getMemref();
          if (auto arg = dst.dyn_cast<BlockArgument>()) {
            if (arg.getOwner()->isEntryBlock()) {
              auto mrTy = arg.getType().cast<MemRefType>();
              int64_t elemBytes = (mrTy.getElementTypeBitWidth() + 7) / 8;
              (void)insertCounter(arg, /*isLoad=*/false, elemBytes);
            }
          }
        } else if (auto aload = dyn_cast<affine::AffineLoadOp>(op)) {
          Value src = aload.getMemref();
          if (auto arg = src.dyn_cast<BlockArgument>()) {
            if (arg.getOwner()->isEntryBlock()) {
              auto mrTy = arg.getType().cast<MemRefType>();
              int64_t elemBytes = (mrTy.getElementTypeBitWidth() + 7) / 8;
              (void)insertCounter(arg, /*isLoad=*/true, elemBytes);
            }
          }
        } else if (auto astore = dyn_cast<affine::AffineStoreOp>(op)) {
          Value dst = astore.getMemref();
          if (auto arg = dst.dyn_cast<BlockArgument>()) {
            if (arg.getOwner()->isEntryBlock()) {
              auto mrTy = arg.getType().cast<MemRefType>();
              int64_t elemBytes = (mrTy.getElementTypeBitWidth() + 7) / 8;
              (void)insertCounter(arg, /*isLoad=*/false, elemBytes);
            }
          }
        }
      });

      // Insert reportMemrefIO() before returns of the top function(s).
      if (func->hasAttr("top") || !anyTop) {
        SmallVector<func::ReturnOp> returns;
        func.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
        for (auto ret : returns) {
          OpBuilder b(ret);
          b.create<func::CallOp>(ret.getLoc(), reportFn, ValueRange{});
        }
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
mlir::allo::createInsertIOProfilingPass() {
  return std::make_unique<InsertIOProfilingPass>();
}

bool mlir::allo::applyInsertIOProfiling(ModuleOp &module) {
  PassManager pm(module.getContext());
  pm.addPass(createInsertIOProfilingPass());
  return succeeded(pm.run(module));
} 