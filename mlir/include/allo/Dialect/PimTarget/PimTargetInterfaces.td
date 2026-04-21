//===- PimTargetInterfaces.td - PimTarget interfaces -*- tablegen -*-===//
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef PIM_TARGET_INTERFACES
#define PIM_TARGET_INTERFACES

include "mlir/IR/OpBase.td"

def CostAnalysisOpInterface : OpInterface<"CostAnalysisOpInterface"> {
  let description = [{
    Op-level cost query used by autoscheduler / annotation passes.

    Implemented by `pim_target.unit` and `pim_target.op`. Each method
    reads a corresponding attribute from the op's attribute dictionary
    and returns a default (`0` / `0.0`) when the attribute is absent.

    Phase 2 keeps the interface deliberately simple (flat attributes on
    the op). The structured `CostAttr` and the lambda-valued cost
    functions described in Report 11 §3.6 / §8.4 are phase 2b.
  }];

  let cppNamespace = "::mlir::pim_target";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Per-op latency in cycles. Returns 0 when unset.",
      /*retTy=*/"int64_t",
      /*methodName=*/"getLatency",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        ::mlir::Operation *op = $_op.getOperation();
        if (auto attr = op->getAttrOfType<::mlir::IntegerAttr>("latency"))
          return attr.getInt();
        return 0;
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Ops retired per cycle. Returns 0.0 when unset.",
      /*retTy=*/"double",
      /*methodName=*/"getThroughput",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        ::mlir::Operation *op = $_op.getOperation();
        if (auto attr = op->getAttrOfType<::mlir::FloatAttr>("throughput"))
          return attr.getValueAsDouble();
        if (auto attr = op->getAttrOfType<::mlir::IntegerAttr>("throughput"))
          return static_cast<double>(attr.getInt());
        return 0.0;
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Energy (pJ) per invocation. Returns 0.0 when unset.",
      /*retTy=*/"double",
      /*methodName=*/"getEnergy",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        ::mlir::Operation *op = $_op.getOperation();
        if (auto attr = op->getAttrOfType<::mlir::FloatAttr>("energy_pJ"))
          return attr.getValueAsDouble();
        return 0.0;
      }]
    >,
  ];
}

#endif // PIM_TARGET_INTERFACES
