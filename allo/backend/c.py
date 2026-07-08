# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable C emission from Allo MLIR.

This module is deliberately smaller than :mod:`allo.backend.hls`.  It reuses
Allo's MLIR lowering and structured C emitter, but removes the HLS runtime and
header surface from the generated translation unit.  Backends such as UPMEM
can therefore wrap the resulting functions in a device-specific ABI without
re-parsing the original Python AST or maintaining a second loop/code emitter.

The supported type surface is ordinary C11 scalar types.  Arbitrary-width and
fixed-point Allo values still lower to ``ap_int``/``ap_fixed`` and are rejected
explicitly because those C++ templates are not available to a portable C
compiler.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import io
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np

from .._mlir.dialects import allo as allo_d
from .._mlir.ir import Context, Location, Module, UnitAttr
from .._mlir.passmanager import PassManager
from ..ir.transform import find_func_in_module
from ..passes import (
    _mlir_lower_pipeline,
    analyze_arg_load_store,
    decompose_library_function,
)
from ..utils import get_func_inputs_outputs


class PortableCError(RuntimeError):
    """Raised when lowered MLIR requires a C++/HLS-only source construct."""


_NUMPY_DTYPES = {
    "i1": np.dtype(np.bool_),
    "ui1": np.dtype(np.bool_),
    "i8": np.dtype(np.int8),
    "ui8": np.dtype(np.uint8),
    "i16": np.dtype(np.int16),
    "ui16": np.dtype(np.uint16),
    "i32": np.dtype(np.int32),
    "ui32": np.dtype(np.uint32),
    "i64": np.dtype(np.int64),
    "ui64": np.dtype(np.uint64),
    "index": np.dtype(np.int64),
    "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
}


@dataclass(frozen=True)
class CArgument:
    """One argument in the emitted C ABI, in positional order.

    ``mode`` is one of ``in``, ``out``, ``both``, ``scalar``, or ``func`` and
    comes from Allo's interprocedural load/store analysis.  Returned memrefs
    are materialized by the emitter as trailing ``out`` arguments.
    """

    index: int
    dtype: str
    shape: tuple[int, ...]
    mode: str
    source: str = "argument"

    @property
    def n_elements(self) -> int:
        result = 1
        for extent in self.shape:
            result *= extent
        return result


_HLS_INCLUDES = re.compile(
    r"^#include <(?:algorithm|ap_axi_sdata\.h|ap_fixed\.h|ap_int\.h|"
    r"hls_math\.h|hls_stream\.h|hls_vector\.h)>\s*$",
    re.MULTILINE,
)
_HLS_PRAGMA = re.compile(r"^[ \t]*#pragma\s+HLS\b.*$", re.MULTILINE)
_UNSUPPORTED_CXX = (
    (re.compile(r"\bap_(?:u?int)\s*<"), "integer wider than 64 bits"),
    (re.compile(r"\bap_(?:u?fixed)\s*<"), "fixed-point type"),
    (re.compile(r"\bhls::"), "HLS library call or type"),
    (re.compile(r"\b(?:half|bfloat16)\b"), "non-C11 floating-point type"),
)


def _portable_c11(hls_source: str, *, wrap_wide_integers: bool = False) -> str:
    """Normalize Allo's structured HLS output into a C11 translation unit."""

    def replace_ap_integer(match):
        unsigned, width_text = match.groups()
        width = int(width_text)
        if width > 64:
            if wrap_wide_integers:
                # APU v1's uint16 programming model is modular.  All
                # caller-visible values are truncated to 16 bits, so retaining
                # the low 64 bits of a widened add/multiply tree preserves the
                # observable result while remaining compilable by ARC GCC.
                # Use unsigned storage even for a signed HLS temporary to
                # avoid undefined signed-overflow behavior in C.
                return "uint64_t"
            return match.group(0)
        storage_width = next(
            candidate for candidate in (8, 16, 32, 64) if width <= candidate
        )
        return f"{'u' if unsigned else ''}int{storage_width}_t"

    # The structured emitter uses ap_int for widened intermediates even when
    # every source/ABI value is a native i32 (for example i32 + 1 -> i33).
    # Widen those temporaries to the next standard C storage width.  The
    # emitter's following cast preserves the original result semantics.
    hls_source = re.sub(r"\bap_(u?)int\s*<\s*(\d+)\s*>", replace_ap_integer, hls_source)

    for pattern, description in _UNSUPPORTED_CXX:
        if pattern.search(hls_source):
            raise PortableCError(
                f"portable C emission does not support {description}; "
                "use a standard C integer or floating-point Allo type"
            )

    source = _HLS_INCLUDES.sub("", hls_source)
    source = _HLS_PRAGMA.sub("", source)
    source = re.sub(r"^using namespace std;\s*$", "", source, flags=re.MULTILINE)
    source = source.replace("std::", "")

    # The HLS emitter relies on <algorithm> for these two calls.  A macro is
    # sufficient here because emitted operands are SSA temporaries/constants,
    # so neither operand has side effects or can be evaluated unexpectedly.
    source = re.sub(r"\bmin\s*\(", "ALLO_C_MIN(", source)
    source = re.sub(r"\bmax\s*\(", "ALLO_C_MAX(", source)

    preamble = """\
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

#ifndef ALLO_C_MIN
#define ALLO_C_MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef ALLO_C_MAX
#define ALLO_C_MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifdef __cplusplus
extern "C" {
#endif
"""
    # The generated banner says C++ even though the normalized body is C11.
    source = source.replace("*- C++ -*-", "*- C -*-")
    first_include = source.find("#include")
    if first_include >= 0:
        source = source[:first_include] + preamble + source[first_include:]
    else:
        source = preamble + "\n" + source
    source = re.sub(r"\n{4,}", "\n\n\n", source).strip()
    return source + "\n\n#ifdef __cplusplus\n}\n#endif\n"


class MLIRToCModule:
    """Lower an Allo MLIR module and expose portable C plus ABI metadata.

    The input module is cloned, so constructing this object cannot consume or
    mutate the schedule module later used by matching/autoscheduling.
    """

    def __init__(self, mod, top_func_name: str, *, wrap_wide_integers=False):
        self.top_func_name = top_func_name
        self.source_mlir = str(mod)
        self._context = Context()
        with self._context, Location.unknown():
            allo_d.register_dialect(self._context)
            self.module = Module.parse(self.source_mlir, self._context)
            top = find_func_in_module(self.module, top_func_name)
            if top is None:
                raise ValueError(f"MLIR module has no function named {top_func_name!r}")

            input_types, output_types = get_func_inputs_outputs(top)
            modes = analyze_arg_load_store(self.module)[top_func_name]
            arguments = [
                CArgument(i, dtype, tuple(shape), modes[i])
                for i, (dtype, shape) in enumerate(input_types)
            ]
            arguments.extend(
                CArgument(
                    len(arguments) + i,
                    dtype,
                    tuple(shape),
                    "out",
                    source="result",
                )
                for i, (dtype, shape) in enumerate(output_types)
            )
            self.arguments = tuple(arguments)

            top.attributes["top"] = UnitAttr.get()
            self.module = decompose_library_function(self.module)
            _mlir_lower_pipeline(self.module, lower_linalg=True)
            PassManager.parse(
                "builtin.module("
                "empty-tensor-to-alloc-tensor,"
                "func.func(convert-linalg-to-affine-loops)"
                ")"
            ).run(self.module.operation)
            self.lowered_mlir = str(self.module)

            output = io.StringIO()
            if not allo_d.emit_vhls(self.module, output):
                raise PortableCError("Allo's structured MLIR-to-C emission failed")
            self.hls_source = output.getvalue()
            self.source = _portable_c11(
                self.hls_source, wrap_wide_integers=wrap_wide_integers
            )

    @property
    def c_source(self) -> str:
        """Alias used by device backends that treat the module as an artifact."""

        return self.source

    def compile(
        self,
        compiler: str | None = None,
        *,
        source: str | None = None,
        compile_flags=(),
    ) -> "CompiledCModule":
        """Compile the portable source into a NumPy-callable shared library.

        This host implementation is primarily a correctness oracle for device
        wrappers.  UPMEM can consume the exact same ``source`` function inside
        its DPU entry point while using a different compiler and MRAM ABI.
        """

        return CompiledCModule(
            self,
            compiler=compiler,
            source=source,
            compile_flags=compile_flags,
        )


class CompiledCModule:
    """Host shared-library execution for :class:`MLIRToCModule`."""

    def __init__(
        self,
        artifact: MLIRToCModule,
        compiler: str | None = None,
        *,
        source: str | None = None,
        compile_flags=(),
    ):
        self.artifact = artifact
        self._tmpdir = tempfile.TemporaryDirectory(prefix="allo-c-")
        root = Path(self._tmpdir.name)
        compiler = compiler or shutil.which("clang") or shutil.which("gcc")
        if compiler is None:
            raise PortableCError("no C compiler found (tried clang and gcc)")
        cxx = Path(compiler).name.endswith("++")
        source_path = root / ("kernel.cc" if cxx else "kernel.c")
        library_path = root / "kernel.so"
        source_path.write_text(
            artifact.source if source is None else source, encoding="utf-8"
        )

        command = [
            compiler,
            "-std=c++17" if cxx else "-std=c11",
            "-O2",
            "-fPIC",
            "-shared",
            *tuple(compile_flags),
            str(source_path),
            "-lm",
            "-o",
            str(library_path),
        ]
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        if process.returncode != 0:
            raise PortableCError(
                "portable C compilation failed:\n"
                + " ".join(command)
                + "\n"
                + process.stderr
            )

        self.library_path = str(library_path)
        self.library = ctypes.CDLL(self.library_path)
        try:
            self.function = getattr(self.library, artifact.top_func_name)
        except AttributeError as error:
            raise PortableCError(
                f"compiled library does not export {artifact.top_func_name!r}"
            ) from error
        self.function.restype = None

    @staticmethod
    def _dtype(argument: CArgument) -> np.dtype:
        try:
            return _NUMPY_DTYPES[argument.dtype]
        except KeyError as error:
            raise PortableCError(
                f"host NumPy ABI does not support MLIR type {argument.dtype!r}"
            ) from error

    def __call__(self, *inputs):
        source_arguments = [
            argument
            for argument in self.artifact.arguments
            if argument.source == "argument"
        ]
        if len(inputs) != len(source_arguments):
            raise TypeError(
                f"{self.artifact.top_func_name} expects {len(source_arguments)} "
                f"arguments, received {len(inputs)}"
            )

        call_arguments = []
        keepalive = []
        for argument, value in zip(source_arguments, inputs):
            dtype = self._dtype(argument)
            if not argument.shape:
                scalar = np.ctypeslib.as_ctypes_type(dtype.type)(value)
                call_arguments.append(scalar)
                keepalive.append(scalar)
                continue
            if not isinstance(value, np.ndarray):
                raise TypeError(f"argument {argument.index} must be a NumPy array")
            if value.dtype != dtype:
                raise TypeError(
                    f"argument {argument.index} must have dtype {dtype}, got {value.dtype}"
                )
            if value.shape != argument.shape:
                raise ValueError(
                    f"argument {argument.index} must have shape {argument.shape}, "
                    f"got {value.shape}"
                )
            if not value.flags.c_contiguous:
                raise ValueError(f"argument {argument.index} must be C-contiguous")
            ctype = np.ctypeslib.as_ctypes_type(dtype.type)
            call_arguments.append(value.ctypes.data_as(ctypes.POINTER(ctype)))
            keepalive.append(value)

        results = []
        for argument in self.artifact.arguments:
            if argument.source != "result":
                continue
            dtype = self._dtype(argument)
            output = np.empty(argument.shape or (), dtype=dtype)
            ctype = np.ctypeslib.as_ctypes_type(dtype.type)
            # emit_vhls uniformly lowers source returns to trailing output
            # pointers, including scalar returns represented by a 0-D array.
            call_arguments.append(output.ctypes.data_as(ctypes.POINTER(ctype)))
            keepalive.append(output)
            results.append(output)

        self.function(*call_arguments)
        if not results:
            return None
        if len(results) == 1:
            return results[0].item() if results[0].shape == () else results[0]
        return tuple(
            result.item() if result.shape == () else result for result in results
        )


def emit_c_from_mlir(
    mod, top_func_name: str, *, wrap_wide_integers=False
) -> MLIRToCModule:
    """Convenience function returning a retained MLIR-to-C artifact."""

    return MLIRToCModule(mod, top_func_name, wrap_wide_integers=wrap_wide_integers)


__all__ = [
    "CArgument",
    "CompiledCModule",
    "MLIRToCModule",
    "PortableCError",
    "emit_c_from_mlir",
]
