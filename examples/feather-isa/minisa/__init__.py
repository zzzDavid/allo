# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MINISA: Minimal Instruction Set Architecture for FEATHER+

This package provides a VN-level (Virtual Neuron) programming abstraction
for the FEATHER reconfigurable accelerator, reducing control overhead while
preserving dataflow and layout flexibility.

Key Components:
- isa.py: Instruction definitions (SetIVNLayout, SetWVNLayout, SetOVNLayout, SetMapping)
- layout.py: VN layout descriptors and encoding
- interpreter.py: MINISA interpreter state machine
- lowering.py: MINISA to FEATHER control translation
- birrd_codegen.py: BIRRD instruction generation
"""

from .isa import (
    SetIVNLayout,
    SetWVNLayout,
    SetOVNLayout,
    SetMapping,
    MINISAInstruction,
)

from .layout import (
    VNLayout,
    IVNLayout,
    WVNLayout,
    OVNLayout,
    LayoutOrder,
)

from .interpreter import MINISAInterpreter

from .lowering import MINISALowering

__all__ = [
    # Instructions
    'SetIVNLayout',
    'SetWVNLayout',
    'SetOVNLayout',
    'SetMapping',
    'MINISAInstruction',
    # Layouts
    'VNLayout',
    'IVNLayout',
    'WVNLayout',
    'OVNLayout',
    'LayoutOrder',
    # Interpreter
    'MINISAInterpreter',
    # Lowering
    'MINISALowering',
]
