#!/usr/bin/env python3
"""Test if FEATHER example imports and builds correctly"""

import sys
sys.path.insert(0, "/home/nz264/shared/allo/examples/feather")

from allo.ir.types import int8
import allo.dataflow as df
from feather import get_feather_top

# Try to create and build the FEATHER example
AH, AW = 4, 4
Ty = int8

print("Creating FEATHER top...")
top = get_feather_top(AW, AH, Ty)
print("✓ Created successfully")

print("\nBuilding with simulator...")
try:
    mod = df.build(top, target="simulator")
    print("✓ Built successfully!")
except Exception as e:
    print(f"✗ Build failed: {e}")
    import traceback
    traceback.print_exc()
