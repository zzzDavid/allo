// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef TENON_APU_G2_GSI_LIBRARY_H_
#define TENON_APU_G2_GSI_LIBRARY_H_

#include <device/vector_core/gsi_libgal.h>
#include <transport_gsi_library.h>

GSI_LIBRARY_ENTRY_POINT_FUNCTION_PROTOTYPE(tenon_apu_g2_u16_add, uint64_t);
GSI_LIBRARY_ENTRY_POINT_FUNCTION_PROTOTYPE(tenon_apu_g2_typed, uint64_t);
GSI_LIBRARY_ENTRY_POINT_FUNCTION_PROTOTYPE(tenon_apu_g2_composed_dot,
                                           uint64_t);

#endif
