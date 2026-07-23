// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef TENON_APU_G2_GSI_LIBRARY_H_
#define TENON_APU_G2_GSI_LIBRARY_H_

#include <device/vector_core/gsi_libgal.h>
#include <transport_gsi_library.h>

GSI_LIBRARY_ENTRY_POINT_FUNCTION_PROTOTYPE(tenon_apu_g2_u16_add, uint64_t);
GSI_LIBRARY_ENTRY_POINT_FUNCTION_PROTOTYPE(tenon_apu_g2_u16_atax, uint64_t);
GSI_LIBRARY_ENTRY_POINT_FUNCTION_PROTOTYPE(tenon_apu_g2_u16_block_sum,
                                           uint64_t);
GSI_LIBRARY_ENTRY_POINT_FUNCTION_PROTOTYPE(tenon_apu_g2_u16_div, uint64_t);
GSI_LIBRARY_ENTRY_POINT_FUNCTION_PROTOTYPE(tenon_apu_g2_u16_dot_tile, uint64_t);
GSI_LIBRARY_ENTRY_POINT_FUNCTION_PROTOTYPE(tenon_apu_g2_u16_fill, uint64_t);
GSI_LIBRARY_ENTRY_POINT_FUNCTION_PROTOTYPE(tenon_apu_g2_u16_gemv, uint64_t);
GSI_LIBRARY_ENTRY_POINT_FUNCTION_PROTOTYPE(tenon_apu_g2_u16_gesummv, uint64_t);
GSI_LIBRARY_ENTRY_POINT_FUNCTION_PROTOTYPE(tenon_apu_g2_u16_minmax, uint64_t);
GSI_LIBRARY_ENTRY_POINT_FUNCTION_PROTOTYPE(tenon_apu_g2_u16_mul, uint64_t);
GSI_LIBRARY_ENTRY_POINT_FUNCTION_PROTOTYPE(tenon_apu_g2_u16_select_lt,
                                           uint64_t);
GSI_LIBRARY_ENTRY_POINT_FUNCTION_PROTOTYPE(tenon_apu_g2_u16_shift_right,
                                           uint64_t);
GSI_LIBRARY_ENTRY_POINT_FUNCTION_PROTOTYPE(tenon_apu_g2_u16_sqrt, uint64_t);
GSI_LIBRARY_ENTRY_POINT_FUNCTION_PROTOTYPE(tenon_apu_g2_u16_sub, uint64_t);
GSI_LIBRARY_ENTRY_POINT_FUNCTION_PROTOTYPE(tenon_apu_g2_u16_gemm_chunk,
                                           uint64_t);
GSI_LIBRARY_ENTRY_POINT_FUNCTION_PROTOTYPE(tenon_apu_g2_typed, uint64_t);
GSI_LIBRARY_ENTRY_POINT_FUNCTION_PROTOTYPE(tenon_apu_g2_composed_dot,
                                           uint64_t);

#endif
