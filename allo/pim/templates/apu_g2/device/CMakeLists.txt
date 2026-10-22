# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20...3.30)
project(tenon_apu_g2_tasks LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

set(CMAKE_BUILD_TYPE ArcDebug)
# ArcDebug is required to select the ARC toolchain, but the GSI runtime and
# wrapper optimization policy are selected independently.  Use the installed
# release runtime and make the wrapper optimization explicit: the SDK's
# GSI_SYSTEM_OPT_FLAGS lookup is keyed by CMAKE_BUILD_TYPE and therefore
# resolves to an empty list for the required nonstandard "ArcDebug" value.
set(GSI_SYSTEM_CMAKE_BUILD_TYPE Release)
set(GSI_SYSTEM_PRODUCT_NAME g2)
set(GSI_SYSTEM_TARGET_NAME vector_core)

find_package(GSI_SYSTEM_g2 REQUIRED CONFIG)
find_package(GSI_GTML_g2 REQUIRED CONFIG)
find_package(g2_transport REQUIRED CONFIG
  PATHS /opt/gsi/lib/cmake/g2_transport NO_DEFAULT_PATH)
find_file(GTML_MD_UPDATE_SCRIPT g2_md_update.py
  PATHS /opt/gsi/g2/vector_core/bin NO_DEFAULT_PATH REQUIRED)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  /opt/gsi/include/64vl/g2_64vl
  /opt/gsi/include/g2_common
  /opt/gsi/include/gtml/g2_gtml
  /opt/gsi/include/gsi
  /opt/gsi/include/gsi/device
  ${g2_transport_INCLUDE_DIRS}
  ${g2_transport_TASK_INCLUDE_DIRS})

add_executable(tenon_apu_g2_tasks
  ${g2_transport_TASK_SOURCES}
  apu_g2_u16_add.cc
  apu_g2_u16_atax.cc
  apu_g2_u16_block_sum.cc
  apu_g2_u16_div.cc
  apu_g2_u16_dot_tile.cc
  apu_g2_u16_fill.cc
  apu_g2_u16_gemv.cc
  apu_g2_u16_gesummv.cc
  apu_g2_u16_minmax.cc
  apu_g2_u16_mul.cc
  apu_g2_u16_select.cc
  apu_g2_u16_shift_right.cc
  apu_g2_u16_sqrt.cc
  apu_g2_u16_sub.cc
  apu_g2_typed.cc
  apu_g2_composed_dot.cc
  apu_g2_u16_gemm.cc
  ${CMAKE_CURRENT_BINARY_DIR}/gsi_library_entry_points_wrapper.cpp)

include(gsi_generate_gsi_library_entry_point_cxx_function_wrappers)
gsi_system_set_c_and_cxx_compiler_options()

target_include_directories(tenon_apu_g2_tasks PRIVATE
  ${GTML_INCLUDE_DIRS}
  ${GSI_SYSTEM_INCLUDE_DIRS}
  /opt/gsi/include/64vl/g2_64vl
  /opt/gsi/include/g2_common
  /opt/gsi/include/gtml/g2_gtml
  /opt/gsi/include/gsi
  /opt/gsi/include/gsi/device
  ${g2_transport_INCLUDE_DIRS}
  ${g2_transport_TASK_INCLUDE_DIRS})
target_compile_options(tenon_apu_g2_tasks PRIVATE
  ${GSI_SYSTEM_OPT_FLAGS}
  -O3
  -DNDEBUG)
target_compile_definitions(tenon_apu_g2_tasks PRIVATE ARC=1)
target_link_libraries(tenon_apu_g2_tasks PRIVATE GTML::arc_standalone)

include(${GSI_SYSTEM_TARGET_NAME})
include(gsi_linker_product_load_address)
include(gsi_product_extra_defs)

add_custom_command(TARGET tenon_apu_g2_tasks POST_BUILD
  COMMAND python3 ${GTML_MD_UPDATE_SCRIPT}
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tenon_apu_g2_tasks
    ${GSI_SYSTEM_TOOLCHAIN_PREFIX}
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMENT "Patching MD relocations")

include(gsi_generate_gsi_library_from_mod_exe)
gsi_library_from_mod_exe_file(tenon_apu_g2_tasks
  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  tenon_apu_g2_tasks.update tenon_apu_g2_tasks.update.bin)
