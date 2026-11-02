# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(tenon_apu_g2_host LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)

if(NOT DEFINED GSI_TARGET)
  set(GSI_TARGET DEVICE)
endif()
if(NOT GSI_TARGET STREQUAL "DEVICE")
  message(FATAL_ERROR "The Tenon APUg2 runtime is hardware-only")
endif()
if(NOT DEFINED BOARD_DEVICE_LIB)
  message(FATAL_ERROR "BOARD_DEVICE_LIB must name an absolute .update.bin")
endif()

# The runtime passes the values allo.spmw_simenv resolves
# (TENON_APU_G2_SDK_ROOT, TENON_APU_G2_DEVICE); the defaults are the installed
# card's layout.  The GSI_GTML_g2 config package is found by CMake's default
# search, not under this root.
set(TENON_APU_G2_SDK_ROOT "/opt/gsi" CACHE PATH "GSI Gemini-II SDK root")
set(TENON_APU_G2_DEVICE_INDEX "0" CACHE STRING "Gemini-II card index (apu-NN)")

find_package(GSI_GTML_g2 REQUIRED CONFIG)
find_package(g2_device REQUIRED CONFIG
  PATHS ${TENON_APU_G2_SDK_ROOT}/lib/cmake/g2_device NO_DEFAULT_PATH)
find_package(g2_transport REQUIRED CONFIG
  PATHS ${TENON_APU_G2_SDK_ROOT}/lib/cmake/g2_transport NO_DEFAULT_PATH)

# One host executable per public runtime: the uint16 add runner
# (allo.pim.apu_g2_runtime), the typed VL64 programs
# (allo.pim.apu_g2_typed_runtime) and the composed contractions
# (allo.pim.apu_g2_composed_runtime).
function(tenon_apu_g2_host name source)
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/device
    ${TENON_APU_G2_SDK_ROOT}/include/64vl/g2_64vl
    ${TENON_APU_G2_SDK_ROOT}/include/g2_common
    ${TENON_APU_G2_SDK_ROOT}/include/gtml/g2_gtml
    ${TENON_APU_G2_SDK_ROOT}/include/gsi
    ${TENON_APU_G2_SDK_ROOT}/include/gsi/device
    ${g2_transport_INCLUDE_DIRS}
    ${g2_transport_TASK_INCLUDE_DIRS})
  target_compile_definitions(${name} PRIVATE
    DEVICE_SIDE_LIB_LOCATION="${BOARD_DEVICE_LIB}" TENON_APU_G2_HARDWARE=1
    TENON_APU_G2_DEVICE_INDEX=${TENON_APU_G2_DEVICE_INDEX})
  target_link_libraries(${name} PRIVATE g2_device::g2_device)

  # The installed GDML target leaves these shared/static dependencies to the
  # application.  Preserve their order after the imported static device
  # archive.
  target_link_options(${name} PRIVATE -Wl,--no-as-needed)
  target_link_directories(${name} PRIVATE
    ${TENON_APU_G2_SDK_ROOT}/g2/Linux/lib/amd64)
  target_link_libraries(${name} PRIVATE
    gdml gsi_g2_api glog cyclic_buffer pthread)
endfunction()

tenon_apu_g2_host(tenon_apu_g2_u16_add host.cc)
tenon_apu_g2_host(tenon_apu_g2_typed host_typed.cc)
tenon_apu_g2_host(tenon_apu_g2_composed_dot host_composed_dot.cc)
