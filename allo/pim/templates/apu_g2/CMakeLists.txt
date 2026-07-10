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

find_package(GSI_GTML_g2 REQUIRED CONFIG)
find_package(g2_device REQUIRED CONFIG
  PATHS /opt/gsi/lib/cmake/g2_device NO_DEFAULT_PATH)
find_package(g2_transport REQUIRED CONFIG
  PATHS /opt/gsi/lib/cmake/g2_transport NO_DEFAULT_PATH)

add_executable(tenon_apu_g2_u16_add host.cc)
target_include_directories(tenon_apu_g2_u16_add PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/device
  /opt/gsi/include/64vl/g2_64vl
  /opt/gsi/include/g2_common
  /opt/gsi/include/gtml/g2_gtml
  /opt/gsi/include/gsi
  /opt/gsi/include/gsi/device
  ${g2_transport_INCLUDE_DIRS}
  ${g2_transport_TASK_INCLUDE_DIRS})
target_compile_definitions(tenon_apu_g2_u16_add PRIVATE
  DEVICE_SIDE_LIB_LOCATION="${BOARD_DEVICE_LIB}" TENON_APU_G2_HARDWARE=1)
target_link_libraries(tenon_apu_g2_u16_add PRIVATE g2_device::g2_device)

# The installed GDML target leaves these shared/static dependencies to the
# application.  Preserve their order after the imported static device archive.
target_link_options(tenon_apu_g2_u16_add PRIVATE -Wl,--no-as-needed)
target_link_directories(tenon_apu_g2_u16_add PRIVATE
  /opt/gsi/g2/Linux/lib/amd64)
target_link_libraries(tenon_apu_g2_u16_add PRIVATE
  gdml gsi_g2_api glog cyclic_buffer pthread)

add_executable(tenon_apu_g2_u16_gesummv host_gesummv.cc)
target_include_directories(tenon_apu_g2_u16_gesummv PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/device
  /opt/gsi/include/64vl/g2_64vl
  /opt/gsi/include/g2_common
  /opt/gsi/include/gtml/g2_gtml
  /opt/gsi/include/gsi
  /opt/gsi/include/gsi/device
  ${g2_transport_INCLUDE_DIRS}
  ${g2_transport_TASK_INCLUDE_DIRS})
target_compile_definitions(tenon_apu_g2_u16_gesummv PRIVATE
  DEVICE_SIDE_LIB_LOCATION="${BOARD_DEVICE_LIB}" TENON_APU_G2_HARDWARE=1)
target_link_libraries(tenon_apu_g2_u16_gesummv PRIVATE g2_device::g2_device)
target_link_options(tenon_apu_g2_u16_gesummv PRIVATE -Wl,--no-as-needed)
target_link_directories(tenon_apu_g2_u16_gesummv PRIVATE
  /opt/gsi/g2/Linux/lib/amd64)
target_link_libraries(tenon_apu_g2_u16_gesummv PRIVATE
  gdml gsi_g2_api glog cyclic_buffer pthread)

add_executable(tenon_apu_g2_u16_gemv host_gemv.cc)
target_include_directories(tenon_apu_g2_u16_gemv PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/device
  /opt/gsi/include/64vl/g2_64vl
  /opt/gsi/include/g2_common
  /opt/gsi/include/gtml/g2_gtml
  /opt/gsi/include/gsi
  /opt/gsi/include/gsi/device
  ${g2_transport_INCLUDE_DIRS}
  ${g2_transport_TASK_INCLUDE_DIRS})
target_compile_definitions(tenon_apu_g2_u16_gemv PRIVATE
  DEVICE_SIDE_LIB_LOCATION="${BOARD_DEVICE_LIB}" TENON_APU_G2_HARDWARE=1)
target_link_libraries(tenon_apu_g2_u16_gemv PRIVATE g2_device::g2_device)
target_link_options(tenon_apu_g2_u16_gemv PRIVATE -Wl,--no-as-needed)
target_link_directories(tenon_apu_g2_u16_gemv PRIVATE
  /opt/gsi/g2/Linux/lib/amd64)
target_link_libraries(tenon_apu_g2_u16_gemv PRIVATE
  gdml gsi_g2_api glog cyclic_buffer pthread)

add_executable(tenon_apu_g2_u16_atax host_atax.cc)
target_include_directories(tenon_apu_g2_u16_atax PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/device
  /opt/gsi/include/64vl/g2_64vl
  /opt/gsi/include/g2_common
  /opt/gsi/include/gtml/g2_gtml
  /opt/gsi/include/gsi
  /opt/gsi/include/gsi/device
  ${g2_transport_INCLUDE_DIRS}
  ${g2_transport_TASK_INCLUDE_DIRS})
target_compile_definitions(tenon_apu_g2_u16_atax PRIVATE
  DEVICE_SIDE_LIB_LOCATION="${BOARD_DEVICE_LIB}" TENON_APU_G2_HARDWARE=1)
target_link_libraries(tenon_apu_g2_u16_atax PRIVATE g2_device::g2_device)
target_link_options(tenon_apu_g2_u16_atax PRIVATE -Wl,--no-as-needed)
target_link_directories(tenon_apu_g2_u16_atax PRIVATE
  /opt/gsi/g2/Linux/lib/amd64)
target_link_libraries(tenon_apu_g2_u16_atax PRIVATE
  gdml gsi_g2_api glog cyclic_buffer pthread)

add_executable(tenon_apu_g2_u16_block_sum host_block_sum.cc)
target_include_directories(tenon_apu_g2_u16_block_sum PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/device
  /opt/gsi/include/64vl/g2_64vl
  /opt/gsi/include/g2_common
  /opt/gsi/include/gtml/g2_gtml
  /opt/gsi/include/gsi
  /opt/gsi/include/gsi/device
  ${g2_transport_INCLUDE_DIRS}
  ${g2_transport_TASK_INCLUDE_DIRS})
target_compile_definitions(tenon_apu_g2_u16_block_sum PRIVATE
  DEVICE_SIDE_LIB_LOCATION="${BOARD_DEVICE_LIB}" TENON_APU_G2_HARDWARE=1)
target_link_libraries(tenon_apu_g2_u16_block_sum PRIVATE g2_device::g2_device)
target_link_options(tenon_apu_g2_u16_block_sum PRIVATE -Wl,--no-as-needed)
target_link_directories(tenon_apu_g2_u16_block_sum PRIVATE
  /opt/gsi/g2/Linux/lib/amd64)
target_link_libraries(tenon_apu_g2_u16_block_sum PRIVATE
  gdml gsi_g2_api glog cyclic_buffer pthread)

add_executable(tenon_apu_g2_u16_dot_tile host_dot_tile.cc)
target_include_directories(tenon_apu_g2_u16_dot_tile PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/device
  /opt/gsi/include/64vl/g2_64vl
  /opt/gsi/include/g2_common
  /opt/gsi/include/gtml/g2_gtml
  /opt/gsi/include/gsi
  /opt/gsi/include/gsi/device
  ${g2_transport_INCLUDE_DIRS}
  ${g2_transport_TASK_INCLUDE_DIRS})
target_compile_definitions(tenon_apu_g2_u16_dot_tile PRIVATE
  DEVICE_SIDE_LIB_LOCATION="${BOARD_DEVICE_LIB}" TENON_APU_G2_HARDWARE=1)
target_link_libraries(tenon_apu_g2_u16_dot_tile PRIVATE g2_device::g2_device)
target_link_options(tenon_apu_g2_u16_dot_tile PRIVATE -Wl,--no-as-needed)
target_link_directories(tenon_apu_g2_u16_dot_tile PRIVATE
  /opt/gsi/g2/Linux/lib/amd64)
target_link_libraries(tenon_apu_g2_u16_dot_tile PRIVATE
  gdml gsi_g2_api glog cyclic_buffer pthread)

add_executable(tenon_apu_g2_u16_gemm host_u16_gemm.cc)
target_include_directories(tenon_apu_g2_u16_gemm PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/device
  /opt/gsi/include/64vl/g2_64vl
  /opt/gsi/include/g2_common
  /opt/gsi/include/gtml/g2_gtml
  /opt/gsi/include/gsi
  /opt/gsi/include/gsi/device
  ${g2_transport_INCLUDE_DIRS}
  ${g2_transport_TASK_INCLUDE_DIRS})
target_compile_definitions(tenon_apu_g2_u16_gemm PRIVATE
  DEVICE_SIDE_LIB_LOCATION="${BOARD_DEVICE_LIB}" TENON_APU_G2_HARDWARE=1)
target_link_libraries(tenon_apu_g2_u16_gemm PRIVATE g2_device::g2_device)
target_link_options(tenon_apu_g2_u16_gemm PRIVATE -Wl,--no-as-needed)
target_link_directories(tenon_apu_g2_u16_gemm PRIVATE
  /opt/gsi/g2/Linux/lib/amd64)
target_link_libraries(tenon_apu_g2_u16_gemm PRIVATE
  gdml gsi_g2_api glog cyclic_buffer pthread)

add_executable(tenon_apu_g2_u16_fill host_fill.cc)
target_include_directories(tenon_apu_g2_u16_fill PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/device
  /opt/gsi/include/64vl/g2_64vl
  /opt/gsi/include/g2_common
  /opt/gsi/include/gtml/g2_gtml
  /opt/gsi/include/gsi
  /opt/gsi/include/gsi/device
  ${g2_transport_INCLUDE_DIRS}
  ${g2_transport_TASK_INCLUDE_DIRS})
target_compile_definitions(tenon_apu_g2_u16_fill PRIVATE
  DEVICE_SIDE_LIB_LOCATION="${BOARD_DEVICE_LIB}" TENON_APU_G2_HARDWARE=1)
target_link_libraries(tenon_apu_g2_u16_fill PRIVATE g2_device::g2_device)
target_link_options(tenon_apu_g2_u16_fill PRIVATE -Wl,--no-as-needed)
target_link_directories(tenon_apu_g2_u16_fill PRIVATE
  /opt/gsi/g2/Linux/lib/amd64)
target_link_libraries(tenon_apu_g2_u16_fill PRIVATE
  gdml gsi_g2_api glog cyclic_buffer pthread)

add_executable(tenon_apu_g2_u16_select_lt host_select.cc)
target_include_directories(tenon_apu_g2_u16_select_lt PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/device
  /opt/gsi/include/64vl/g2_64vl
  /opt/gsi/include/g2_common
  /opt/gsi/include/gtml/g2_gtml
  /opt/gsi/include/gsi
  /opt/gsi/include/gsi/device
  ${g2_transport_INCLUDE_DIRS}
  ${g2_transport_TASK_INCLUDE_DIRS})
target_compile_definitions(tenon_apu_g2_u16_select_lt PRIVATE
  DEVICE_SIDE_LIB_LOCATION="${BOARD_DEVICE_LIB}" TENON_APU_G2_HARDWARE=1)
target_link_libraries(tenon_apu_g2_u16_select_lt PRIVATE g2_device::g2_device)
target_link_options(tenon_apu_g2_u16_select_lt PRIVATE -Wl,--no-as-needed)
target_link_directories(tenon_apu_g2_u16_select_lt PRIVATE
  /opt/gsi/g2/Linux/lib/amd64)
target_link_libraries(tenon_apu_g2_u16_select_lt PRIVATE
  gdml gsi_g2_api glog cyclic_buffer pthread)

add_executable(tenon_apu_g2_u16_shift_right host_shift_right.cc)
target_include_directories(tenon_apu_g2_u16_shift_right PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/device
  /opt/gsi/include/64vl/g2_64vl
  /opt/gsi/include/g2_common
  /opt/gsi/include/gtml/g2_gtml
  /opt/gsi/include/gsi
  /opt/gsi/include/gsi/device
  ${g2_transport_INCLUDE_DIRS}
  ${g2_transport_TASK_INCLUDE_DIRS})
target_compile_definitions(tenon_apu_g2_u16_shift_right PRIVATE
  DEVICE_SIDE_LIB_LOCATION="${BOARD_DEVICE_LIB}" TENON_APU_G2_HARDWARE=1)
target_link_libraries(tenon_apu_g2_u16_shift_right PRIVATE g2_device::g2_device)
target_link_options(tenon_apu_g2_u16_shift_right PRIVATE -Wl,--no-as-needed)
target_link_directories(tenon_apu_g2_u16_shift_right PRIVATE
  /opt/gsi/g2/Linux/lib/amd64)
target_link_libraries(tenon_apu_g2_u16_shift_right PRIVATE
  gdml gsi_g2_api glog cyclic_buffer pthread)

add_executable(tenon_apu_g2_u16_sqrt host_sqrt.cc)
target_include_directories(tenon_apu_g2_u16_sqrt PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/device
  /opt/gsi/include/64vl/g2_64vl
  /opt/gsi/include/g2_common
  /opt/gsi/include/gtml/g2_gtml
  /opt/gsi/include/gsi
  /opt/gsi/include/gsi/device
  ${g2_transport_INCLUDE_DIRS}
  ${g2_transport_TASK_INCLUDE_DIRS})
target_compile_definitions(tenon_apu_g2_u16_sqrt PRIVATE
  DEVICE_SIDE_LIB_LOCATION="${BOARD_DEVICE_LIB}" TENON_APU_G2_HARDWARE=1)
target_link_libraries(tenon_apu_g2_u16_sqrt PRIVATE g2_device::g2_device)
target_link_options(tenon_apu_g2_u16_sqrt PRIVATE -Wl,--no-as-needed)
target_link_directories(tenon_apu_g2_u16_sqrt PRIVATE
  /opt/gsi/g2/Linux/lib/amd64)
target_link_libraries(tenon_apu_g2_u16_sqrt PRIVATE
  gdml gsi_g2_api glog cyclic_buffer pthread)

add_executable(tenon_apu_g2_u16_minmax host_minmax.cc)
target_include_directories(tenon_apu_g2_u16_minmax PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/device
  /opt/gsi/include/64vl/g2_64vl
  /opt/gsi/include/g2_common
  /opt/gsi/include/gtml/g2_gtml
  /opt/gsi/include/gsi
  /opt/gsi/include/gsi/device
  ${g2_transport_INCLUDE_DIRS}
  ${g2_transport_TASK_INCLUDE_DIRS})
target_compile_definitions(tenon_apu_g2_u16_minmax PRIVATE
  DEVICE_SIDE_LIB_LOCATION="${BOARD_DEVICE_LIB}" TENON_APU_G2_HARDWARE=1)
target_link_libraries(tenon_apu_g2_u16_minmax PRIVATE g2_device::g2_device)
target_link_options(tenon_apu_g2_u16_minmax PRIVATE -Wl,--no-as-needed)
target_link_directories(tenon_apu_g2_u16_minmax PRIVATE
  /opt/gsi/g2/Linux/lib/amd64)
target_link_libraries(tenon_apu_g2_u16_minmax PRIVATE
  gdml gsi_g2_api glog cyclic_buffer pthread)

add_executable(tenon_apu_g2_u16_mul host_mul.cc)
target_include_directories(tenon_apu_g2_u16_mul PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/device
  /opt/gsi/include/64vl/g2_64vl
  /opt/gsi/include/g2_common
  /opt/gsi/include/gtml/g2_gtml
  /opt/gsi/include/gsi
  /opt/gsi/include/gsi/device
  ${g2_transport_INCLUDE_DIRS}
  ${g2_transport_TASK_INCLUDE_DIRS})
target_compile_definitions(tenon_apu_g2_u16_mul PRIVATE
  DEVICE_SIDE_LIB_LOCATION="${BOARD_DEVICE_LIB}" TENON_APU_G2_HARDWARE=1)
target_link_libraries(tenon_apu_g2_u16_mul PRIVATE g2_device::g2_device)
target_link_options(tenon_apu_g2_u16_mul PRIVATE -Wl,--no-as-needed)
target_link_directories(tenon_apu_g2_u16_mul PRIVATE
  /opt/gsi/g2/Linux/lib/amd64)
target_link_libraries(tenon_apu_g2_u16_mul PRIVATE
  gdml gsi_g2_api glog cyclic_buffer pthread)

add_executable(tenon_apu_g2_u16_div host_div.cc)
target_include_directories(tenon_apu_g2_u16_div PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/device
  /opt/gsi/include/64vl/g2_64vl
  /opt/gsi/include/g2_common
  /opt/gsi/include/gtml/g2_gtml
  /opt/gsi/include/gsi
  /opt/gsi/include/gsi/device
  ${g2_transport_INCLUDE_DIRS}
  ${g2_transport_TASK_INCLUDE_DIRS})
target_compile_definitions(tenon_apu_g2_u16_div PRIVATE
  DEVICE_SIDE_LIB_LOCATION="${BOARD_DEVICE_LIB}" TENON_APU_G2_HARDWARE=1)
target_link_libraries(tenon_apu_g2_u16_div PRIVATE g2_device::g2_device)
target_link_options(tenon_apu_g2_u16_div PRIVATE -Wl,--no-as-needed)
target_link_directories(tenon_apu_g2_u16_div PRIVATE
  /opt/gsi/g2/Linux/lib/amd64)
target_link_libraries(tenon_apu_g2_u16_div PRIVATE
  gdml gsi_g2_api glog cyclic_buffer pthread)

add_executable(tenon_apu_g2_u16_sub host_sub.cc)
target_include_directories(tenon_apu_g2_u16_sub PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/device
  /opt/gsi/include/64vl/g2_64vl
  /opt/gsi/include/g2_common
  /opt/gsi/include/gtml/g2_gtml
  /opt/gsi/include/gsi
  /opt/gsi/include/gsi/device
  ${g2_transport_INCLUDE_DIRS}
  ${g2_transport_TASK_INCLUDE_DIRS})
target_compile_definitions(tenon_apu_g2_u16_sub PRIVATE
  DEVICE_SIDE_LIB_LOCATION="${BOARD_DEVICE_LIB}" TENON_APU_G2_HARDWARE=1)
target_link_libraries(tenon_apu_g2_u16_sub PRIVATE g2_device::g2_device)
target_link_options(tenon_apu_g2_u16_sub PRIVATE -Wl,--no-as-needed)
target_link_directories(tenon_apu_g2_u16_sub PRIVATE
  /opt/gsi/g2/Linux/lib/amd64)
target_link_libraries(tenon_apu_g2_u16_sub PRIVATE
  gdml gsi_g2_api glog cyclic_buffer pthread)
