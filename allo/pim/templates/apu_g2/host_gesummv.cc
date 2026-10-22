// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <copy.h>
#include <device.h>
#include <g2_data_types.h>

#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "apu_g2_params.h"

#ifndef DEVICE_SIDE_LIB_LOCATION
#error DEVICE_SIDE_LIB_LOCATION must name the APUg2 task .update.bin
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint32_t kGroups = 16;
constexpr uint32_t kVectors = 4;
constexpr uint32_t kBits = 16;
constexpr uint32_t kElementsPerGroup = 4096;
constexpr uint64_t kVectorLanes = uint64_t{kGroups} * kElementsPerGroup;
constexpr uint64_t kPackElements = kVectors * kVectorLanes;

constexpr uint32_t kMatricesRow = 64;
constexpr uint32_t kXRow = 160;
constexpr uint32_t kScalarsRow = 192;
constexpr uint32_t kTmpRow = 320;
constexpr uint32_t kYRow = 448;
constexpr uint32_t kTempRow = 2800;
constexpr uint32_t kIndexRow = 2928;

double to_us(Clock::duration value) {
  return std::chrono::duration<double, std::micro>(value).count();
}

std::vector<uint16_t> read_u16_file(const std::string &path,
                                    uint64_t element_count) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    throw std::runtime_error("cannot open input " + path);
  const auto bytes = file.tellg();
  const auto expected = static_cast<std::streamoff>(element_count * 2);
  if (bytes != expected)
    throw std::runtime_error("wrong byte count for " + path);
  file.seekg(0);
  std::vector<uint16_t> values(element_count);
  if (!file.read(reinterpret_cast<char *>(values.data()), bytes))
    throw std::runtime_error("cannot read input " + path);
  return values;
}

void fill_pack(g2_data::VectorPackRef &ref,
               const std::vector<uint16_t> &values) {
  uint64_t offset = 0;
  for (uint32_t vector = 0; vector < kVectors; ++vector)
    for (uint32_t group = 0; group < kGroups; ++group)
      for (uint32_t element = 0; element < kElementsPerGroup; ++element)
        ref.ivecs[group][vector][element] = values[offset++];
}

void fill_vector(g2_data::VectorRef &ref, const std::vector<uint16_t> &values) {
  uint64_t offset = 0;
  for (uint32_t group = 0; group < kGroups; ++group)
    for (uint32_t element = 0; element < kElementsPerGroup; ++element)
      ref.ivecs[group][0][element] = values[offset++];
}

void write_pack(const std::string &path, const g2_data::VectorPackRef &ref) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file)
    throw std::runtime_error("cannot open output " + path);
  for (uint32_t vector = 0; vector < kVectors; ++vector)
    for (uint32_t group = 0; group < kGroups; ++group)
      for (uint32_t element = 0; element < kElementsPerGroup; ++element) {
        const uint16_t value =
            static_cast<uint16_t>(ref.ivecs[group][vector][element]);
        file.write(reinterpret_cast<const char *>(&value), sizeof(value));
      }
  if (!file)
    throw std::runtime_error("cannot write output " + path);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 8) {
    std::cerr << "usage: " << argv[0]
              << " MATRICES.bin X.bin SCALARS.bin TMP.bin Y.bin"
                 " LOG_BLOCK_SIZE REPETITIONS\n";
    return 2;
  }

  try {
    const uint64_t parsed_log = std::stoull(argv[6]);
    const uint64_t parsed_repetitions = std::stoull(argv[7]);
    if (parsed_log > 8)
      throw std::invalid_argument("log block size must be in [0, 8]");
    if (parsed_repetitions == 0 || parsed_repetitions > UINT32_MAX)
      throw std::invalid_argument("repetitions must fit nonzero uint32");

    const auto all_start = Clock::now();
    const auto matrix_values = read_u16_file(argv[1], kPackElements);
    const auto x_values = read_u16_file(argv[2], kVectorLanes);
    const auto scalar_values = read_u16_file(argv[3], kPackElements);

    using namespace gtml_transport;
    Device device(DEVICE_SIDE_LIB_LOCATION, DeviceConfig{0, 0});
    device.init_gtml({kTempRow, 128, kIndexRow, 16});

    const gsi::VectorPackDescriptor descriptor(kVectors, kBits,
                                               gsi::G2_TYPES::UINT, kBits);
    const gsi::VectorPack matrices(descriptor, kMatricesRow);
    const gsi::Vector x(kBits, gsi::G2_TYPES::UINT, kXRow);
    const gsi::VectorPack scalars(descriptor, kScalarsRow);
    const gsi::VectorPack tmp(descriptor, kTmpRow);
    const gsi::VectorPack y(descriptor, kYRow);
    g2_data::VectorPackRef matrices_ref(matrices, kGroups);
    g2_data::VectorRef x_ref(x, kGroups);
    g2_data::VectorPackRef scalars_ref(scalars, kGroups);
    g2_data::VectorPackRef tmp_ref(tmp, kGroups);
    g2_data::VectorPackRef y_ref(y, kGroups);
    fill_pack(matrices_ref, matrix_values);
    fill_vector(x_ref, x_values);
    fill_pack(scalars_ref, scalar_values);

    g2_data::L1ContainerRef index(kIndexRow, 16, kGroups);
    index.fill_idx(16);

    const auto h2d_start = Clock::now();
    copy_to_l1(device, index);
    copy_to_l1(device, matrices_ref);
    copy_to_l1(device, x_ref);
    copy_to_l1(device, scalars_ref);
    const auto h2d_stop = Clock::now();

    ApuG2GesummvTimings timings{};
    const uint64_t timings_l5 = device.alloc_l5(sizeof(timings));
    bool timings_l5_live = true;
    try {
      device.mem_to_l5(&timings, sizeof(timings), timings_l5);
      const ApuG2GesummvParams params{kMatricesRow,
                                      kXRow,
                                      kScalarsRow,
                                      kTmpRow,
                                      kYRow,
                                      static_cast<uint32_t>(parsed_log),
                                      static_cast<uint32_t>(parsed_repetitions),
                                      timings_l5};
      const auto task_start = Clock::now();
      const int task_rc = device.run_task("tenon_apu_g2_u16_gesummv", params);
      const auto task_stop = Clock::now();
      if (task_rc != 0)
        throw std::runtime_error("device task returned " +
                                 std::to_string(task_rc));
      device.l5_to_mem(timings_l5, &timings, sizeof(timings));
      device.free_l5(timings_l5);
      timings_l5_live = false;

      const auto d2h_start = Clock::now();
      copy_from_l1(device, tmp_ref);
      copy_from_l1(device, y_ref);
      const auto d2h_stop = Clock::now();
      write_pack(argv[4], tmp_ref);
      write_pack(argv[5], y_ref);
      const auto all_stop = Clock::now();

      std::cout << "device_gemv_ticks=" << timings.gemv_ticks << '\n'
                << "device_scale_combine_ticks=" << timings.scale_combine_ticks
                << '\n'
                << "device_pipeline_ticks=" << timings.pipeline_ticks << '\n'
                << "device_final_pipeline_ticks="
                << timings.final_pipeline_ticks << '\n';
      std::cout << std::fixed << std::setprecision(3)
                << "h2d_us=" << to_us(h2d_stop - h2d_start) << '\n'
                << "host_task_us=" << to_us(task_stop - task_start) << '\n'
                << "d2h_us=" << to_us(d2h_stop - d2h_start) << '\n'
                << "end_to_end_us=" << to_us(all_stop - all_start) << '\n';
      std::cout << "PASS physical_outputs=" << 2 * kPackElements << '\n';
      return 0;
    } catch (...) {
      if (timings_l5_live)
        device.free_l5(timings_l5);
      throw;
    }
  } catch (const std::exception &error) {
    std::cerr << "APUg2 GESUMMV host failure: " << error.what() << '\n';
    return 1;
  }
}
