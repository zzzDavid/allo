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
constexpr uint64_t kPackElements =
    uint64_t{kGroups} * kVectors * kElementsPerGroup;

constexpr uint32_t kLeftRow = 64;
constexpr uint32_t kRightRow = 160;
constexpr uint32_t kAccumulatorRow = 256;
constexpr uint32_t kOutRow = 320;
constexpr uint32_t kTempRow = 2800;
constexpr uint32_t kIndexRow = 2928;

double to_us(Clock::duration value) {
  return std::chrono::duration<double, std::micro>(value).count();
}

std::vector<uint16_t> read_pack(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    throw std::runtime_error("cannot open input " + path);
  const auto bytes = file.tellg();
  if (bytes != static_cast<std::streamoff>(kPackElements * sizeof(uint16_t)))
    throw std::runtime_error("wrong byte count for " + path);
  file.seekg(0);
  std::vector<uint16_t> values(kPackElements);
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
  if (argc != 10) {
    std::cerr << "usage: " << argv[0]
              << " LEFT.bin RIGHT.bin ACCUMULATOR.bin OUT.bin LOG_BLOCK_SIZE"
                 " ALPHA BETA ENABLE_EPILOGUE REPETITIONS\n";
    return 2;
  }

  try {
    const uint64_t parsed_log = std::stoull(argv[5]);
    const uint64_t parsed_alpha = std::stoull(argv[6]);
    const uint64_t parsed_beta = std::stoull(argv[7]);
    const uint64_t parsed_enable_epilogue = std::stoull(argv[8]);
    const uint64_t parsed_repetitions = std::stoull(argv[9]);
    if (parsed_log > 8)
      throw std::invalid_argument("log block size must be in [0, 8]");
    if (parsed_repetitions == 0 || parsed_repetitions > UINT32_MAX)
      throw std::invalid_argument("repetitions must fit nonzero uint32");
    if (parsed_alpha > UINT16_MAX || parsed_beta > UINT16_MAX)
      throw std::invalid_argument("alpha and beta must fit uint16");
    if (parsed_enable_epilogue > 1)
      throw std::invalid_argument("enable epilogue must be zero or one");

    const auto all_start = Clock::now();
    const auto left_values = read_pack(argv[1]);
    const auto right_values = read_pack(argv[2]);
    const auto accumulator_values = read_pack(argv[3]);

    using namespace gtml_transport;
    Device device(DEVICE_SIDE_LIB_LOCATION, DeviceConfig{0, 0});
    device.init_gtml({kTempRow, 128, kIndexRow, 16});

    const gsi::VectorPackDescriptor descriptor(kVectors, kBits,
                                               gsi::G2_TYPES::UINT, kBits);
    const gsi::VectorPack left(descriptor, kLeftRow);
    const gsi::VectorPack right(descriptor, kRightRow);
    const gsi::VectorPack accumulator(descriptor, kAccumulatorRow);
    const gsi::VectorPack out(descriptor, kOutRow);
    g2_data::VectorPackRef left_ref(left, kGroups);
    g2_data::VectorPackRef right_ref(right, kGroups);
    g2_data::VectorPackRef accumulator_ref(accumulator, kGroups);
    g2_data::VectorPackRef out_ref(out, kGroups);
    fill_pack(left_ref, left_values);
    fill_pack(right_ref, right_values);
    fill_pack(accumulator_ref, accumulator_values);

    g2_data::L1ContainerRef index(kIndexRow, 16, kGroups);
    index.fill_idx(16);

    const auto h2d_start = Clock::now();
    copy_to_l1(device, index);
    copy_to_l1(device, left_ref);
    copy_to_l1(device, right_ref);
    copy_to_l1(device, accumulator_ref);
    const auto h2d_stop = Clock::now();

    ApuG2DotTileTimings timings{};
    const uint64_t timings_l5 = device.alloc_l5(sizeof(timings));
    bool timings_l5_live = true;
    try {
      device.mem_to_l5(&timings, sizeof(timings), timings_l5);
      const ApuG2DotTileParams params{
          kLeftRow,
          kRightRow,
          kAccumulatorRow,
          kOutRow,
          static_cast<uint32_t>(parsed_log),
          static_cast<uint32_t>(parsed_repetitions),
          static_cast<uint32_t>(parsed_alpha),
          static_cast<uint32_t>(parsed_beta),
          static_cast<uint32_t>(parsed_enable_epilogue),
          timings_l5};
      const auto task_start = Clock::now();
      const int task_rc = device.run_task("tenon_apu_g2_u16_dot_tile", params);
      const auto task_stop = Clock::now();
      if (task_rc != 0)
        throw std::runtime_error("device task returned " +
                                 std::to_string(task_rc));
      device.l5_to_mem(timings_l5, &timings, sizeof(timings));
      device.free_l5(timings_l5);
      timings_l5_live = false;

      const auto d2h_start = Clock::now();
      copy_from_l1(device, out_ref);
      const auto d2h_stop = Clock::now();
      write_pack(argv[4], out_ref);
      const auto all_stop = Clock::now();

      std::cout << "device_pipeline_ticks=" << timings.pipeline_ticks << '\n'
                << "device_final_pipeline_ticks="
                << timings.final_pipeline_ticks << '\n';
      std::cout << std::fixed << std::setprecision(3)
                << "h2d_us=" << to_us(h2d_stop - h2d_start) << '\n'
                << "host_task_us=" << to_us(task_stop - task_start) << '\n'
                << "d2h_us=" << to_us(d2h_stop - d2h_start) << '\n'
                << "end_to_end_us=" << to_us(all_stop - all_start) << '\n';
      std::cout << "PASS physical_outputs=" << kPackElements << '\n';
      std::cout << "epilogue_enabled=" << parsed_enable_epilogue << '\n';
      return 0;
    } catch (...) {
      if (timings_l5_live)
        device.free_l5(timings_l5);
      throw;
    }
  } catch (const std::exception &error) {
    std::cerr << "APUg2 dot-tile host failure: " << error.what() << '\n';
    return 1;
  }
}
