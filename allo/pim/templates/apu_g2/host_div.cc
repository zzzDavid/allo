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
constexpr uint64_t kElementCount =
    uint64_t{kGroups} * kVectors * kElementsPerGroup;

constexpr uint32_t kLhsRow = 64;
constexpr uint32_t kRhsRow = 256;
constexpr uint32_t kOutRow = 448;
constexpr uint32_t kTempRow = 2800;
constexpr uint32_t kIndexRow = 2928;

double to_us(Clock::duration value) {
  return std::chrono::duration<double, std::micro>(value).count();
}

std::vector<uint16_t> read_u16_file(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    throw std::runtime_error("cannot open input " + path);
  const auto bytes = file.tellg();
  if (bytes != static_cast<std::streamoff>(kElementCount * sizeof(uint16_t)))
    throw std::runtime_error("wrong byte count for " + path);
  file.seekg(0);
  std::vector<uint16_t> values(kElementCount);
  if (!file.read(reinterpret_cast<char *>(values.data()), bytes))
    throw std::runtime_error("cannot read input " + path);
  return values;
}

void write_u16_file(const std::string &path,
                    const g2_data::VectorPackRef &ref) {
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

void fill_ref(g2_data::VectorPackRef &ref,
              const std::vector<uint16_t> &values) {
  uint64_t offset = 0;
  for (uint32_t vector = 0; vector < kVectors; ++vector)
    for (uint32_t group = 0; group < kGroups; ++group)
      for (uint32_t element = 0; element < kElementsPerGroup; ++element)
        ref.ivecs[group][vector][element] = values[offset++];
}

void verify(const g2_data::VectorPackRef &out, const std::vector<uint16_t> &lhs,
            const std::vector<uint16_t> &rhs) {
  uint64_t offset = 0;
  for (uint32_t vector = 0; vector < kVectors; ++vector) {
    for (uint32_t group = 0; group < kGroups; ++group) {
      for (uint32_t element = 0; element < kElementsPerGroup;
           ++element, ++offset) {
        if (rhs[offset] == 0)
          throw std::runtime_error("zero divisor reached host verifier");
        const uint16_t expected =
            static_cast<uint16_t>(lhs[offset] / rhs[offset]);
        const uint16_t actual =
            static_cast<uint16_t>(out.ivecs[group][vector][element]);
        if (actual != expected) {
          throw std::runtime_error(
              "hardware mismatch at vector=" + std::to_string(vector) +
              " group=" + std::to_string(group) +
              " element=" + std::to_string(element) + " expected=" +
              std::to_string(expected) + " actual=" + std::to_string(actual));
        }
      }
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: " << argv[0]
              << " LHS.bin RHS.bin OUT.bin REPETITIONS\n";
    return 2;
  }

  try {
    const uint64_t parsed_repetitions = std::stoull(argv[4]);
    if (parsed_repetitions == 0 || parsed_repetitions > UINT32_MAX)
      throw std::invalid_argument("repetitions must fit nonzero uint32");
    const uint32_t repetitions = static_cast<uint32_t>(parsed_repetitions);

    const auto all_start = Clock::now();
    const auto lhs_values = read_u16_file(argv[1]);
    const auto rhs_values = read_u16_file(argv[2]);
    for (const uint16_t value : rhs_values)
      if (value == 0)
        throw std::invalid_argument("rhs contains zero divisor");

    using namespace gtml_transport;
    Device device(DEVICE_SIDE_LIB_LOCATION, DeviceConfig{0, 0});
    device.init_gtml({kTempRow, 128, kIndexRow, 16});

    const gsi::VectorPackDescriptor descriptor(kVectors, kBits,
                                               gsi::G2_TYPES::UINT, kBits);
    const gsi::VectorPack lhs(descriptor, kLhsRow);
    const gsi::VectorPack rhs(descriptor, kRhsRow);
    const gsi::VectorPack out(descriptor, kOutRow);
    g2_data::VectorPackRef lhs_ref(lhs, kGroups);
    g2_data::VectorPackRef rhs_ref(rhs, kGroups);
    g2_data::VectorPackRef out_ref(out, kGroups);
    fill_ref(lhs_ref, lhs_values);
    fill_ref(rhs_ref, rhs_values);

    g2_data::L1ContainerRef index(kIndexRow, 16, kGroups);
    index.fill_idx(16);

    const auto h2d_start = Clock::now();
    copy_to_l1(device, index);
    copy_to_l1(device, lhs_ref);
    copy_to_l1(device, rhs_ref);
    const auto h2d_stop = Clock::now();

    ApuG2DivTimings timings{};
    const uint64_t timings_l5 = device.alloc_l5(sizeof(timings));
    bool timings_l5_live = true;
    try {
      device.mem_to_l5(&timings, sizeof(timings), timings_l5);
      const ApuG2DivParams params{kLhsRow, kRhsRow, kOutRow, repetitions,
                                  timings_l5};
      const auto task_start = Clock::now();
      const int task_rc = device.run_task("tenon_apu_g2_u16_div", params);
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
      verify(out_ref, lhs_values, rhs_values);
      write_u16_file(argv[3], out_ref);
      const auto all_stop = Clock::now();

      std::cout << "device_pipeline_ticks=" << timings.pipeline_ticks << '\n';
      std::cout << "device_final_pipeline_ticks="
                << timings.final_pipeline_ticks << '\n';
      std::cout << std::fixed << std::setprecision(3)
                << "h2d_us=" << to_us(h2d_stop - h2d_start) << '\n'
                << "host_task_us=" << to_us(task_stop - task_start) << '\n'
                << "d2h_us=" << to_us(d2h_stop - d2h_start) << '\n'
                << "end_to_end_us=" << to_us(all_stop - all_start) << '\n';
      std::cout << "PASS checked=" << kElementCount << '\n';
      return 0;
    } catch (...) {
      if (timings_l5_live)
        device.free_l5(timings_l5);
      throw;
    }
  } catch (const std::exception &error) {
    std::cerr << "APUg2 div host failure: " << error.what() << '\n';
    return 1;
  }
}
