// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <copy.h>
#include <device.h>
#include <g2_data_types.h>

#include <algorithm>
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
constexpr uint32_t kColumnsPerGroup = 4096;
constexpr uint32_t kBits = 16;
constexpr uint32_t kAccumulatorRow = 64;
constexpr uint32_t kWeightRow = 640;
constexpr uint32_t kProductRow = 2704;
constexpr uint32_t kTempRow = 2800;
constexpr uint32_t kIndexRow = 2928;

double to_us(Clock::duration value) {
  return std::chrono::duration<double, std::micro>(value).count();
}

std::vector<uint16_t> read_u16(const std::string &path, uint64_t elements) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    throw std::runtime_error("cannot open input " + path);
  const auto bytes = file.tellg();
  const auto expected = static_cast<std::streamoff>(elements * sizeof(uint16_t));
  if (bytes != expected)
    throw std::runtime_error("wrong byte count for " + path);
  file.seekg(0);
  std::vector<uint16_t> values(elements);
  if (!file.read(reinterpret_cast<char *>(values.data()), bytes))
    throw std::runtime_error("cannot read input " + path);
  return values;
}

void write_u16(const std::string &path, const std::vector<uint16_t> &values) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file)
    throw std::runtime_error("cannot open output " + path);
  file.write(reinterpret_cast<const char *>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(uint16_t)));
  if (!file)
    throw std::runtime_error("cannot write output " + path);
}

uint32_t parse_extent(const char *text, const char *name) {
  const uint64_t value = std::stoull(text);
  if (value == 0 || value > UINT32_MAX)
    throw std::invalid_argument(std::string(name) + " must fit nonzero uint32");
  return static_cast<uint32_t>(value);
}

uint16_t parse_u16(const char *text, const char *name) {
  const uint64_t value = std::stoull(text);
  if (value > UINT16_MAX)
    throw std::invalid_argument(std::string(name) + " must fit uint16");
  return static_cast<uint16_t>(value);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 11) {
    std::cerr << "usage: " << argv[0]
              << " A.bin B.bin C.bin OUT.bin M K N ALPHA BETA BCHUNK\n";
    return 2;
  }

  try {
    const uint32_t M = parse_extent(argv[5], "M");
    const uint32_t K = parse_extent(argv[6], "K");
    const uint32_t N = parse_extent(argv[7], "N");
    const uint16_t alpha = parse_u16(argv[8], "alpha");
    const uint16_t beta = parse_u16(argv[9], "beta");
    const uint32_t bchunk = parse_extent(argv[10], "BCHUNK");
    if (M > kGroups * kColumnsPerGroup)
      throw std::invalid_argument("M exceeds one VL64 vector");
    if (bchunk > APUG2_U16_GEMM_BMAX)
      throw std::invalid_argument("BCHUNK exceeds the L1-resident maximum");

    const auto all_start = Clock::now();
    const auto A = read_u16(argv[1], uint64_t{M} * K);
    const auto B = read_u16(argv[2], uint64_t{K} * N);
    const auto C = read_u16(argv[3], uint64_t{M} * N);
    std::vector<uint16_t> output(uint64_t{M} * N);

    using namespace gtml_transport;
    Device device(DEVICE_SIDE_LIB_LOCATION, DeviceConfig{0, 0});
    device.init_gtml({kTempRow, 128, kIndexRow, 16});

    const uint32_t groups = (M + kColumnsPerGroup - 1) / kColumnsPerGroup;
    g2_data::L1ContainerRef index(kIndexRow, 16, groups);
    index.fill_idx(16);

    g2_data::VectorPackRef weights(
        gsi::VectorPackDescriptor{APUG2_U16_GEMM_CHUNK, kBits,
                                  gsi::G2_TYPES::UINT, kBits},
        kWeightRow, groups);

    ApuG2U16GemmTimings timings{};
    const uint64_t timings_l5 = device.alloc_l5(sizeof(timings));
    bool timings_live = true;
    double h2d_us = 0.0;
    double host_task_us = 0.0;
    double d2h_us = 0.0;
    uint64_t device_ticks = 0;
    uint64_t task_count = 0;
    uint64_t weight_uploads = 0;

    try {
      auto start = Clock::now();
      copy_to_l1(device, index);
      h2d_us += to_us(Clock::now() - start);

      for (uint32_t column_begin = 0; column_begin < N;
           column_begin += bchunk) {
        const uint32_t active_columns =
            std::min(bchunk, N - column_begin);
        g2_data::VectorPackRef accumulators(
            gsi::VectorPackDescriptor{active_columns, kBits,
                                      gsi::G2_TYPES::UINT, kBits},
            kAccumulatorRow, groups);

        for (uint32_t group = 0; group < groups; ++group)
          for (uint32_t column = 0; column < active_columns; ++column)
            for (uint32_t lane = 0; lane < kColumnsPerGroup; ++lane) {
              const uint32_t row = group * kColumnsPerGroup + lane;
              const uint16_t value =
                  row < M
                      ? static_cast<uint16_t>(
                            uint32_t{beta} * C[uint64_t{row} * N +
                                                column_begin + column])
                      : uint16_t{0};
              accumulators.ivecs[group][column][lane] = value;
            }

        start = Clock::now();
        copy_to_l1(device, accumulators);
        h2d_us += to_us(Clock::now() - start);

        for (uint32_t reduction_begin = 0; reduction_begin < K;
             reduction_begin += APUG2_U16_GEMM_CHUNK) {
          const uint32_t reduction_count = std::min(
              APUG2_U16_GEMM_CHUNK, K - reduction_begin);
          for (uint32_t group = 0; group < groups; ++group)
            for (uint32_t reduction = 0;
                 reduction < APUG2_U16_GEMM_CHUNK; ++reduction)
              for (uint32_t lane = 0; lane < kColumnsPerGroup; ++lane) {
                const uint32_t row = group * kColumnsPerGroup + lane;
                const uint32_t depth = reduction_begin + reduction;
                weights.ivecs[group][reduction][lane] =
                    row < M && reduction < reduction_count
                        ? A[uint64_t{row} * K + depth]
                        : uint16_t{0};
              }

          start = Clock::now();
          copy_to_l1(device, weights);
          h2d_us += to_us(Clock::now() - start);
          ++weight_uploads;

          ApuG2U16GemmParams params{};
          params.accumulator_l1_row = kAccumulatorRow;
          params.weight_l1_row = kWeightRow;
          params.product_l1_row = kProductRow;
          params.num_columns = active_columns;
          params.reduction_count = reduction_count;
          params.timings_l5 = timings_l5;
          for (uint32_t column = 0; column < active_columns; ++column)
            for (uint32_t reduction = 0; reduction < reduction_count;
                 ++reduction) {
              const uint16_t rhs =
                  B[uint64_t{reduction_begin + reduction} * N +
                    column_begin + column];
              params.scalars[column * APUG2_U16_GEMM_CHUNK + reduction] =
                  static_cast<uint16_t>(uint32_t{alpha} * rhs);
            }

          timings = {};
          device.mem_to_l5(&timings, sizeof(timings), timings_l5);
          start = Clock::now();
          const int rc =
              device.run_task("tenon_apu_g2_u16_gemm_chunk", params);
          host_task_us += to_us(Clock::now() - start);
          if (rc != 0)
            throw std::runtime_error("device task returned " +
                                     std::to_string(rc));
          device.l5_to_mem(timings_l5, &timings, sizeof(timings));
          if (timings.pipeline_ticks == 0)
            throw std::runtime_error("device task returned zero pipeline ticks");
          device_ticks += timings.pipeline_ticks;
          ++task_count;
        }

        start = Clock::now();
        copy_from_l1(device, accumulators);
        d2h_us += to_us(Clock::now() - start);
        for (uint32_t row = 0; row < M; ++row) {
          const uint32_t group = row / kColumnsPerGroup;
          const uint32_t lane = row % kColumnsPerGroup;
          for (uint32_t column = 0; column < active_columns; ++column)
            output[uint64_t{row} * N + column_begin + column] =
                static_cast<uint16_t>(
                    accumulators.ivecs[group][column][lane]);
        }
      }

      device.free_l5(timings_l5);
      timings_live = false;
      write_u16(argv[4], output);
      const auto all_stop = Clock::now();

      std::cout << "device_pipeline_ticks=" << device_ticks << '\n'
                << "hardware_tasks=" << task_count << '\n'
                << "weight_uploads=" << weight_uploads << '\n';
      std::cout << std::fixed << std::setprecision(3)
                << "h2d_us=" << h2d_us << '\n'
                << "host_task_us=" << host_task_us << '\n'
                << "d2h_us=" << d2h_us << '\n'
                << "end_to_end_us=" << to_us(all_stop - all_start) << '\n';
      std::cout << "PASS logical_outputs=" << uint64_t{M} * N << '\n';
      return 0;
    } catch (...) {
      if (timings_live)
        device.free_l5(timings_l5);
      throw;
    }
  } catch (const std::exception &error) {
    std::cerr << "APUg2 uint16 GEMM host failure: " << error.what() << '\n';
    return 1;
  }
}
