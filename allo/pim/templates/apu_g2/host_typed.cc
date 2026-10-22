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
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "apu_g2_params.h"
#include "apu_g2_typed_bits.h"

#ifndef DEVICE_SIDE_LIB_LOCATION
#error DEVICE_SIDE_LIB_LOCATION must name the APUg2 task .update.bin
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint32_t kGroups = 16;
constexpr uint32_t kVectors = 4;
constexpr uint32_t kElementsPerGroup = 4096;
constexpr uint64_t kPackElements =
    uint64_t{kGroups} * kVectors * kElementsPerGroup;
constexpr uint32_t kMaximumValueBits = 24;
constexpr uint32_t kSlotAlignmentRows = 64;
constexpr uint32_t kSlotLiveRows = kVectors * kMaximumValueBits;
constexpr uint32_t kSlotPitchRows =
    ((kSlotLiveRows + kSlotAlignmentRows - 1) / kSlotAlignmentRows) *
    kSlotAlignmentRows;
constexpr uint32_t kLhsRow = 64;
// One four-vector bit-sliced pack occupies stride * kVectors rows.  Every
// dynamic-width value gets an aligned slot sized from the maximum legal type.
constexpr uint32_t kRhsRow = kLhsRow + kSlotPitchRows;
constexpr uint32_t kOutRow = kRhsRow + kSlotPitchRows;
constexpr uint32_t kScratchRow = kOutRow + kSlotPitchRows;
static_assert(kSlotLiveRows == 96);
static_assert(kSlotPitchRows == 128);
static_assert(kLhsRow + kSlotLiveRows <= kRhsRow);
static_assert(kRhsRow + kSlotLiveRows <= kOutRow);
static_assert(kOutRow + kSlotLiveRows <= kScratchRow);
constexpr uint32_t kTempRow = 2800;
constexpr uint32_t kIndexRow = 2928;
constexpr uint32_t kTempRows = 128;
constexpr uint32_t kIndexRows = 16;
static_assert(kScratchRow + kSlotLiveRows <= kTempRow);
static_assert(kTempRow + kTempRows <= kIndexRow);
static_assert(kIndexRow + kIndexRows <= 3072);

double to_us(Clock::duration value) {
  return std::chrono::duration<double, std::micro>(value).count();
}

uint32_t parse_u32(const char *text, const char *name) {
  const uint64_t value = std::stoull(text);
  if (value > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument(std::string(name) + " must fit uint32");
  return static_cast<uint32_t>(value);
}

template <typename T>
std::vector<int64_t> read_native_pack(const std::string &path) {
  static_assert(std::is_integral_v<T>);
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    throw std::runtime_error("cannot open input " + path);
  const auto bytes = file.tellg();
  const auto expected = static_cast<std::streamoff>(kPackElements * sizeof(T));
  if (bytes != expected)
    throw std::runtime_error("wrong byte count for " + path + ": expected " +
                             std::to_string(expected) + ", got " +
                             std::to_string(bytes));
  file.seekg(0);
  std::vector<T> native(kPackElements);
  if (!file.read(reinterpret_cast<char *>(native.data()), bytes))
    throw std::runtime_error("cannot read input " + path);
  std::vector<int64_t> values(kPackElements);
  for (uint64_t i = 0; i < kPackElements; ++i)
    values[i] = static_cast<int64_t>(native[i]);
  return values;
}

std::vector<int64_t> read_pack(const std::string &path, uint32_t bits,
                               bool is_signed) {
  if (bits <= 8)
    return is_signed ? read_native_pack<int8_t>(path)
                     : read_native_pack<uint8_t>(path);
  if (bits <= 16)
    return is_signed ? read_native_pack<int16_t>(path)
                     : read_native_pack<uint16_t>(path);
  return is_signed ? read_native_pack<int32_t>(path)
                   : read_native_pack<uint32_t>(path);
}

void fill_pack(g2_data::VectorPackRef &ref,
               const std::vector<int64_t> &values) {
  uint64_t offset = 0;
  for (uint32_t vector = 0; vector < kVectors; ++vector)
    for (uint32_t group = 0; group < kGroups; ++group)
      for (uint32_t element = 0; element < kElementsPerGroup; ++element)
        ref.ivecs[group][vector][element] = values[offset++];
}

void clear_pack(g2_data::VectorPackRef &ref) {
  for (uint32_t group = 0; group < kGroups; ++group)
    for (uint32_t vector = 0; vector < kVectors; ++vector)
      for (uint32_t element = 0; element < kElementsPerGroup; ++element)
        ref.ivecs[group][vector][element] = 0;
}

template <typename T>
void write_native_pack(const std::string &path,
                       const g2_data::VectorPackRef &ref, uint32_t bits,
                       bool is_signed) {
  static_assert(std::is_integral_v<T>);
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file)
    throw std::runtime_error("cannot open output " + path);
  for (uint32_t vector = 0; vector < kVectors; ++vector)
    for (uint32_t group = 0; group < kGroups; ++group)
      for (uint32_t element = 0; element < kElementsPerGroup; ++element) {
        const int64_t canonical = apu_g2_canonical_scalar(
            ref.ivecs[group][vector][element], bits, is_signed);
        const T value = static_cast<T>(canonical);
        file.write(reinterpret_cast<const char *>(&value), sizeof(value));
      }
  if (!file)
    throw std::runtime_error("cannot write output " + path);
}

void write_pack(const std::string &path, const g2_data::VectorPackRef &ref,
                uint32_t bits, bool is_signed) {
  if (bits <= 8) {
    if (is_signed)
      write_native_pack<int8_t>(path, ref, bits, is_signed);
    else
      write_native_pack<uint8_t>(path, ref, bits, is_signed);
  } else if (bits <= 16) {
    if (is_signed)
      write_native_pack<int16_t>(path, ref, bits, is_signed);
    else
      write_native_pack<uint16_t>(path, ref, bits, is_signed);
  } else if (is_signed) {
    write_native_pack<int32_t>(path, ref, bits, is_signed);
  } else {
    write_native_pack<uint32_t>(path, ref, bits, is_signed);
  }
}

gsi::G2_TYPES value_type(bool is_signed) {
  return is_signed ? gsi::G2_TYPES::INT : gsi::G2_TYPES::UINT;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 13) {
    std::cerr << "usage: " << argv[0]
              << " OP LHS.bin RHS.bin|- OUT.bin LHS_BITS LHS_SIGNED"
                 " RHS_BITS RHS_SIGNED OUT_BITS OUT_SIGNED LOG_REDUCTION"
                 " REPETITIONS\n";
    return 2;
  }

  try {
    const uint32_t operation = parse_u32(argv[1], "operation");
    const uint32_t lhs_bits = parse_u32(argv[5], "lhs bits");
    const bool lhs_signed = parse_u32(argv[6], "lhs signed") != 0;
    const uint32_t rhs_bits = parse_u32(argv[7], "rhs bits");
    const bool rhs_signed = parse_u32(argv[8], "rhs signed") != 0;
    const uint32_t out_bits = parse_u32(argv[9], "out bits");
    const bool out_signed = parse_u32(argv[10], "out signed") != 0;
    const uint32_t log_reduction = parse_u32(argv[11], "log reduction");
    const uint32_t repetitions = parse_u32(argv[12], "repetitions");
    if (repetitions == 0)
      throw std::invalid_argument("repetitions must be nonzero");
    if (lhs_bits == 0 || lhs_bits > 24 || out_bits == 0 || out_bits > 24)
      throw std::invalid_argument("lhs/out bits must be in [1, 24]");
    const bool unary = operation == APUG2_TYPED_BLOCK_SUM;
    if (!unary && (rhs_bits == 0 || rhs_bits > 24))
      throw std::invalid_argument("rhs bits must be in [1, 24]");

    const auto all_start = Clock::now();
    const auto lhs_values = read_pack(argv[2], lhs_bits, lhs_signed);
    std::vector<int64_t> rhs_values;
    if (!unary)
      rhs_values = read_pack(argv[3], rhs_bits, rhs_signed);

    using namespace gtml_transport;
    Device device(DEVICE_SIDE_LIB_LOCATION, DeviceConfig{0, 0});
    device.init_gtml({kTempRow, 128, kIndexRow, 16});

    const gsi::VectorPackDescriptor lhs_desc(kVectors, lhs_bits,
                                             value_type(lhs_signed), lhs_bits);
    const uint32_t physical_rhs_bits = unary ? 1 : rhs_bits;
    const gsi::VectorPackDescriptor rhs_desc(
        kVectors, physical_rhs_bits, value_type(rhs_signed), physical_rhs_bits);
    const gsi::VectorPackDescriptor out_desc(kVectors, out_bits,
                                             value_type(out_signed), out_bits);
    const gsi::VectorPack lhs(lhs_desc, kLhsRow);
    const gsi::VectorPack rhs(rhs_desc, kRhsRow);
    const gsi::VectorPack out(out_desc, kOutRow);
    g2_data::VectorPackRef lhs_ref(lhs, kGroups);
    g2_data::VectorPackRef rhs_ref(rhs, kGroups);
    g2_data::VectorPackRef out_ref(out, kGroups);
    fill_pack(lhs_ref, lhs_values);
    if (!unary)
      fill_pack(rhs_ref, rhs_values);
    clear_pack(out_ref);

    g2_data::L1ContainerRef index(kIndexRow, 16, kGroups);
    index.fill_idx(16);
    copy_to_l1(device, index);

    const auto h2d_start = Clock::now();
    copy_to_l1(device, lhs_ref);
    if (!unary)
      copy_to_l1(device, rhs_ref);
    copy_to_l1(device, out_ref);
    const auto h2d_stop = Clock::now();

    ApuG2TypedTimings timings{};
    const uint64_t timings_l5 = device.alloc_l5(sizeof(timings));
    bool timings_l5_live = true;
    try {
      device.mem_to_l5(&timings, sizeof(timings), timings_l5);
      const ApuG2TypedParams params{operation,
                                    kLhsRow,
                                    kRhsRow,
                                    kOutRow,
                                    kScratchRow,
                                    lhs_bits,
                                    rhs_bits,
                                    out_bits,
                                    lhs_signed ? 1U : 0U,
                                    rhs_signed ? 1U : 0U,
                                    out_signed ? 1U : 0U,
                                    log_reduction,
                                    repetitions,
                                    timings_l5};

      const auto task_start = Clock::now();
      const int task_rc = device.run_task("tenon_apu_g2_typed", params);
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
      write_pack(argv[4], out_ref, out_bits, out_signed);
      const auto all_stop = Clock::now();

      std::cout << "device_pipeline_ticks=" << timings.pipeline_ticks << '\n'
                << "device_final_pipeline_ticks="
                << timings.final_pipeline_ticks << '\n';
      std::cout << std::fixed << std::setprecision(3)
                << "h2d_us=" << to_us(h2d_stop - h2d_start) << '\n'
                << "host_task_us=" << to_us(task_stop - task_start) << '\n'
                << "d2h_us=" << to_us(d2h_stop - d2h_start) << '\n'
                << "end_to_end_us=" << to_us(all_stop - all_start) << '\n';
      std::cout << "target=hardware\n"
                << "device_library=" << DEVICE_SIDE_LIB_LOCATION << '\n'
                << "task_status=" << task_rc << '\n'
                << "timed_scope=direct_vl64_pipeline\n"
                << "completion_barrier_included=1\n"
                << "independent_final_correctness_call=1\n"
                << "PASS physical_outputs=" << kPackElements << '\n'
                << "typed_operation=" << operation << '\n'
                << "lhs_bits=" << lhs_bits << '\n'
                << "rhs_bits=" << rhs_bits << '\n'
                << "out_bits=" << out_bits << '\n'
                << "log_reduction=" << log_reduction << '\n';
      return 0;
    } catch (...) {
      if (timings_l5_live)
        device.free_l5(timings_l5);
      throw;
    }
  } catch (const std::exception &error) {
    std::cerr << "APUg2 typed host failure: " << error.what() << '\n';
    return 1;
  }
}
