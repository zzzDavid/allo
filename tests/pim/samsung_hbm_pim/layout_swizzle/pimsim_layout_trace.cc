// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Standalone Samsung HBM-PIM layout-trace timing probe.
//
// The input is intentionally smaller than a DRAMSim trace: every non-comment
// line is
//
//     round bank row col
//
// where bank is the global Samsung bank number [0, 16) and col is a burst
// column [0, 32).  A tuple is replicated over the complete 64-channel fabric.
// Tuples in one round are queued together; a barrier is then attached to the
// last request on every active channel.  This mirrors the full-fabric issue
// and barrier behavior of PIMKernel::addTransactionAll/PIMKernel::addBarrier,
// while leaving the vendor PIMSimulator source tree untouched.

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Burst.h"
#include "MultiChannelMemorySystem.h"
#include "tests/KernelAddrGen.h"

using DRAMSim::BurstType;
using DRAMSim::MultiChannelMemorySystem;

namespace {

constexpr unsigned kSamsungChannels = 64;
constexpr unsigned kSamsungBanks = 16;
constexpr unsigned kSamsungRows = 16384;
constexpr unsigned kSamsungBurstColumns = 32;

struct TraceRequest {
  uint64_t round;
  unsigned bank;
  unsigned row;
  unsigned col;
};

struct Options {
  std::string trace_path;
  std::string pimsim_root;
  std::string output_dir = "/tmp";
  bool single_channel = false;
};

std::string join_path(const std::string &lhs, const std::string &rhs) {
  if (lhs.empty() || lhs == ".")
    return lhs.empty() ? rhs : lhs + "/" + rhs;
  if (lhs.back() == '/')
    return lhs + rhs;
  return lhs + "/" + rhs;
}

uint64_t parse_uint(const std::string &token, const std::string &field,
                    size_t line_number) {
  if (token.empty() || token[0] == '-') {
    throw std::runtime_error("line " + std::to_string(line_number) +
                             ": invalid " + field + " '" + token + "'");
  }
  const bool is_hex = token.size() > 2 && token[0] == '0' &&
                      (token[1] == 'x' || token[1] == 'X');
  size_t consumed = 0;
  uint64_t value = 0;
  try {
    value = std::stoull(token, &consumed, is_hex ? 16 : 10);
  } catch (const std::exception &) {
    throw std::runtime_error("line " + std::to_string(line_number) +
                             ": invalid " + field + " '" + token + "'");
  }
  if (consumed != token.size()) {
    throw std::runtime_error("line " + std::to_string(line_number) +
                             ": invalid " + field + " '" + token + "'");
  }
  return value;
}

std::vector<TraceRequest> read_trace(const std::string &path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("cannot open trace: " + path);

  std::vector<TraceRequest> requests;
  std::string line;
  size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    const size_t comment = line.find('#');
    if (comment != std::string::npos)
      line.erase(comment);

    std::istringstream fields(line);
    std::string round_token;
    std::string bank_token;
    std::string row_token;
    std::string col_token;
    std::string extra;
    if (!(fields >> round_token))
      continue;
    if (!(fields >> bank_token >> row_token >> col_token) ||
        (fields >> extra)) {
      throw std::runtime_error("line " + std::to_string(line_number) +
                               ": expected exactly 'round bank row col'");
    }

    const uint64_t round = parse_uint(round_token, "round", line_number);
    const uint64_t bank = parse_uint(bank_token, "bank", line_number);
    const uint64_t row = parse_uint(row_token, "row", line_number);
    const uint64_t col = parse_uint(col_token, "col", line_number);
    if (bank >= kSamsungBanks) {
      throw std::runtime_error("line " + std::to_string(line_number) +
                               ": bank must be in [0, 16)");
    }
    if (row >= kSamsungRows) {
      throw std::runtime_error("line " + std::to_string(line_number) +
                               ": row must be in [0, 16384)");
    }
    if (col >= kSamsungBurstColumns) {
      throw std::runtime_error("line " + std::to_string(line_number) +
                               ": col must be in [0, 32)");
    }
    if (!requests.empty() && round < requests.back().round) {
      throw std::runtime_error(
          "line " + std::to_string(line_number) +
          ": rounds must be grouped in nondecreasing order");
    }
    requests.push_back(TraceRequest{round, static_cast<unsigned>(bank),
                                    static_cast<unsigned>(row),
                                    static_cast<unsigned>(col)});
  }
  if (requests.empty())
    throw std::runtime_error("trace has no requests: " + path);
  return requests;
}

void usage(const char *argv0) {
  std::cerr
      << "usage: " << argv0
      << " [--pimsim-root DIR] [--output-dir DIR] [--single-channel] TRACE\n";
}

Options parse_options(int argc, char **argv) {
  Options options;
  const char *env_root = std::getenv("PIMSIMULATOR_ROOT");
  options.pimsim_root = env_root ? env_root : ".";

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--pimsim-root" || arg == "--output-dir") {
      if (++i >= argc)
        throw std::runtime_error(arg + " requires a value");
      if (arg == "--pimsim-root")
        options.pimsim_root = argv[i];
      else
        options.output_dir = argv[i];
    } else if (arg == "--single-channel") {
      options.single_channel = true;
    } else if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("unknown option: " + arg);
    } else if (!options.trace_path.empty()) {
      throw std::runtime_error("expected exactly one trace path");
    } else {
      options.trace_path = arg;
    }
  }
  if (options.trace_path.empty())
    throw std::runtime_error("missing trace path");
  return options;
}

void add_full_channel_barrier(
    const std::shared_ptr<MultiChannelMemorySystem> &memory,
    unsigned active_channels) {
  for (unsigned channel = 0; channel < active_channels; ++channel) {
    if (!memory->addBarrier(channel)) {
      throw std::runtime_error("failed to add round barrier on channel " +
                               std::to_string(channel));
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    const std::vector<TraceRequest> requests = read_trace(options.trace_path);

    const std::string device_ini =
        join_path(options.pimsim_root, "ini/HBM2_samsung_2M_16B_x64.ini");
    const std::string system_ini =
        join_path(options.pimsim_root, "system_hbm_64ch.ini");
    auto memory = std::make_shared<MultiChannelMemorySystem>(
        device_ini, system_ini, options.output_dir, "tenon_layout_trace",
        256 * 64 * 2);

    unsigned configured_channels = 0;
    unsigned configured_banks = 0;
    unsigned configured_rows = 0;
    unsigned configured_cols = 0;
    unsigned burst_length = 0;
    memory->getIniUint("NUM_CHANS", &configured_channels);
    memory->getIniUint("NUM_BANKS", &configured_banks);
    memory->getIniUint("NUM_ROWS", &configured_rows);
    memory->getIniUint("NUM_COLS", &configured_cols);
    memory->getIniUint("BL", &burst_length);
    if (configured_channels != kSamsungChannels ||
        configured_banks != kSamsungBanks || configured_rows != kSamsungRows ||
        burst_length == 0 ||
        configured_cols / burst_length != kSamsungBurstColumns) {
      throw std::runtime_error(
          "installed configuration is not Samsung 64-channel HBM-PIM");
    }

    // PIMAddrManager is the simulator's inverse for Scheme8.  Split the
    // trace's global bank into Samsung's bank-group and within-group bank.
    PIMAddrManager addresses(kSamsungChannels, 1);
    BurstType read_sink;
    const unsigned active_channels =
        options.single_channel ? 1 : kSamsungChannels;
    uint64_t physical_requests = 0;
    uint64_t round_count = 0;
    uint64_t current_round = requests.front().round;

    for (const TraceRequest &request : requests) {
      if (request.round != current_round) {
        add_full_channel_barrier(memory, active_channels);
        ++round_count;
        current_round = request.round;
      }
      const unsigned bank_group = request.bank / 4;
      const unsigned bank_in_group = request.bank % 4;
      for (unsigned channel = 0; channel < active_channels; ++channel) {
        const uint64_t address = addresses.addrGen(
            channel, 0, bank_group, bank_in_group, request.row, request.col);
        if (!memory->addTransaction(false, address, "TENON_LAYOUT_TRACE",
                                    &read_sink)) {
          throw std::runtime_error("simulator rejected a trace request");
        }
        ++physical_requests;
      }
    }
    add_full_channel_barrier(memory, active_channels);
    ++round_count;

    while (memory->hasPendingTransactions() != 0)
      memory->update();

    // This is the probe's sole machine-readable result line.  Other vendor
    // diagnostics, if enabled in a local configuration, lack this prefix.
    std::cout << "PIM_LAYOUT_CYCLES cycles=" << memory->currentClockCycle
              << " rounds=" << round_count
              << " logical_requests=" << requests.size()
              << " physical_requests=" << physical_requests
              << " channels=" << active_channels << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "pimsim_layout_trace: " << error.what() << std::endl;
    usage(argv[0]);
    return 2;
  }
}
