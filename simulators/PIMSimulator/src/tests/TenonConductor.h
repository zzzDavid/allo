// Tenon plan-driven conductor for PIMSimulator.
//
// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The compiler emits a per-pass CRF program and a plan (operand placement and
// pass structure derived from the selected placement). The conductor uploads
// that CRF verbatim, walks it with an exact model of PIMRank::doPIM, and
// issues the column commands each instruction execution consumes at the
// addresses the plan assigns. It never regenerates microcode (PIMCmdGen).
// Contract: experiments/allo/docs/release/design/samsung-execution.md.

#ifndef TENON_CONDUCTOR_H
#define TENON_CONDUCTOR_H

#include <cstdint>
#include <string>
#include <vector>

#include "PIMCmd.h"

// One instruction execution observed by every PIM block of a channel: the
// CRF word `pc` executed for `triggers` consecutive column commands (8 for an
// auto-mode instruction or FILL, loopCounter+1 for NOP, 1 otherwise).
struct TenonCrfStep
{
    int pc;
    int triggers;
};

// Statically executes `crf` from PC 0 the way PIMRank::doPIM consumes column
// commands (auto repeat, NOP repeat, inline JUMP with a single jump register,
// EXIT look-ahead) and returns the instruction executions until EXIT.
// Throws when the program counter leaves the program or more than
// `max_triggers` column commands would be consumed.
std::vector<TenonCrfStep> tenonWalkCrf(const std::vector<DRAMSim::PIMCmd>& crf,
                                       long max_triggers);

// Physical geometry a plan was derived for; checked against the kernel.
struct TenonGeometry
{
    int channels = 0;
    int pim_blocks = 0;
    int grf_a = 0;
    int grf_b = 0;
    int lanes = 0;
    int crf_words = 0;
};

// REDUCE (contraction) plan.  Slot s = ((j * channels + c) * pim_blocks + b)
// * rows_per_block + g holds one output row of pass j on block b of channel c
// in GRF_B entry g.  The weight file rows are already in slot order.
struct TenonReducePlan
{
    TenonGeometry geometry;
    int out_tiles = 0;             // J: passes over the slot fabric per column
    int k_tiles = 0;               // T: reduction tiles of grf_a * lanes elements
    std::vector<int> tile_fiber;   // bank fiber (0 even, 1 odd) holding tile t
    int rows_per_block = 0;        // G: GRF_B entries per block per pass
    bool vector_from_bank = false; // CRF fills GRF_A from the even bank (else host)
    int columns = 1;               // N: reduction columns (GEMM N, batch B)
    bool weight_resident = true;   // preload the matrix once for all columns
    // Row pairing of the MAC groups of one tile.  A tile's G*E columns span
    // two physical rows (cols_per_row columns each): groups 0..G/2-1 read
    // the first row, G/2..G-1 the second, and the GRF_B entry of a group is
    // selected by the row's parity and the column (PIMRank::getGrfIdxHigh).
    // issue_order 0 (tile_major): groups in order, one barrier each.
    // issue_order 1 (row_pair): group g and group g+G/2 of the same tile
    // are issued back to back under one barrier, so the tile's two rows
    // meet inside one barrier group (the same GRF_A vector slice, distinct
    // GRF_B entries).
    int issue_order = 0;
    // half_fiber[h]: the bank fiber holding half h (physical row h) of every
    // tile whose base fiber is the even fiber, the plan's rendering of the
    // selected F2 half -> bank map.  {0, 0} keeps both rows in the tile's
    // fiber (the vendor layout); {0, 1} stages the second row in the odd
    // fiber at the same column position (the swizzled layout).  split_rows
    // is derived: half_fiber[0] != half_fiber[1].
    int half_fiber[2] = {0, 0};
    int split_rows = 0;
};

// ELTWISE plan.  Every operand is laid out flat (preloadNoReplacement order)
// starting at its base row; one tile = channels * banks * grf entries bursts.
struct TenonEltwisePlan
{
    TenonGeometry geometry;
    int tiles = 0;                 // elementwise tiles
    int inputs = 0;                // number of input operands
    std::vector<int> input_rows;   // base DRAM row of each input operand
    // Operand stored with even/odd banks exchanged: lets a program read the
    // even-half partner of an operand through the odd-bank load (and back).
    std::vector<bool> input_swap_parity;
    int output_row = 0;            // base DRAM row of the output
};

struct TenonAudit
{
    long mac_groups[2] = {0, 0};   // auto MAC executions per bank fiber
    long arith_groups[2] = {0, 0}; // auto ADD/MUL/MAD executions per bank fiber
    long fills = 0;                // FILL executions
    long drains = 0;               // NOP-drain executions
    long host_vector_writes = 0;   // GRF_A host broadcast bursts per channel
    long column_commands = 0;      // PIM-triggering column commands per channel
    long passes = 0;               // HAB_PIM sessions
    long matrix_preloads = 0;      // full matrix stagings
    std::uint64_t crf_uploads = 0; // programCrf calls
};

std::string tenonReadPlanText(const std::string& path);
TenonReducePlan tenonParseReducePlan(const std::string& text);
TenonEltwisePlan tenonParseEltwisePlan(const std::string& text);

#endif
