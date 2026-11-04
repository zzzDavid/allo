// Tenon plan-driven conductor: implementation (included by tests/PIMKernel.cpp).
//
// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// See tests/TenonConductor.h for the contract.  All arithmetic happens in the
// unmodified PIMRank/PIMBlock datapath; this file only uploads the supplied
// CRF, places operands, and issues the column commands the CRF consumes.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "tests/TenonConductor.h"

namespace
{
[[noreturn]] void tenonFail(const std::string& message)
{
    throw std::runtime_error(message);
}

// Optional normalized-address trace of every transaction the Tenon conductor
// issues, for the release evidence (TENON_CONDUCTOR_ADDRESS_TRACE=<path>):
// one line per DRAM transaction of channel 0 ("<tag> ch bank row col"
// after PIMAddrManager::addrGenSafe's column-to-row carry; "W" prefix for
// writes), "BARRIER" when a barrier is attached, "PASS n j" at each pass.
struct TenonAddressTrace
{
    std::FILE* file = nullptr;
    TenonAddressTrace()
    {
        const char* path = std::getenv("TENON_CONDUCTOR_ADDRESS_TRACE");
        if (path && *path) file = std::fopen(path, "w");
    }
    ~TenonAddressTrace()
    {
        if (file) std::fclose(file);
    }
    void line(const std::string& text)
    {
        if (file) std::fprintf(file, "%s\n", text.c_str());
    }
};
TenonAddressTrace& tenonTrace()
{
    static TenonAddressTrace trace;
    return trace;
}

const char* tenonOpdName(PIMOpdType opd)
{
    switch (opd)
    {
        case PIMOpdType::A_OUT: return "A_OUT";
        case PIMOpdType::M_OUT: return "M_OUT";
        case PIMOpdType::EVEN_BANK: return "EVEN_BANK";
        case PIMOpdType::ODD_BANK: return "ODD_BANK";
        case PIMOpdType::GRF_A: return "GRF_A";
        case PIMOpdType::GRF_B: return "GRF_B";
        case PIMOpdType::SRF_M: return "SRF_M";
        case PIMOpdType::SRF_A: return "SRF_A";
    }
    return "?";
}

std::string tenonWord(const std::vector<PIMCmd>& crf, int pc)
{
    std::ostringstream text;
    text << "CRF word " << pc << " (" << crf[pc].toStr() << ")";
    return text.str();
}

PIMOpdType tenonBankOpd(int fiber)
{
    return fiber == 0 ? PIMOpdType::EVEN_BANK : PIMOpdType::ODD_BANK;
}

bool tenonReadsBank(const PIMCmd& cmd, int fiber)
{
    PIMOpdType bank = tenonBankOpd(fiber);
    if (cmd.type_ == PIMCmdType::FILL || cmd.type_ == PIMCmdType::MOV)
        return cmd.src0_ == bank;
    if (cmd.type_ == PIMCmdType::MAD && cmd.src2_ == bank)
        return true;
    return cmd.src0_ == bank || cmd.src1_ == bank;
}

// Per-word data binding parsed from the plan: which operand stream the
// column commands of an instruction execution address.
struct TenonBind
{
    std::string kind;  // matrix | vector | input | drain
    int bank = -1;     // 0 even, 1 odd
    int operand = -1;  // input operand index (ELTWISE)
    int offset = 0;    // first column of the tile this word addresses
};

std::map<int, TenonBind> tenonParseBinds(const std::string& text)
{
    std::map<int, TenonBind> binds;
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line))
    {
        std::istringstream fields(line);
        std::string key;
        if (!(fields >> key) || key != "bind")
            continue;
        int pc = -1;
        TenonBind bind;
        if (!(fields >> pc >> bind.kind >> bind.bank))
            tenonFail("malformed plan line: " + line);
        if (fields >> bind.operand)
            fields >> bind.offset;
        if (binds.count(pc))
            tenonFail("plan binds CRF word " + std::to_string(pc) + " twice");
        binds[pc] = bind;
    }
    return binds;
}

std::map<std::string, std::vector<long>> tenonPlanFields(const std::string& text)
{
    std::map<std::string, std::vector<long>> fields;
    std::istringstream lines(text);
    std::string line;
    bool header = false;
    while (std::getline(lines, line))
    {
        auto hash = line.find('#');
        if (hash != std::string::npos)
            line.resize(hash);
        std::istringstream tokens(line);
        std::string key;
        if (!(tokens >> key))
            continue;
        if (key == "tenon_samsung_plan")
        {
            std::string version;
            tokens >> version;
            if (version != "v1")
                tenonFail("unsupported plan version: " + version);
            header = true;
            continue;
        }
        if (key == "bind" || key == "kind")
            continue;
        std::vector<long> values;
        long value;
        while (tokens >> value) values.push_back(value);
        if (values.empty())
            tenonFail("plan field without values: " + key);
        fields[key] = values;
    }
    if (!header)
        tenonFail("plan file lacks the 'tenon_samsung_plan v1' header");
    return fields;
}

long tenonScalar(const std::map<std::string, std::vector<long>>& fields, const std::string& key)
{
    auto it = fields.find(key);
    if (it == fields.end() || it->second.size() != 1)
        tenonFail("plan requires exactly one value for '" + key + "'");
    return it->second[0];
}

TenonGeometry tenonParseGeometry(const std::map<std::string, std::vector<long>>& fields)
{
    TenonGeometry geometry;
    geometry.channels = (int)tenonScalar(fields, "channels");
    geometry.pim_blocks = (int)tenonScalar(fields, "pim_blocks");
    geometry.grf_a = (int)tenonScalar(fields, "grf_a");
    geometry.grf_b = (int)tenonScalar(fields, "grf_b");
    geometry.lanes = (int)tenonScalar(fields, "lanes");
    geometry.crf_words = (int)tenonScalar(fields, "crf_words");
    return geometry;
}
}  // namespace

std::vector<TenonCrfStep> tenonWalkCrf(const std::vector<PIMCmd>& crf, long max_triggers)
{
    // Mirrors PIMRank::doPIM: per column command, resolve JUMPs inline,
    // execute one instruction (auto/FILL instructions repeat 8 times, NOP
    // repeats loopCounter+1 times), advance, and stop issuing once the next
    // word is EXIT (crfExit_).
    std::vector<TenonCrfStep> steps;
    const int n = (int)crf.size();
    if (n == 0)
        tenonFail("empty CRF program");
    int pc = 0, last_jump = -1, num_jump = -1, last_repeat = -1, num_repeat = -1;
    bool exited = crf[0].type_ == PIMCmdType::EXIT;
    long triggers = 0;
    while (!exited)
    {
        int executed = -1;
        bool fresh = false;
        PIMCmd cmd;
        do
        {
            if (pc < 0 || pc >= n)
                tenonFail("CRF program counter reached word " + std::to_string(pc) +
                          " of a " + std::to_string(n) + "-word program without EXIT");
            cmd = crf[pc];
            if (cmd.type_ == PIMCmdType::EXIT)
            {
                exited = true;
                break;
            }
            if (cmd.type_ == PIMCmdType::JUMP)
            {
                if (last_jump != pc && cmd.loopCounter_ > 0)
                {
                    last_jump = pc;
                    num_jump = cmd.loopCounter_;
                }
                if (num_jump > 0)
                {
                    pc -= cmd.loopOffset_;
                    num_jump--;
                }
            }
            else
            {
                executed = pc;
                bool repeats = cmd.type_ == PIMCmdType::FILL || cmd.isAuto_ ||
                               cmd.type_ == PIMCmdType::NOP;
                if (repeats)
                {
                    if (last_repeat != pc)
                    {
                        last_repeat = pc;
                        num_repeat = (cmd.type_ == PIMCmdType::NOP && !cmd.isAuto_)
                                         ? cmd.loopCounter_
                                         : 8 - 1;
                        fresh = true;
                    }
                    if (num_repeat > 0)
                    {
                        pc -= 1;
                        num_repeat--;
                    }
                    else
                        last_repeat = -1;
                }
                else
                    fresh = true;
            }
            pc++;
            if (pc >= 0 && pc < n && crf[pc].type_ == PIMCmdType::EXIT)
                exited = true;
        } while (cmd.type_ == PIMCmdType::JUMP);
        if (executed < 0)
            break;
        if (++triggers > max_triggers)
            tenonFail("CRF program consumes more than " + std::to_string(max_triggers) +
                      " column commands without reaching EXIT");
        if (fresh || steps.empty() || steps.back().pc != executed)
            steps.push_back(TenonCrfStep{executed, 1});
        else
            steps.back().triggers++;
    }
    return steps;
}

std::string tenonReadPlanText(const std::string& path)
{
    std::ifstream file(path);
    if (!file)
        tenonFail("cannot open plan file: " + path);
    std::ostringstream text;
    text << file.rdbuf();
    return text.str();
}

TenonReducePlan tenonParseReducePlan(const std::string& text)
{
    auto fields = tenonPlanFields(text);
    TenonReducePlan plan;
    plan.geometry = tenonParseGeometry(fields);
    plan.out_tiles = (int)tenonScalar(fields, "out_tiles");
    plan.k_tiles = (int)tenonScalar(fields, "k_tiles");
    plan.rows_per_block = (int)tenonScalar(fields, "rows_per_block");
    plan.vector_from_bank = tenonScalar(fields, "vector_from_bank") != 0;
    plan.columns = (int)tenonScalar(fields, "columns");
    plan.weight_resident = tenonScalar(fields, "weight_resident") != 0;
    auto it = fields.find("tile_fiber");
    if (it == fields.end())
        tenonFail("plan requires 'tile_fiber'");
    for (long fiber : it->second) plan.tile_fiber.push_back((int)fiber);
    auto order = fields.find("issue_order");
    if (order != fields.end())
    {
        if (order->second.size() != 1 || (order->second[0] != 0 && order->second[0] != 1))
            tenonFail("issue_order must be 0 or 1");
        plan.issue_order = (int)order->second[0];
    }
    auto halves = fields.find("half_fiber");
    if (halves != fields.end())
    {
        if (halves->second.size() != 2)
            tenonFail("half_fiber names the fiber of each of a tile's two rows");
        for (int h = 0; h < 2; ++h)
        {
            if (halves->second[h] != 0 && halves->second[h] != 1)
                tenonFail("half_fiber values must be 0 (even) or 1 (odd)");
            plan.half_fiber[h] = (int)halves->second[h];
        }
        if (plan.half_fiber[0] != 0)
            tenonFail("half_fiber[0] must be the even fiber (the tile's base fiber)");
    }
    plan.split_rows = plan.half_fiber[0] != plan.half_fiber[1];
    if (plan.split_rows)
    {
        for (int f : plan.tile_fiber)
            if (f != 0)
                tenonFail("a split half -> bank map stages every tile's first row in the even fiber");
    }
    return plan;
}

TenonEltwisePlan tenonParseEltwisePlan(const std::string& text)
{
    auto fields = tenonPlanFields(text);
    TenonEltwisePlan plan;
    plan.geometry = tenonParseGeometry(fields);
    plan.tiles = (int)tenonScalar(fields, "tiles");
    plan.inputs = (int)tenonScalar(fields, "inputs");
    plan.output_row = (int)tenonScalar(fields, "output_row");
    auto it = fields.find("input_rows");
    if (it == fields.end())
        tenonFail("plan requires 'input_rows'");
    for (long row : it->second) plan.input_rows.push_back((int)row);
    auto swap = fields.find("input_swap_parity");
    if (swap != fields.end())
        for (long flag : swap->second) plan.input_swap_parity.push_back(flag != 0);
    else
        plan.input_swap_parity.assign(plan.input_rows.size(), false);
    return plan;
}

void PIMKernel::tenonDescribe(std::ostream& out)
{
    out << "PIM_GEOMETRY channels=" << num_pim_chans_ << " ranks=" << num_pim_ranks_
        << " pim_blocks=" << num_pim_blocks_ << " banks=" << num_banks_
        << " bank_groups=" << num_bank_groups_ << " grf_a=" << num_grfA_
        << " grf_b=" << num_grfB_ << " lanes=" << (int)(sizeof(null_bst_.u16Data_) / 2)
        << " crf_words=" << (int)(sizeof(crf_bst_) / sizeof(crf_bst_[0])) * 8
        << " cols_per_row=" << pim_addr_mgr_->num_cols_per_bl_ << " rows=" << pim_addr_mgr_->num_rows_
        << std::endl;
}

void PIMKernel::tenonCheckGeometry(const TenonGeometry& geometry, size_t crf_words)
{
    const int lanes = (int)(sizeof(null_bst_.u16Data_) / 2);
    const int capacity = (int)(sizeof(crf_bst_) / sizeof(crf_bst_[0])) * 8;
    if (geometry.channels != num_pim_chans_ || geometry.pim_blocks != num_pim_blocks_ ||
        geometry.grf_a != num_grfA_ || geometry.grf_b != num_grfB_ || geometry.lanes != lanes ||
        geometry.crf_words != capacity)
        tenonFail("plan geometry disagrees with the simulator (run pim_driver --describe)");
    if (num_pim_ranks_ != 1)
        tenonFail("the Tenon conductor drives one rank per channel");
    if ((int)crf_words > capacity)
        tenonFail("CRF program has " + std::to_string(crf_words) + " words; the CRF holds " +
                  std::to_string(capacity));
}

void PIMKernel::tenonRunReduce(const TenonReducePlan& plan, const std::string& plan_text,
                               vector<PIMCmd>& crf, NumpyBurstType& weight,
                               NumpyBurstType& vectors, vector<BurstType>& result,
                               TenonAudit& audit, uint64_t& preload, uint64_t& execution)
{
    tenonCheckGeometry(plan.geometry, crf.size());
    const int C = num_pim_chans_, P = num_pim_blocks_, G = plan.rows_per_block;
    const int E = num_grfA_, J = plan.out_tiles, T = plan.k_tiles, N = plan.columns;
    const int banks_per_group = num_banks_ / num_bank_groups_;
    // Auto-mode MAC addresses GRF_A by col&7 and GRF_B by (row&1)<<2|(col>>3)&3,
    // so one row group is exactly grf_a columns and a tile holds grf_b groups.
    if (G != num_grfB_ || E != 8 || num_grfB_ != 8)
        tenonFail("REDUCE requires rows_per_block == grf_b == grf_a == 8 (auto-mode indexing)");
    if (J <= 0 || T <= 0 || N <= 0 || (int)plan.tile_fiber.size() != T)
        tenonFail("REDUCE plan needs out_tiles, k_tiles, columns > 0 and one fiber per tile");
    std::vector<int> fiber_tiles[2];
    std::vector<int> local_index(T, 0);
    for (int t = 0; t < T; ++t)
    {
        int f = plan.tile_fiber[t];
        if (f != 0 && f != 1)
            tenonFail("tile_fiber values must be 0 (even bank) or 1 (odd bank)");
        local_index[t] = (int)fiber_tiles[f].size();
        fiber_tiles[f].push_back(t);
    }
    const long slots = (long)J * C * P * G;
    if (weight.bShape.size() != 2 || (long)weight.bShape[0] != slots ||
        (int)weight.bShape[1] != T * E)
        tenonFail("REDUCE matrix must be [out_tiles*channels*pim_blocks*rows_per_block, "
                  "k_tiles*grf_a*lanes] in slot order");
    if (vectors.bShape.size() != 2 || (int)vectors.bShape[0] != N ||
        (int)vectors.bShape[1] != T * E)
        tenonFail("REDUCE vectors must be [columns, k_tiles*grf_a*lanes]");

    // Row map: optional plan field 'row_slots' gives the slot of every matrix
    // row (a permutation of the slots).  Row r is staged at slot row_slots[r]
    // and the result drained from that slot is returned as row r.  Without
    // the field, row r occupies slot r.  Addresses and command order depend
    // only on slots, so the row map changes data placement, not the stream.
    std::vector<long> slot_row(slots);
    for (long s = 0; s < slots; ++s) slot_row[s] = s;
    long rows_moved = 0;
    uint64_t row_digest = 0xcbf29ce484222325ULL;
    bool row_permuted = false;
    {
        auto plan_fields = tenonPlanFields(plan_text);
        auto it = plan_fields.find("row_slots");
        if (it != plan_fields.end())
        {
            const std::vector<long>& row_slots = it->second;
            if ((long)row_slots.size() != slots)
                tenonFail("row_slots must give one slot per matrix row (" + std::to_string(slots) +
                          " rows), got " + std::to_string(row_slots.size()));
            std::vector<char> taken(slots, 0);
            for (long r = 0; r < slots; ++r)
            {
                long s = row_slots[r];
                if (s < 0 || s >= slots || taken[s])
                    tenonFail("row_slots is not a permutation of the slots (row " +
                              std::to_string(r) + " -> slot " + std::to_string(s) + ")");
                taken[s] = 1;
                slot_row[s] = r;
                rows_moved += (s != r);
                row_digest = (row_digest ^ (uint64_t)s) * 0x100000001b3ULL;
            }
            row_permuted = true;
        }
    }
    {
        std::ostringstream digest;
        digest << std::hex;
        digest.width(16);
        digest.fill('0');
        digest << row_digest;
        std::cout << "TENON_ROW_MAP map=" << (row_permuted ? "permuted" : "identity")
                  << " rows_moved=" << rows_moved
                  << " digest=" << (row_permuted ? digest.str() : std::string("none")) << std::endl;
    }

    // Bind and validate the program against the plan.
    auto binds = tenonParseBinds(plan_text);
    auto steps = tenonWalkCrf(crf, 64L * (long)(T + 2) * (G + 2) * 8);
    struct Issue
    {
        char kind;  // 'M' matrix group, 'F' vector fill, 'D' drain
        int fiber, tile, local, group;
    };
    std::vector<Issue> issues;
    long groups[2] = {0, 0};
    int filled_tile = -1, drains = 0;
    for (const auto& step : steps)
    {
        auto bound = binds.find(step.pc);
        if (bound == binds.end())
            tenonFail(tenonWord(crf, step.pc) + " executes but the plan binds no operand to it");
        const PIMCmd& cmd = crf[step.pc];
        const TenonBind& bind = bound->second;
        if (drains)
            tenonFail(tenonWord(crf, step.pc) + " executes after the GRF_B drain");
        if (bind.kind == "matrix")
        {
            int f = bind.bank;
            // split_rows: the odd-fiber MAC reads the second rows of the
            // even fiber's tiles, so it is accounted on fiber 0's tiles.
            const int tile_source = plan.split_rows ? 0 : f;
            bool operands = cmd.dst_ == PIMOpdType::GRF_B &&
                            ((cmd.src0_ == PIMOpdType::GRF_A && cmd.src1_ == tenonBankOpd(f)) ||
                             (cmd.src1_ == PIMOpdType::GRF_A && cmd.src0_ == tenonBankOpd(f)));
            if (cmd.type_ != PIMCmdType::MAC || !cmd.isAuto_ || !operands || step.triggers != E)
                tenonFail(tenonWord(crf, step.pc) + " is bound as the fiber-" + std::to_string(f) +
                          " matrix MAC but is not an auto MAC GRF_B <- GRF_A * " +
                          tenonOpdName(tenonBankOpd(f)));
            long e = groups[f]++;
            int local, group;
            if (plan.split_rows)
            {
                // Each fiber's MAC covers half of every tile's groups: the
                // even fiber the first row (groups 0..G/2-1), the odd fiber
                // the second row (groups G/2..G-1).
                local = (int)(e / (G / 2));
                group = (int)(e % (G / 2)) + (f ? G / 2 : 0);
            }
            else if (plan.issue_order)
            {
                // row_pair on one fiber: the CRF repeats one MAC G times per
                // tile; execution q of a tile serves group q/2 of the first
                // row when q is even and group q/2 + G/2 of the second row
                // when q is odd, so the pair (g, g+G/2) is adjacent.
                local = (int)(e / G);
                int q = (int)(e % G);
                group = (q % 2 == 0) ? q / 2 : q / 2 + G / 2;
            }
            else
            {
                local = (int)(e / G);
                group = (int)(e % G);
            }
            if (local >= (int)fiber_tiles[tile_source].size())
                tenonFail(tenonWord(crf, step.pc) + " issues more MAC groups on fiber " +
                          std::to_string(f) + " than the plan places tiles there (" +
                          std::to_string(fiber_tiles[tile_source].size() * G) + " groups)");
            int tile = fiber_tiles[tile_source][local];
            if (plan.vector_from_bank && group == 0 && filled_tile != tile)
                tenonFail(tenonWord(crf, step.pc) + " starts tile " + std::to_string(tile) +
                          " but the CRF did not fill GRF_A with that tile");
            issues.push_back(Issue{'M', f, tile, local, group});
        }
        else if (bind.kind == "vector")
        {
            if (!plan.vector_from_bank || cmd.type_ != PIMCmdType::FILL ||
                cmd.dst_ != PIMOpdType::GRF_A || cmd.src0_ != PIMOpdType::EVEN_BANK ||
                bind.bank != 0 || step.triggers != E)
                tenonFail(tenonWord(crf, step.pc) +
                          " is bound as the vector fill but the plan or encoding disagree "
                          "(FILL GRF_A <- EVEN_BANK with vector_from_bank=1 expected)");
            issues.push_back(Issue{'F', 0, -1, -1, -1});
            filled_tile = -2;  // resolved below against the next MAC group
        }
        else if (bind.kind == "drain")
        {
            if (cmd.type_ != PIMCmdType::NOP || bind.bank != 1 || step.triggers != G)
                tenonFail(tenonWord(crf, step.pc) + " is bound as the GRF_B drain but is not a NOP "
                          "consuming rows_per_block writes to the odd bank");
            issues.push_back(Issue{'D', 1, -1, -1, -1});
            drains++;
        }
        else
            tenonFail(tenonWord(crf, step.pc) + " has unsupported REDUCE binding " + bind.kind);
        // A fill provides the tile of the next MAC group; resolve it now so
        // the next group's check sees the filled tile.
        if (issues.back().kind == 'F')
        {
            int f_next = -1;
            for (size_t later = &step - &steps[0] + 1; later < steps.size(); ++later)
            {
                auto b = binds.find(steps[later].pc);
                if (b != binds.end() && b->second.kind == "matrix")
                {
                    f_next = b->second.bank;
                    break;
                }
            }
            if (f_next < 0)
                tenonFail(tenonWord(crf, step.pc) + " fills GRF_A but no MAC group follows");
            long e = groups[f_next];
            if (plan.split_rows)
                tenonFail("split_rows has no CRF-filled GRF_A variant");
            if (e % G != 0 || e / G >= (long)fiber_tiles[f_next].size())
                tenonFail(tenonWord(crf, step.pc) + " refills GRF_A inside a tile");
            filled_tile = fiber_tiles[f_next][e / G];
            issues.back().tile = filled_tile;
        }
    }
    for (int f = 0; f < 2; ++f)
    {
        long expected = plan.split_rows ? (long)fiber_tiles[0].size() * (G / 2)
                                        : (long)fiber_tiles[f].size() * G;
        if (groups[f] != expected)
            tenonFail("CRF performs " + std::to_string(groups[f]) + " MAC groups on fiber " +
                      std::to_string(f) + "; the plan expects " + std::to_string(expected));
    }
    if (drains != 1)
        tenonFail("CRF must drain GRF_B exactly once per pass, found " + std::to_string(drains));

    const int T0 = (int)fiber_tiles[0].size(), T1 = (int)fiber_tiles[1].size();
    const int tile_cols = G * E;                      // one tile per block per fiber
    const int cols_per_row = pim_addr_mgr_->num_cols_per_bl_;
    if ((plan.issue_order || plan.split_rows) && (G % 2 != 0 || tile_cols != 2 * cols_per_row))
        tenonFail("row_pair/split_rows need a tile of exactly two physical rows (G*E == 2*cols_per_row)");
    const long vector_base = (long)J * T0 * tile_cols;  // even bank, after the matrix
    // Odd bank: after the matrix tiles it holds (split_rows: the second row
    // of every even-fiber tile sits at the same column position).
    const long result_base = (long)J * (plan.split_rows ? T0 : T1) * tile_cols;
    auto fiber_tiles_count = [&](int f) { return f == 0 ? T0 : T1; };
    const char* trace_tag = "stage";
    auto address = [&](int ch, int pb, int parity, long col) {
        int global_bank = 2 * pb + parity;
        unsigned row = 0, column = (unsigned)col;
        uint64_t addr = pim_addr_mgr_->addrGenSafe(ch, 0, global_bank / banks_per_group,
                                                   global_bank % banks_per_group, row, column);
        if (ch == 0 && tenonTrace().file)
            tenonTrace().line(std::string(trace_tag) + " 0 " + std::to_string(global_bank) + " " +
                              std::to_string(row) + " " + std::to_string(column));
        return addr;
    };
    // Normalized addresses of an all-bank PIM column command sequence
    // (PIMKernel::addTransactionAll with bank group 0): channel 0 only, the
    // other channels receive the same addresses.
    auto trace_all = [&](const char* tag, int bank, long col, int count, bool barrier) {
        if (!tenonTrace().file) return;
        unsigned row = 0, column = (unsigned)col;
        for (int i = 0; i < count; ++i)
        {
            pim_addr_mgr_->addrGenSafe(0, 0, 0, bank, row, column);
            tenonTrace().line(std::string(tag) + " 0 " + std::to_string(bank) + " " +
                              std::to_string(row) + " " + std::to_string(column));
            column++;
        }
        if (barrier) tenonTrace().line("BARRIER");
    };

    auto stage_matrix = [&]() {
        trace_tag = "Wstage_matrix";
        // Same order as the vendor preloadGemv: pass, tile, block, row group, burst.
        for (int j = 0; j < J; ++j)
            for (int t = 0; t < T; ++t)
            {
                int f = plan.tile_fiber[t];
                long tile_col = ((long)j * fiber_tiles_count(f) + local_index[t]) * tile_cols;
                for (int block = 0; block < C * P; ++block)
                    for (int g = 0; g < G; ++g)
                    {
                        long slot = ((long)j * C * P + block) * G + g;
                        long row = slot_row[slot];
                        // The selected half -> bank map: half h (physical row
                        // h of the tile) lives in fiber plan.half_fiber[h] at
                        // the same column position.
                        int fiber = plan.split_rows ? plan.half_fiber[(g * E) / cols_per_row] : f;
                        for (int a = 0; a < E; ++a)
                            mem_->addTransaction(
                                true, address(block / P, block % P, fiber, tile_col + g * E + a),
                                &weight.bData[row * T * E + t * E + a]);
                    }
            }
        audit.matrix_preloads++;
    };
    auto stage_vector = [&](int n) {
        trace_tag = "Wstage_vector";
        for (int t = 0; t < T; ++t)
            for (int block = 0; block < C * P; ++block)
                for (int a = 0; a < E; ++a)
                    mem_->addTransaction(
                        true, address(block / P, block % P, 0, vector_base + ((long)n * T + t) * E + a),
                        &vectors.bData[((long)n * T + t) * E + a]);
    };
    auto run_pass = [&](int n, int j) {
        tenonTrace().line("PASS " + std::to_string(n) + " " + std::to_string(j));
        changePIMMode(dramMode::HAB, dramMode::HAB_PIM);  // PC reset, GRF_B zeroed
        audit.passes++;
        // row_pair: one barrier closes the pair (g, g + G/2) of a tile, so
        // its two rows meet inside one barrier group.
        std::vector<int> pair_issued((size_t)T * (G / 2 > 0 ? G / 2 : 1), 0);
        for (const auto& issue : issues)
        {
            if (issue.kind == 'M')
            {
                if (!plan.vector_from_bank && issue.group == 0)
                {
                    // Host vector fill: HAB broadcast of the tile into GRF_A.
                    for (int& ch : pim_chans_)
                    {
                        for (int& ra : pim_ranks_)
                            for (int a = 0; a < E; ++a)
                                mem_->addTransaction(
                                    true, pim_addr_mgr_->addrGen(ch, ra, 0, 1, pim_reg_ra_, 0x8 + a),
                                    "WRIO_TO_GRF_",
                                    &vectors.bData[((long)n * T + issue.tile) * E + a]);
                        mem_->addBarrier(ch);
                    }
                    tenonTrace().line("Whost_fill_grf_a x" + std::to_string(E));
                    tenonTrace().line("BARRIER");
                    audit.host_vector_writes += E;
                }
                const int source_fiber = plan.split_rows ? 0 : issue.fiber;
                long col = ((long)j * fiber_tiles_count(source_fiber) + issue.local) * tile_cols +
                           issue.group * E;
                bool barrier = true;
                if (plan.issue_order)
                {
                    size_t pair = (size_t)issue.local * (G / 2) + (issue.group % (G / 2));
                    barrier = (++pair_issued[pair] == 2);
                }
                trace_all("MAC", issue.fiber, col, E, barrier);
                addTransactionAll(false, 0, issue.fiber, 0, (int)col, "MAC_", &null_bst_, barrier, E);
                audit.mac_groups[issue.fiber]++;
                audit.column_commands += E;
            }
            else if (issue.kind == 'F')
            {
                long col = vector_base + ((long)n * T + issue.tile) * E;
                trace_all("FILL", 0, col, E, true);
                addTransactionAll(false, 0, 0, 0, (int)col, "FILL_GRF_A_", &null_bst_, true, E);
                audit.fills++;
                audit.column_commands += E;
            }
            else
            {
                long col = result_base + ((long)n * J + j) * G;
                trace_all("WDRAIN", 1, col, G, true);
                addTransactionAll(true, 0, 1, 0, (int)col, "GRFB_TO_BANK_", &null_bst_, true, G);
                audit.drains++;
                audit.column_commands += G;
            }
        }
        changePIMMode(dramMode::HAB_PIM, dramMode::HAB);
    };
    auto session = [&](int first, int last) {
        setControl(&bst_hab_pim_, true, getToggleCond(), false, true);
        parkIn();
        changePIMMode(dramMode::SB, dramMode::HAB);
        programCrf(crf);
        audit.crf_uploads++;
        for (int n = first; n < last; ++n)
            for (int j = 0; j < J; ++j) run_pass(n, j);
        changePIMMode(dramMode::HAB, dramMode::SB);
        parkOut();
    };

    uint64_t start = getCycle();
    if (plan.weight_resident)
    {
        stage_matrix();
        if (plan.vector_from_bank)
            for (int n = 0; n < N; ++n) stage_vector(n);
        runPIM();
        preload += getCycle() - start;
        uint64_t exec_start = getCycle();
        session(0, N);
        runPIM();
        execution += getCycle() - exec_start;
    }
    else
    {
        // Restage: every column re-stages the matrix and re-programs the CRF.
        for (int n = 0; n < N; ++n)
        {
            uint64_t stage_start = getCycle();
            stage_matrix();
            if (plan.vector_from_bank)
                stage_vector(n);
            runPIM();
            preload += getCycle() - stage_start;
            uint64_t exec_start = getCycle();
            session(n, n + 1);
            runPIM();
            execution += getCycle() - exec_start;
        }
    }

    // Readback in vendor readResult order: column, pass, block, row group;
    // each drained slot lands at its row's position (row order).
    trace_tag = "readback";
    result.assign((size_t)N * slots, BurstType());
    for (int n = 0; n < N; ++n)
        for (int j = 0; j < J; ++j)
            for (int block = 0; block < C * P; ++block)
                for (int g = 0; g < G; ++g)
                {
                    long slot = ((long)j * C * P + block) * G + g;
                    mem_->addTransaction(
                        false, address(block / P, block % P, 1, result_base + ((long)n * J + j) * G + g),
                        "output", &result[(size_t)n * slots + slot_row[slot]]);
                }
}

void PIMKernel::tenonRunEltwise(const TenonEltwisePlan& plan, const std::string& plan_text,
                                vector<PIMCmd>& crf, std::vector<NumpyBurstType>& inputs,
                                vector<BurstType>& result, TenonAudit& audit, uint64_t& preload,
                                uint64_t& execution)
{
    tenonCheckGeometry(plan.geometry, crf.size());
    const int C = num_pim_chans_, E = num_grf_;
    const int banks_per_group = num_banks_ / num_bank_groups_;
    const long tile_bursts = (long)C * num_banks_ * E;
    if (plan.tiles <= 0 || plan.inputs <= 0 || (int)inputs.size() != plan.inputs ||
        (int)plan.input_rows.size() != plan.inputs ||
        (int)plan.input_swap_parity.size() != plan.inputs)
        tenonFail("ELTWISE plan needs tiles > 0 and one base row per input operand");
    for (auto& operand : inputs)
        if ((long)operand.bData.size() != plan.tiles * tile_bursts)
            tenonFail("ELTWISE operands must hold tiles * channels * banks * grf bursts");

    // The program covers one tile per pass (the compiler does not know the
    // simulated channel count, hence not the tile count); every bound word
    // executes once per pass and the words of one operand cover the tile.
    auto binds = tenonParseBinds(plan_text);
    auto steps = tenonWalkCrf(crf, 64L * (long)crf.size() * E + 64);
    struct Issue
    {
        bool drain;
        int bank, operand, col, count;
    };
    std::vector<Issue> issues;
    std::map<int, int> executions;
    for (const auto& step : steps)
    {
        auto bound = binds.find(step.pc);
        if (bound == binds.end())
            tenonFail(tenonWord(crf, step.pc) + " executes but the plan binds no operand to it");
        const PIMCmd& cmd = crf[step.pc];
        const TenonBind& bind = bound->second;
        executions[step.pc]++;
        if (bind.bank != 0 && bind.bank != 1)
            tenonFail(tenonWord(crf, step.pc) + " is bound to an invalid bank");
        if (bind.offset < 0 || bind.offset + step.triggers > E)
            tenonFail(tenonWord(crf, step.pc) + " addresses columns outside its tile");
        Issue issue{false, bind.bank, -1, bind.offset, step.triggers};
        if (bind.kind == "input")
        {
            if (bind.operand < 0 || bind.operand >= plan.inputs || !tenonReadsBank(cmd, bind.bank))
                tenonFail(tenonWord(crf, step.pc) + " is bound to input " +
                          std::to_string(bind.operand) + " on bank " +
                          std::to_string(bind.bank) + " but does not read that bank");
            issue.operand = bind.operand;
        }
        else if (bind.kind == "drain")
        {
            if (cmd.type_ != PIMCmdType::NOP)
                tenonFail(tenonWord(crf, step.pc) + " is bound as a drain but is not a NOP");
            issue.drain = true;
        }
        else
            tenonFail(tenonWord(crf, step.pc) + " has unsupported ELTWISE binding " + bind.kind);
        // Consecutive single-column words of one operand form one column group.
        if (!issues.empty())
        {
            Issue& last = issues.back();
            if (last.drain == issue.drain && last.bank == issue.bank &&
                last.operand == issue.operand && last.col + last.count == issue.col &&
                last.col + last.count < E)
            {
                last.count += issue.count;
                continue;
            }
        }
        issues.push_back(issue);
    }
    for (const auto& entry : binds)
        if (executions[entry.first] != 1)
            tenonFail(tenonWord(crf, entry.first) + " executes " +
                      std::to_string(executions[entry.first]) + " times per pass; expected once");
    for (const auto& issue : issues)
        if (issue.col != 0 || issue.count != E)
            tenonFail("an ELTWISE column group does not cover a whole tile (" +
                      std::to_string(issue.count) + " of " + std::to_string(E) + " columns)");

    // Flat placement (preloadNoReplacement order): burst x of an operand goes to
    // channel x%C, bank (x/C)%banks_per_group, bank group, then column and row.
    auto flat_address = [&](int base_row, long x, bool swap_parity) {
        int ch = (int)(x % C);
        long rest = x / C;
        int bank = (int)(rest % banks_per_group);
        rest /= banks_per_group;
        int bg = (int)(rest % num_bank_groups_);
        rest /= num_bank_groups_;
        unsigned col = (unsigned)(rest % pim_addr_mgr_->num_cols_per_bl_);
        unsigned row = (unsigned)(base_row + rest / pim_addr_mgr_->num_cols_per_bl_);
        if (swap_parity)
            bank ^= 1;
        return pim_addr_mgr_->addrGen(ch, 0, bg, bank, row, col);
    };
    uint64_t start = getCycle();
    for (int k = 0; k < plan.inputs; ++k)
        for (long x = 0; x < plan.tiles * tile_bursts; ++x)
            mem_->addTransaction(true, flat_address(plan.input_rows[k], x, plan.input_swap_parity[k]),
                                 &inputs[k].bData[x]);
    runPIM();
    preload += getCycle() - start;

    uint64_t exec_start = getCycle();
    setControl(&bst_hab_pim_, true, getToggleCond(), false, false);
    setControl(&bst_hab_, false, getToggleCond(), false, false);
    parkIn();
    changePIMMode(dramMode::SB, dramMode::HAB);
    programCrf(crf);
    audit.crf_uploads++;
    for (int tile = 0; tile < plan.tiles; ++tile)
    {
        changePIMMode(dramMode::HAB, dramMode::HAB_PIM);  // PC reset for this tile
        audit.passes++;
        for (const auto& issue : issues)
        {
            if (issue.drain)
            {
                addTransactionAll(true, 0, issue.bank, plan.output_row, tile * E + issue.col,
                                  "GRF_TO_BANK", &null_bst_, true, issue.count);
                audit.drains++;
            }
            else
            {
                addTransactionAll(false, 0, issue.bank, plan.input_rows[issue.operand],
                                  tile * E + issue.col, "BANK_TO_GRF_", &null_bst_, true,
                                  issue.count);
                audit.arith_groups[issue.bank]++;
            }
            audit.column_commands += issue.count;
        }
        changePIMMode(dramMode::HAB_PIM, dramMode::HAB);
    }
    changePIMMode(dramMode::HAB, dramMode::SB);
    parkOut();
    runPIM();
    execution += getCycle() - exec_start;

    result.assign((size_t)(plan.tiles * tile_bursts), BurstType());
    for (long x = 0; x < plan.tiles * tile_bursts; ++x)
        mem_->addTransaction(false, flat_address(plan.output_row, x, false), &result[x]);
}
