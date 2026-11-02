// pim_driver: command-line conductor that runs one Tenon-compiled program on
// PIMSimulator and writes the drained result bursts to a binary file.
//
// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Routes:
//   pim_driver --describe
//       Print the simulated geometry (channels, PIM blocks, GRF entries,
//       lanes, CRF words) the compiler must plan against.
//   pim_driver --op GENERIC --op-kind REDUCE|ELTWISE --plan plan.txt
//              --crf crf.txt --inputs a.npy[,b.npy...] --out out.bin
//       Tenon conductor (tests/TenonConductor.inc).  Uploads crf.txt verbatim,
//       places operands as plan.txt says, and issues the column commands the
//       CRF consumes.  Never calls PIMCmdGen; the receipt line reports the
//       generator invocation count to prove it.
//   pim_driver --op GENERIC --op-kind TEMPORAL_REDUCE --cmds crf.txt
//              --inputs A.npy,B.npy,init.npy --out-rows M --reduce-k K
//              --out-cols N --temporal-audit prefix --out out.bin
//       Ordered FP16 contraction (tests/TemporalReduce.inc).
//   pim_driver --op GEMV --weight W.npy --in x.npy --output-dim M
//              --input-dim K --out out.bin
//       Vendor reference: the unmodified PIMSimulator preloadGemv/executeGemv/
//       readResult sequence with the vendor-generated CRF.  Used only as the
//       comparison baseline; Tenon programs never take this route.
//
// Inputs are float16 .npy files; outputs are raw 32-byte bursts (16 fp16).
// Every run prints "PIM_CYCLES total=... preload=... exec=... readback=...".
//
// CRF text format (one word per line, key=value fields; mirror of the Python
// PIMCmd dataclass in experiments/allo/allo/spmw_codegen.py):
//   MAC dst=GRF_B src0=GRF_A src1=EVEN_BANK is_auto=1
//   JUMP loop_counter=7 loop_offset=2
//   NOP loop_counter=7
//   EXIT

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "MultiChannelMemorySystem.h"
#include "PIMCmd.h"
#include "tests/KernelAddrGen.h"
#include "tests/PIMCmdGen.h"
#include "tests/PIMKernel.h"
#include "tests/TenonConductor.h"

using namespace DRAMSim;
using namespace std;

static void die(const std::string& msg)
{
    std::cerr << "pim_driver: " << msg << std::endl;
    std::exit(1);
}

static std::vector<std::string> split(const std::string& s, char delim)
{
    std::vector<std::string> out;
    std::string item;
    std::istringstream ss(s);
    while (std::getline(ss, item, delim))
        if (!item.empty())
            out.push_back(item);
    return out;
}

static PIMCmdType parseCmdType(const std::string& s)
{
    if (s == "NOP") return PIMCmdType::NOP;
    if (s == "ADD") return PIMCmdType::ADD;
    if (s == "MUL") return PIMCmdType::MUL;
    if (s == "MAC") return PIMCmdType::MAC;
    if (s == "MAD") return PIMCmdType::MAD;
    if (s == "MOV") return PIMCmdType::MOV;
    if (s == "FILL") return PIMCmdType::FILL;
    if (s == "JUMP") return PIMCmdType::JUMP;
    if (s == "EXIT") return PIMCmdType::EXIT;
    die("unknown CRF opcode: " + s);
    return PIMCmdType::NOP;
}

static PIMOpdType parseOpdType(const std::string& s)
{
    if (s == "A_OUT") return PIMOpdType::A_OUT;
    if (s == "M_OUT") return PIMOpdType::M_OUT;
    if (s == "EVEN_BANK") return PIMOpdType::EVEN_BANK;
    if (s == "ODD_BANK") return PIMOpdType::ODD_BANK;
    if (s == "GRF_A") return PIMOpdType::GRF_A;
    if (s == "GRF_B") return PIMOpdType::GRF_B;
    if (s == "SRF_M") return PIMOpdType::SRF_M;
    if (s == "SRF_A") return PIMOpdType::SRF_A;
    die("unknown CRF operand: " + s);
    return PIMOpdType::A_OUT;
}

// Parse the CRF text, then round-trip every word through the hardware
// encoding so the conductor reasons about exactly what the PIM blocks decode.
static std::vector<PIMCmd> load_crf(const std::string& path)
{
    std::ifstream f(path);
    if (!f)
        die("cannot open CRF file: " + path);
    std::vector<PIMCmd> out;
    std::string line;
    while (std::getline(f, line))
    {
        auto hash = line.find('#');
        if (hash != std::string::npos)
            line.resize(hash);
        std::istringstream ls(line);
        std::string head;
        if (!(ls >> head))
            continue;
        PIMCmdType ct = parseCmdType(head);
        PIMOpdType dst = PIMOpdType::A_OUT, src0 = PIMOpdType::A_OUT;
        PIMOpdType src1 = PIMOpdType::A_OUT, src2 = PIMOpdType::A_OUT;
        int loop_counter = 0, loop_offset = 0, is_auto = 0, dst_idx = 0;
        int src0_idx = 0, src1_idx = 0, is_relu = 0;
        std::string tok;
        while (ls >> tok)
        {
            auto eq = tok.find('=');
            if (eq == std::string::npos)
                die("malformed CRF token (need key=value): " + tok);
            std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
            if (k == "dst") dst = parseOpdType(v);
            else if (k == "src0") src0 = parseOpdType(v);
            else if (k == "src1") src1 = parseOpdType(v);
            else if (k == "src2") src2 = parseOpdType(v);
            else if (k == "loop_counter") loop_counter = std::stoi(v);
            else if (k == "loop_offset") loop_offset = std::stoi(v);
            else if (k == "is_auto") is_auto = std::stoi(v);
            else if (k == "dst_idx") dst_idx = std::stoi(v);
            else if (k == "src0_idx") src0_idx = std::stoi(v);
            else if (k == "src1_idx") src1_idx = std::stoi(v);
            else if (k == "is_relu") is_relu = std::stoi(v);
            else die("unknown CRF field: " + k);
        }
        PIMCmd cmd(ct, dst, src0, src1, src2, is_auto, dst_idx, src0_idx, src1_idx);
        cmd.loopCounter_ = loop_counter;
        cmd.loopOffset_ = loop_offset;
        cmd.isRelu_ = is_relu;
        PIMCmd decoded;
        decoded.fromInt(cmd.toInt());
        out.push_back(decoded);
    }
    if (out.empty())
        die("CRF file parsed to zero words: " + path);
    return out;
}

static std::shared_ptr<PIMKernel> make_kernel()
{
    // The simulated system: 64 pseudo-channels (system_hbm_64ch.ini), one
    // rank each.  --describe reports this geometry to the compiler.
    auto mem = std::make_shared<MultiChannelMemorySystem>(
        "ini/HBM2_samsung_2M_16B_x64.ini", "system_hbm_64ch.ini", ".", "pim_driver",
        256 * 64 * 2);
    return std::make_shared<PIMKernel>(mem, /*chans=*/64, /*ranks=*/1);
}

static void write_bursts(const std::string& path, const BurstType* bst, size_t nbursts)
{
    std::ofstream f(path, std::ios::binary);
    if (!f)
        die("cannot open output file: " + path);
    for (size_t i = 0; i < nbursts; ++i)
        f.write(reinterpret_cast<const char*>(bst[i].u16Data_), 32);
}

static void print_cycles(uint64_t total, uint64_t preload, uint64_t exec, uint64_t readback)
{
    std::cout << "PIM_CYCLES total=" << total << " preload=" << preload << " exec=" << exec
              << " readback=" << readback << std::endl;
}

int main(int argc, char** argv)
{
    std::string op, op_kind, out, weight, in, crf_path, cmds_path, plan_path, inputs_arg;
    std::string temporal_audit_prefix;
    int output_dim = 0, input_dim = 0, out_rows = 0, out_cols = 0, reduce_k = 0;
    bool describe = false;

    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto next = [&] {
            if (i + 1 >= argc)
                die("missing value for " + a);
            return std::string(argv[++i]);
        };
        if (a == "--describe") describe = true;
        else if (a == "--op") op = next();
        else if (a == "--op-kind") op_kind = next();
        else if (a == "--out") out = next();
        else if (a == "--weight") weight = next();
        else if (a == "--in") in = next();
        else if (a == "--output-dim") output_dim = std::stoi(next());
        else if (a == "--input-dim") input_dim = std::stoi(next());
        else if (a == "--crf") crf_path = next();
        else if (a == "--cmds") cmds_path = next();
        else if (a == "--plan") plan_path = next();
        else if (a == "--inputs") inputs_arg = next();
        else if (a == "--out-rows") out_rows = std::stoi(next());
        else if (a == "--out-cols") out_cols = std::stoi(next());
        else if (a == "--reduce-k") reduce_k = std::stoi(next());
        else if (a == "--temporal-audit") temporal_audit_prefix = next();
        else die("unknown flag: " + a);
    }

    auto kernel = make_kernel();
    if (describe)
    {
        kernel->tenonDescribe(std::cout);
        return 0;
    }
    if (op.empty() || out.empty())
        die("--op and --out are required (or --describe)");

    const uint64_t cyc_start = kernel->getCycle();
    uint64_t cyc_preload = 0, cyc_exec = 0, cyc_readback = 0;

    if (op == "GEMV")
    {
        // Vendor reference baseline (unmodified PIMSimulator kernel methods).
        if (weight.empty() || in.empty() || output_dim <= 0 || input_dim <= 0)
            die("GEMV needs --weight --in --output-dim --input-dim");
        NumpyBurstType W, x;
        W.loadFp16(weight);
        x.loadFp16(in);
        unsigned end_col = kernel->getResultColGemv(input_dim / 16, output_dim);
        kernel->preloadGemv(&W);
        kernel->runPIM();
        cyc_preload = kernel->getCycle() - cyc_start;
        uint64_t t1 = kernel->getCycle();
        kernel->executeGemv(&W, &x, false);
        kernel->runPIM();
        cyc_exec = kernel->getCycle() - t1;
        uint64_t t2 = kernel->getCycle();
        std::vector<BurstType> result(output_dim);
        kernel->readResult(result.data(), pimBankType::ODD_BANK, output_dim, 0, 0, end_col);
        kernel->runPIM();
        cyc_readback = kernel->getCycle() - t2;
        write_bursts(out, result.data(), result.size());
        std::cout << "VENDOR_GEMV pimcmdgen_calls=" << PIMCmdGen::getInvocationCount() << std::endl;
        print_cycles(kernel->getCycle() - cyc_start, cyc_preload, cyc_exec, cyc_readback);
        return 0;
    }
    if (op != "GENERIC")
        die("unknown --op (expected GENERIC or the vendor GEMV reference)");

    if (op_kind == "TEMPORAL_REDUCE")
    {
        if (cmds_path.empty())
            die("TEMPORAL_REDUCE requires --cmds");
        std::vector<PIMCmd> cmds = load_crf(cmds_path);
        auto files = split(inputs_arg, ',');
        if (files.size() != 3)
            die("TEMPORAL_REDUCE needs A,B,initial files");
        NumpyBurstType A, B, initial;
        A.loadFp16(files[0]);
        B.loadFp16(files[1]);
        initial.loadFp16(files[2]);
        vector<BurstType> result;
        try
        {
            kernel->runTemporalReduce(A, B, initial, out_rows, reduce_k, out_cols, cmds,
                                      temporal_audit_prefix, result, cyc_preload, cyc_exec,
                                      cyc_readback);
        }
        catch (const std::exception& error)
        {
            die(std::string("TEMPORAL_REDUCE failed: ") + error.what());
        }
        write_bursts(out, result.data(), result.size());
        std::cout << "PIM_TEMPORAL_REDUCE ordered_k=1 host_arithmetic=0 initial=1"
                  << " rows=" << out_rows << " depth=" << reduce_k << " columns=" << out_cols
                  << '\n';
        print_cycles(kernel->getCycle() - cyc_start, cyc_preload, cyc_exec, cyc_readback);
        return 0;
    }

    if (op_kind != "REDUCE" && op_kind != "ELTWISE")
        die("--op GENERIC needs --op-kind REDUCE, ELTWISE or TEMPORAL_REDUCE");
    if (crf_path.empty() || plan_path.empty())
        die("--op-kind " + op_kind + " needs --crf (the compiled program) and --plan");
    std::vector<PIMCmd> crf = load_crf(crf_path);
    auto files = split(inputs_arg, ',');
    std::vector<NumpyBurstType> loaded(files.size());
    for (size_t i = 0; i < files.size(); ++i) loaded[i].loadFp16(files[i]);

    const uint64_t generator_before = PIMCmdGen::getInvocationCount();
    TenonAudit audit;
    vector<BurstType> result;
    try
    {
        std::string plan_text = tenonReadPlanText(plan_path);
        if (op_kind == "REDUCE")
        {
            if (loaded.size() != 2)
                die("REDUCE needs --inputs matrix.npy,vectors.npy");
            TenonReducePlan plan = tenonParseReducePlan(plan_text);
            kernel->tenonRunReduce(plan, plan_text, crf, loaded[0], loaded[1], result, audit,
                                   cyc_preload, cyc_exec);
        }
        else
        {
            TenonEltwisePlan plan = tenonParseEltwisePlan(plan_text);
            kernel->tenonRunEltwise(plan, plan_text, crf, loaded, result, audit, cyc_preload,
                                    cyc_exec);
        }
    }
    catch (const std::exception& error)
    {
        die(op_kind + " conductor rejected the program: " + error.what());
    }
    uint64_t t2 = kernel->getCycle();
    kernel->runPIM();
    cyc_readback = kernel->getCycle() - t2;
    write_bursts(out, result.data(), result.size());
    const uint64_t generator_calls = PIMCmdGen::getInvocationCount() - generator_before;
    if (generator_calls != 0)
        die("the vendor CRF generator was invoked on the Tenon route");
    std::cout << "TENON_CONDUCTOR kind=" << op_kind << " crf_words=" << crf.size()
              << " pimcmdgen_calls=" << generator_calls << " crf_uploads=" << audit.crf_uploads
              << " passes=" << audit.passes << " matrix_preloads=" << audit.matrix_preloads
              << " mac_groups_even=" << audit.mac_groups[0]
              << " mac_groups_odd=" << audit.mac_groups[1]
              << " arith_groups_even=" << audit.arith_groups[0]
              << " arith_groups_odd=" << audit.arith_groups[1] << " fills=" << audit.fills
              << " drains=" << audit.drains << " host_vector_writes=" << audit.host_vector_writes
              << " column_commands=" << audit.column_commands << std::endl;
    print_cycles(kernel->getCycle() - cyc_start, cyc_preload, cyc_exec, cyc_readback);
    return 0;
}
