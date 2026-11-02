// Ordered FP16 contraction conductor. Arithmetic remains in PIMRank/PIMBlock.
// Each physical lane owns one output row; K is issued in ascending source order.
// This conservative path uses one GRF slot and never reduces lanes on the host.
void PIMKernel::runTemporalReduce(
    NumpyBurstType& weights, NumpyBurstType& rhs, NumpyBurstType& initial,
    int rows, int depth, int columns, vector<PIMCmd>& cmds,
    const string& audit_prefix, vector<BurstType>& output,
    uint64_t& preload, uint64_t& execution, uint64_t& readback)
{
    if (rows <= 0 || depth <= 0 || depth > 4096 || columns <= 0 ||
        num_pim_ranks_ != 1 || audit_prefix.empty())
        throw runtime_error("temporal reduction requires M,N>0, 1<=K<=4096, one rank and audit prefix");
    if (weights.u16Data.size() != (size_t)rows * depth ||
        rhs.u16Data.size() != (size_t)depth * columns ||
        initial.u16Data.size() != (size_t)rows * columns)
        throw runtime_error("temporal reduction input sizes disagree with M,K,N");
    vector<PIMCmd> expected = {
        PIMCmd(PIMCmdType::MOV, PIMOpdType::GRF_B, PIMOpdType::EVEN_BANK),
        PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A,
               PIMOpdType::EVEN_BANK),
        PIMCmd(PIMCmdType::JUMP, depth - 1, 2),
        PIMCmd(PIMCmdType::NOP, 0), PIMCmd(PIMCmdType::EXIT, 0)};
    if (cmds.size() != expected.size())
        throw runtime_error("temporal CRF must contain exactly five words");
    for (size_t i = 0; i < cmds.size(); ++i)
        if (cmds[i].toInt() != expected[i].toInt())
            throw runtime_error("temporal CRF does not match its ordered execution contract");

    ofstream addresses(audit_prefix + ".addresses.csv");
    ofstream packed(audit_prefix + ".writes.bin", ios::binary);
    ofstream raw_output(audit_prefix + ".readback.bin", ios::binary);
    if (!addresses || !packed || !raw_output)
        throw runtime_error("cannot create temporal audit files");
    addresses << "kind,tile,n,k,channel,block,address,word_offset\n";
    size_t word_offset = 0;
    auto record = [&](const char* kind, int tile, int n, int k, int ch, int pb,
                      uint64_t address, BurstType* data) {
        addresses << kind << ',' << tile << ',' << n << ',' << k << ','
                  << ch << ',' << pb << ',' << address << ',' << word_offset << '\n';
        if (data) {
            packed.write(reinterpret_cast<const char*>(data->u16Data_), 32);
            word_offset += 16;
        }
    };
    auto physical = [&](int ch, int pb, int parity, unsigned row, unsigned col) {
        int global_bank = 2 * pb + parity;
        int banks_per_group = num_banks_ / num_bank_groups_;
        return pim_addr_mgr_->addrGenSafe(ch, 0, global_bank / banks_per_group,
                                         global_bank % banks_per_group, row, col);
    };
    const int blocks = num_pim_chans_ * num_pim_blocks_;
    const int tile_rows = blocks * 16;
    output.resize(((size_t)rows * columns + 15) / 16);
    setControl(&bst_hab_pim_, true, getToggleCond(), false, false);
    setControl(&bst_hab_, false, getToggleCond(), false, false);
    for (int tile = 0; tile * tile_rows < rows; ++tile) {
        for (int n = 0; n < columns; ++n) {
            // Backing buffers remain stable until all queued transactions finish.
            vector<BurstType> bank_data((size_t)(depth + 1) * blocks);
            vector<BurstType> broadcasts(depth);
            vector<BurstType> results(blocks);
            uint64_t start = getCycle();
            for (int k = -1; k < depth; ++k) {
                for (int ch = 0; ch < num_pim_chans_; ++ch) {
                    for (int pb = 0; pb < num_pim_blocks_; ++pb) {
                        int block = ch * num_pim_blocks_ + pb;
                        auto& burst = bank_data[(size_t)(k + 1) * blocks + block];
                        for (int lane = 0; lane < 16; ++lane) {
                            int m = tile * tile_rows + block * 16 + lane;
                            if (m < rows)
                                burst.u16Data_[lane] = k < 0
                                    ? initial.u16Data[(size_t)m * columns + n]
                                    : weights.u16Data[(size_t)m * depth + k];
                        }
                        auto addr = physical(ch, pb, 0, k + 1, 0);
                        record(k < 0 ? "initial" : "weight", tile, n, k, ch, pb, addr, &burst);
                        mem_->addTransaction(true, addr, "TEMPORAL_PRELOAD", &burst);
                    }
                }
            }
            runPIM();
            preload += getCycle() - start;
            start = getCycle();
            parkIn();
            changePIMMode(dramMode::SB, dramMode::HAB);
            programCrf(cmds);
            changePIMMode(dramMode::HAB, dramMode::HAB_PIM);
            addTransactionAll(false, 0, 0, 0, 0, "TEMPORAL_INITIAL", &null_bst_, true, 1);
            for (int k = 0; k < depth; ++k) {
                auto& broadcast = broadcasts[k];
                for (int lane = 0; lane < 16; ++lane)
                    broadcast.u16Data_[lane] = rhs.u16Data[(size_t)k * columns + n];
                for (int ch = 0; ch < num_pim_chans_; ++ch) {
                    auto addr = pim_addr_mgr_->addrGen(ch, 0, 0, 1, pim_reg_ra_, 0x8);
                    record("rhs", tile, n, k, ch, -1, addr, &broadcast);
                    mem_->addTransaction(true, addr, "TEMPORAL_RHS", &broadcast);
                    mem_->addBarrier(ch);
                }
                // Fixed register indices: no getGrfIdxHigh / cross-lane partials.
                addTransactionAll(false, 0, 0, k + 1, 0, "TEMPORAL_MAC", &null_bst_, true, 1);
            }
            addTransactionAll(true, 0, 1, 0, 0, "TEMPORAL_DRAIN", &null_bst_, true, 1);
            changePIMMode(dramMode::HAB_PIM, dramMode::HAB);
            changePIMMode(dramMode::HAB, dramMode::SB);
            parkOut();
            runPIM();
            execution += getCycle() - start;
            start = getCycle();
            for (int ch = 0; ch < num_pim_chans_; ++ch) {
                for (int pb = 0; pb < num_pim_blocks_; ++pb) {
                    int block = ch * num_pim_blocks_ + pb;
                    auto addr = physical(ch, pb, 1, 0, 0);
                    record("output", tile, n, depth, ch, pb, addr, nullptr);
                    mem_->addTransaction(false, addr, "TEMPORAL_READBACK", &results[block]);
                }
            }
            runPIM();
            readback += getCycle() - start;
            for (int block = 0; block < blocks; ++block) {
                raw_output.write(reinterpret_cast<const char*>(results[block].u16Data_), 32);
                for (int lane = 0; lane < 16; ++lane) {
                    int m = tile * tile_rows + block * 16 + lane;
                    if (m < rows) {
                        size_t i = (size_t)m * columns + n;
                        output[i / 16].u16Data_[i % 16] = results[block].u16Data_[lane];
                    }
                }
            }
        }
    }
    if (!addresses || !packed || !raw_output)
        throw runtime_error("temporal audit write failed");
}
