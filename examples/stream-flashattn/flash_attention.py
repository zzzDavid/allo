# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Systolic Array Flash Attention
=============================

Implements Flash Attention on a 4x4 systolic array using Allo's dataflow
programming model. Uses the same notation as:

  "From Online Softmax to FlashAttention" by Zihao Ye
  (UW CSE 599M, May 2023)

Algorithm (one-pass FlashAttention, per query row k):

  for i <- 1, N do
      x_i  <- Q[k,:] * K^T[:,i]
      m_i  <- max(m_{i-1}, x_i)
      d'_i <- d'_{i-1} * exp(m_{i-1} - m_i) + exp(x_i - m_i)
      o'_i <- o'_{i-1} * d'_{i-1} * exp(m_{i-1} - m_i) / d'_i
             + exp(x_i - m_i) / d'_i * V[i,:]
  end
  O[k,:] <- o'_N

Systolic array mapping (4x4 compute PEs with border I/O PEs):

       j=0      j=1      j=2      j=3      j=4      j=5
  i=0 [corner] [K0,V0v] [K1,V1v] [K2,V2v] [K3,V3v] [corner]
  i=1 [Q0>]   [PE(0,0)] [PE(0,1)] [PE(0,2)] [PE(0,3)] [->O0]
  i=2 [Q1>]   [PE(1,0)] [PE(1,1)] [PE(1,2)] [PE(1,3)] [->O1]
  i=3 [Q2>]   [PE(2,0)] [PE(2,1)] [PE(2,2)] [PE(2,3)] [->O2]
  i=4 [Q3>]   [PE(3,0)] [PE(3,1)] [PE(3,2)] [PE(3,3)] [->O3]
  i=5 [corner] [drainK] [drainK] [drainK] [drainK] [corner]

  - Q[k,:] vectors stream horizontally (left -> right), each row independent
  - K[i,:] and V[i,:] vectors stream vertically (top -> bottom), daisy-chained
  - Flash attention state (m, d', o') propagates left -> right through compute PEs
  - All 4 Q rows are processed in parallel; the state chain is the sequential part
"""

import os

import allo
from allo.ir.types import float32, Stream
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

# Problem dimensions
L = 4  # Sequence length (number of tokens)
D = 4  # Head dimension

# PE array: 4x4 compute core + 1-wide border for I/O
P0 = L + 2  # 6 rows
P1 = L + 2  # 6 columns


@df.region()
def top(Q: float32[L, D], K: float32[L, D], V: float32[L, D], O: float32[L, D]):
    # --- Data streams ---
    fifo_Q: Stream[float32[D], 4][P0, P1]  # Q row vectors, horizontal
    fifo_K: Stream[float32[D], 4][P0, P1]  # K row vectors, vertical
    fifo_V: Stream[float32[D], 4][P0, P1]  # V row vectors, vertical

    # --- Flash attention state streams (horizontal, left -> right) ---
    fifo_m: Stream[float32, 4][P0, P1]     # running max m_i
    fifo_d: Stream[float32, 4][P0, P1]     # running normalizer d'_i
    fifo_o: Stream[float32[D], 4][P0, P1]  # running output o'_i

    @df.kernel(mapping=[P0, P1], args=[Q, K, V, O])
    def pe(
        local_Q: float32[L, D],
        local_K: float32[L, D],
        local_V: float32[L, D],
        local_O: float32[L, D],
    ):
        i, j = df.get_pid()

        # ---- Corner PEs: no-op ----
        with allo.meta_if(i in {0, P0 - 1} and j in {0, P1 - 1}):
            pass

        # ---- Top row (i=0, j=1..L): feed K[j-1,:] and V[j-1,:] downward ----
        with allo.meta_elif(i == 0):
            k_vec: float32[D] = 0
            v_vec: float32[D] = 0
            for dd in range(D):
                k_vec[dd] = local_K[j - 1, dd]
                v_vec[dd] = local_V[j - 1, dd]
            fifo_K[i + 1, j].put(k_vec)
            fifo_V[i + 1, j].put(v_vec)

        # ---- Left column (i=1..L, j=0): feed Q[i-1,:] and init state ----
        with allo.meta_elif(j == 0):
            q_vec: float32[D] = 0
            for dd in range(D):
                q_vec[dd] = local_Q[i - 1, dd]
            fifo_Q[i, j + 1].put(q_vec)
            # Initial state: m_0 = -inf, d'_0 = 0, o'_0 = 0
            m_init: float32 = -1e30
            d_init: float32 = 0.0
            o_init: float32[D] = 0
            fifo_m[i, j + 1].put(m_init)
            fifo_d[i, j + 1].put(d_init)
            fifo_o[i, j + 1].put(o_init)

        # ---- Bottom row (i=L+1, j=1..L): drain K, V ----
        with allo.meta_elif(i == P0 - 1):
            k_drain: float32[D] = fifo_K[i, j].get()
            v_drain: float32[D] = fifo_V[i, j].get()

        # ---- Right column (i=1..L, j=L+1): drain Q, collect O[k,:] = o'_N ----
        with allo.meta_elif(j == P1 - 1):
            q_drain: float32[D] = fifo_Q[i, j].get()
            m_drain: float32 = fifo_m[i, j].get()
            d_drain: float32 = fifo_d[i, j].get()
            o_final: float32[D] = fifo_o[i, j].get()
            for dd in range(D):
                local_O[i - 1, dd] = o_final[dd]

        # ---- Compute PEs (i=1..L, j=1..L): one-pass FlashAttention ----
        with allo.meta_else():
            # Receive streamed data from neighbors
            q_vec: float32[D] = fifo_Q[i, j].get()  # Q[k,:]
            k_vec: float32[D] = fifo_K[i, j].get()  # K^T[:,i] = K[i,:]
            v_vec: float32[D] = fifo_V[i, j].get()  # V[i,:]

            # --- x_i = Q[k,:] * K^T[:,i]  (dot product) ---
            x: float32 = 0.0
            for dd in range(D):
                x += q_vec[dd] * k_vec[dd]

            # Receive state from left neighbor
            m_prev: float32 = fifo_m[i, j].get()    # m_{i-1}
            d_prev: float32 = fifo_d[i, j].get()    # d'_{i-1}
            o_prev: float32[D] = fifo_o[i, j].get()  # o'_{i-1}

            # --- m_i = max(m_{i-1}, x_i) ---
            m_new: float32 = m_prev
            if x > m_prev:
                m_new = x

            # --- d'_i = d'_{i-1} * exp(m_{i-1} - m_i) + exp(x_i - m_i) ---
            exp_prev: float32 = allo.exp(m_prev - m_new)
            exp_x: float32 = allo.exp(x - m_new)
            d_new: float32 = d_prev * exp_prev + exp_x

            # --- o'_i = o'_{i-1} * d'_{i-1}*exp(m_{i-1}-m_i)/d'_i
            #           + exp(x_i-m_i)/d'_i * V[i,:] ---
            alpha: float32 = d_prev * exp_prev / d_new
            beta: float32 = exp_x / d_new
            o_new: float32[D] = 0
            for dd in range(D):
                o_new[dd] = o_prev[dd] * alpha + beta * v_vec[dd]

            # Forward data to neighbors
            fifo_Q[i, j + 1].put(q_vec)  # Q flows right
            fifo_K[i + 1, j].put(k_vec)  # K flows down
            fifo_V[i + 1, j].put(v_vec)  # V flows down

            # Forward updated state to right neighbor
            fifo_m[i, j + 1].put(m_new)  # m_i ->
            fifo_d[i, j + 1].put(d_new)  # d'_i ->
            fifo_o[i, j + 1].put(o_new)  # o'_i ->


def flash_attention_ref(Q, K, V):
    """Reference: O = softmax(Q @ K^T) @ V"""
    X = Q @ K.T
    m = np.max(X, axis=-1, keepdims=True)
    A = np.exp(X - m)
    A = A / np.sum(A, axis=-1, keepdims=True)
    return A @ V


def test_flash_attention():
    np.random.seed(42)
    Q_np = np.random.randn(L, D).astype(np.float32)
    K_np = np.random.randn(L, D).astype(np.float32)
    V_np = np.random.randn(L, D).astype(np.float32)
    O_np = np.zeros((L, D), dtype=np.float32)

    O_ref = flash_attention_ref(Q_np, K_np, V_np)

    print("Building dataflow simulator...")
    sim_mod = df.build(top, target="simulator")

    print("Running simulation...")
    sim_mod(Q_np, K_np, V_np, O_np)

    print("Q =")
    print(Q_np)
    print("\nK =")
    print(K_np)
    print("\nV =")
    print(V_np)
    print("\nExpected O =")
    print(O_ref)
    print("\nComputed O =")
    print(O_np)

    max_err = np.max(np.abs(O_np - O_ref))
    print(f"\nMax absolute error: {max_err:.2e}")

    np.testing.assert_allclose(O_np, O_ref, atol=1e-5)
    print("Flash Attention dataflow simulator PASSED!")

    # --- HLS C-simulation (csim) ---
    if hls.is_available("vitis_hls"):
        print("\nBuilding Vitis HLS csim...")
        s = df.customize(top)
        s.partition("top:Q", dim=0)
        s.partition("top:K", dim=0)
        s.partition("top:V", dim=0)
        s.partition("top:O", dim=0)
        project_dir = "/scratch/nz264/flash_attention_csim"
        os.makedirs(project_dir, exist_ok=True)
        mod = s.build(
            target="vitis_hls", mode="csim", project=project_dir
        )
        O_hls = np.zeros((L, D), dtype=np.float32)
        print("Running csim...")
        mod(Q_np, K_np, V_np, O_hls)

        print("\nCsim O =")
        print(O_hls)
        max_err_hls = np.max(np.abs(O_hls - O_ref))
        print(f"Max absolute error (csim vs ref): {max_err_hls:.2e}")

        np.testing.assert_allclose(O_hls, O_ref, atol=1e-5)
        print("Flash Attention Vitis HLS csim PASSED!")
    else:
        print("\nVitis HLS not available, skipping csim.")


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "128"
    test_flash_attention()
    del os.environ["OMP_NUM_THREADS"]
