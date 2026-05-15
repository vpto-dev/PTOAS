# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Flash Attention redesign sketch.

This file is intentionally a design demo rather than runnable ``ptodsl`` code.
The goal is to make the *proposed* API layering explicit and keep the semantic
contracts clean:

    flash_attention(...)           user-facing wrapper
      └─ @pto.jit flash_attention_kernel
           ├─ Tile Ops                 tload / tstore at the GM↔UB boundary
           └─ @pto.ukernel  one KV-block worth of MTE/sync orchestration
                ├─ @pto.cube   matrix products (QK^T and P@V)
                ├─ @pto.simd   row-wise online softmax
                └─ @pto.simt   scalar metadata and output blending

Design rules illustrated here:

1. ``@pto.jit`` marks a launchable kernel template.  It owns JIT compilation,
   cache lookup, and runtime launch binding, instead of forcing users to hop
   through extra builder objects for common cases.
2. The Python wrapper owns ergonomic runtime concerns such as output allocation,
   default stream handling, and extracting shape/stride metadata from tensors.
3. ``@pto.jit`` also owns the top-level logical tiling, tile allocation, and
   loop scheduling for one already-selected per-head 2D slice.  It should not
   manually spell low-level DMA details for every micro step.
4. ``ukernel`` owns the per-block execution sandwich: stage the current K/V
   block with explicit micro-instructions, synchronize, call hardware-bound
   sub-kernels, and manage scratch/state.
5. ``@pto.jit`` may use tile ops such as ``tload`` / ``tstore`` at the logical
   scheduling boundary, but ``ukernel`` stays below that abstraction level.
   Once execution enters ``ukernel``, GM<->UB movement is expressed with
   MTE micro-instructions such as ``mte_load`` instead of tile ops.
   ``mte_load`` / ``mte_store`` accept partitions and tiles directly,
   deriving strides and burst sizes from the type metadata.
6. ``simd`` / ``simt`` / ``cube`` are hardware boundaries. They do not expose
   vreg values across the function boundary. Data crosses the boundary through
   UB-backed tiles or typed UB pointers only.
7. L3 sub-kernels can also be called directly from ``@pto.jit`` (compiler
   handles MTE + sync) or written inline as context managers
   (``with pto.simd():`` etc.). This sketch uses the explicit
   ``@pto.ukernel`` → L3 path for full micro-instruction control, but
   simpler kernels can skip the ukernel layer.
8. Online-softmax state is made explicit with ping-pong tiles
   (``m_prev``/``m_next``, ``l_prev``/``l_next``, ``o_prev``/``o_next``).
   Hiding these dependencies with in-place aliases makes the algorithm harder
   to read and obscures what the DSL needs to express.

The API spellings below are approximate and intentionally favor the redesign
surface over today's exact binding details.

Because this sketch targets a tracing-style frontend, any control flow that
must reach MLIR is expressed with structured DSL constructs such as
``pto.for_`` instead of native Python ``for`` loops.

Scalar literals and simple index/integer conversions are also shown in their
authored form.  The intended frontend behavior is to lift Python ``int``
literals and obvious scalar arithmetic into the corresponding MLIR scalar ops
implicitly, rather than forcing authors to spell ``pto.const(...)`` or
``index_cast(...)`` at every use site.
"""

from ptodsl import pto


# ═══════════════════════════════════════════════════════════════════════════════
# Public API sketch
# ═══════════════════════════════════════════════════════════════════════════════
#
# This section intentionally sketches the *desired* public surface, not today's
# exact implementation details.  The split follows the common industry pattern:
#
# - a user-facing tensor wrapper
# - a launchable JIT kernel entry
# - hardware-bound sub-kernels below it
#
# The low-level kernel body should not double as the user-facing runtime API.
#
# Two intended usage styles:
#
# 1. Direct call (most users):
#      out = flash_attention(Q, K, V, causal=True)
#
# 2. Compile first, then launch repeatedly:
#      compiled = flash_attention_kernel.compile(BLOCK_Q=128, BLOCK_KV=128, CAUSAL=True)
#      compiled[batch * heads, stream](
#          Q, K, V, O,
#      )

def flash_attention(
    Q,
    K,
    V,
    *,
    O=None,
    causal=False,
    block_q=128,
    block_kv=128,
    stream=None,
):
    """
    User-facing convenience wrapper.

    This is the API most end users should call.  It mirrors mainstream tensor
    libraries: infer runtime metadata from tensors, allocate the output when the
    caller does not provide one, then compile and launch the JIT kernel.
    """
    if O is None:
        O = pto.empty_like(Q)

    batch, seq_q, heads, dim = Q.shape
    _, seq_k, _, _ = K.shape

    compiled = flash_attention_kernel.compile(
        BLOCK_Q=block_q,
        BLOCK_KV=block_kv,
        CAUSAL=causal,
    )

    compiled[batch * heads, stream](Q, K, V, O)
    return O

@pto.jit(target="a5")
def flash_attention_kernel(
    Q,                      # Python/framework tensor, logical [batch, seq_q, heads, dim]
    K,                      # Python/framework tensor, logical [batch, seq_k, heads, dim]
    V,                      # Python/framework tensor, logical [batch, seq_k, heads, dim]
    O,                      # Python/framework tensor, logical [batch, seq_q, heads, dim]
    *,
    BLOCK_Q: pto.constexpr = 128,
    BLOCK_KV: pto.constexpr = 128,
    CAUSAL: pto.constexpr = False,
    NUM_STAGES: pto.constexpr = 2,
):
    """
    Launchable device entry.

    ``@pto.jit`` is the compile + launch boundary.  Inputs/outputs at this
    boundary are Python-native tensor objects; PTO-specific ``TensorView``
    descriptors are materialized inside the JIT body rather than exposed in the
    public signature.  Tile sizes and specialization knobs remain constexpr
    metadata.

    A launch instance is responsible for one ``(batch, head)`` slice.  The
    per-slice logical tiling is expressed directly in this top-level JIT entry.
    """
    batch, seq_q, heads, dim = Q.shape
    _, seq_k, _, _ = K.shape

    q_view = pto.make_tensor_view(Q, shape=[batch, seq_q, heads, dim], strides=Q.strides)
    k_view = pto.make_tensor_view(K, shape=[batch, seq_k, heads, dim], strides=K.strides)
    v_view = pto.make_tensor_view(V, shape=[batch, seq_k, heads, dim], strides=V.strides)
    o_view = pto.make_tensor_view(O, shape=[batch, seq_q, heads, dim], strides=O.strides)

    # Make the SPMD launch contract explicit in the authored surface.
    # This sketch uses one block per (batch, head) slice and does not further
    # split work across subblocks, but the runtime indices still belong in a
    # realistic launchable entry.
    block_idx = pto.get_block_idx()
    block_num = pto.get_block_num()
    subblock_idx = pto.get_subblock_idx()
    subblock_num = pto.get_subblock_num()

    # Current mapping:
    # - launch grid = batch * heads
    # - block_idx selects one (batch, head) slice
    # - subblock_idx is queried explicitly, but no extra intra-block partition
    #   is modeled in this sketch yet
    _ = block_num
    _ = subblock_idx
    _ = subblock_num

    batch_idx = block_idx // heads
    head_idx = block_idx % heads

    q_head = pto.select_head_view(
        q_view,
        batch=batch_idx,
        head=head_idx,
        shape=[seq_q, dim],
    )
    k_head = pto.select_head_view(
        k_view,
        batch=batch_idx,
        head=head_idx,
        shape=[seq_k, dim],
    )
    v_head = pto.select_head_view(
        v_view,
        batch=batch_idx,
        head=head_idx,
        shape=[seq_k, dim],
    )
    o_head = pto.select_head_view(
        o_view,
        batch=batch_idx,
        head=head_idx,
        shape=[seq_q, dim],
    )

    Br = BLOCK_Q
    Bc = BLOCK_KV

    q_blocks = (seq_q + Br - 1) // Br
    kv_blocks = (seq_k + Bc - 1) // Bc

    # UB resident logical tiles for one selected (batch, head) slice.
    q_tile = pto.alloc_tile(shape=[Br, dim], dtype=pto.f32)
    k_tile = pto.alloc_tile(shape=[Bc, dim], dtype=pto.f32)
    v_tile = pto.alloc_tile(shape=[Bc, dim], dtype=pto.f32)

    o_prev_tile = pto.alloc_tile(shape=[Br, dim], dtype=pto.f32)
    o_next_tile = pto.alloc_tile(shape=[Br, dim], dtype=pto.f32)
    m_prev_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32)
    m_next_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32)
    l_prev_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32)
    l_next_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32)

    s_tile = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32)
    p_tile = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32)
    pv_tile = pto.alloc_tile(shape=[Br, dim], dtype=pto.f32)
    alpha_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32)
    beta_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32)

    # Cube-local scratch is explicit; it should not be conflated with UB tiles.
    q_l0a = pto.alloc_tile(shape=[Br, dim], dtype=pto.f16, memory_space=pto.MemorySpace.LEFT)
    p_l0a = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f16, memory_space=pto.MemorySpace.LEFT)
    rhs_l0b = pto.alloc_tile(shape=[Bc, dim], dtype=pto.f16, memory_space=pto.MemorySpace.RIGHT)
    qk_acc_tile = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, memory_space=pto.MemorySpace.ACC)
    pv_acc_tile = pto.alloc_tile(shape=[Br, dim], dtype=pto.f32, memory_space=pto.MemorySpace.ACC)

    # SIMT metadata buffer.  A tiny raw-pointer island is acceptable at the
    # ukernel boundary because this is scalar control data, not user-facing math.
    meta_tile = pto.alloc_tile(shape=[3, 1], dtype=pto.i32)
    meta_ptr = pto.tile_buf_addr(meta_tile)

    with pto.for_(0, q_blocks, step=1) as qi:
        q_part = pto.partition_view(q_head, offsets=[qi * Br, 0], sizes=[Br, dim])
        o_part = pto.partition_view(o_head, offsets=[qi * Br, 0], sizes=[Br, dim])

        pto.tload(q_part, q_tile)

        # Initial online-softmax state for this Q block.
        # ``CAUSAL`` is threaded at the API boundary even though the masking
        # details are intentionally omitted from this design-focused sketch.
        m_prev_tile.fill(float("-inf"))
        l_prev_tile.fill(0.0)
        o_prev_tile.fill(0.0)

        kv_loop = pto.for_(0, kv_blocks, step=1).carry(
            m=m_prev_tile,
            l=l_prev_tile,
            o=o_prev_tile,
        )
        with kv_loop:
            kj = kv_loop.iv
            m_cur = kv_loop.m
            l_cur = kv_loop.l
            o_cur = kv_loop.o
            k_part = pto.partition_view(k_head, offsets=[kj * Bc, 0], sizes=[Bc, dim])
            v_part = pto.partition_view(v_head, offsets=[kj * Bc, 0], sizes=[Bc, dim])

            kv_block_process(
                q_tile,
                k_part,
                v_part,
                k_tile,
                v_tile,
                o_cur,
                o_next_tile,
                m_cur,
                l_cur,
                m_next_tile,
                l_next_tile,
                s_tile,
                p_tile,
                pv_tile,
                alpha_tile,
                beta_tile,
                q_l0a,
                p_l0a,
                rhs_l0b,
                qk_acc_tile,
                pv_acc_tile,
                meta_ptr,
            )

            # Loop-carried state is still explicit, but the authored surface no
            # longer mirrors raw scf.iter_args / scf.yield spellings.
            kv_loop.update(
                m=m_next_tile,
                l=l_next_tile,
                o=o_next_tile,
            )

        o_final_tile = kv_loop.final("o")
        pto.tstore(o_final_tile, o_part)


# ═══════════════════════════════════════════════════════════════════════════════
# Level 3: hardware-bound sub-kernels
# ═══════════════════════════════════════════════════════════════════════════════
#
# Boundary contract:
# - Tile arguments are UB-backed or cube-local buffers carrying addressable
#   storage.
# - No vector register escapes a simd function.
# - No implicit global-memory access happens inside these kernels.


@pto.cube
def qk_matmul(
    q_tile: pto.Tile,      # UB, [Br, dim]
    k_tile: pto.Tile,      # UB, [Bc, dim]
    q_l0a: pto.Tile,       # LEFT scratch
    k_l0b: pto.Tile,       # RIGHT scratch
    s_acc: pto.Tile,       # ACC scratch
    s_tile: pto.Tile,      # UB, [Br, Bc] output
):
    """
    Compute ``S = Q @ K^T`` for one attention block.

    The key point for the redesign is that the cube kernel consumes UB tiles and
    explicit cube-local scratch, rather than pretending a UB tile can also stand
    in for LEFT/RIGHT/ACC state.
    """
    m = pto.tile_valid_rows(q_tile)
    k = pto.tile_valid_cols(q_tile)
    n = pto.tile_valid_rows(k_tile)

    # Caller owns scratch lifetime.  The cube kernel only expresses dataflow.
    pto.mte_l1_l0a(q_tile, q_l0a, m, k)
    pto.mte_l1_l0b(k_tile, k_l0b, k, n, transpose=True)
    pto.mad(q_l0a, k_l0b, s_acc)
    pto.mte_l0c_ub(s_acc, s_tile, m, n)


@pto.cube
def pv_matmul(
    p_tile: pto.Tile,      # UB, [Br, Bc]
    v_tile: pto.Tile,      # UB, [Bc, dim]
    p_l0a: pto.Tile,       # LEFT scratch (reused)
    v_l0b: pto.Tile,       # RIGHT scratch (reused)
    pv_acc: pto.Tile,      # ACC scratch (reused)
    pv_tile: pto.Tile,     # UB, [Br, dim] output
):
    """
    Compute ``PV = P @ V`` for the current block.

    This keeps the second matrix product on the cube path as well, instead of
    accidentally collapsing it into an elementwise vector expression.
    """
    m = pto.tile_valid_rows(p_tile)
    k = pto.tile_valid_cols(p_tile)
    n = pto.tile_valid_cols(v_tile)

    pto.mte_l1_l0a(p_tile, p_l0a, m, k)
    pto.mte_l1_l0b(v_tile, v_l0b, k, n)
    pto.mad(p_l0a, v_l0b, pv_acc)
    pto.mte_l0c_ub(pv_acc, pv_tile, m, n)


@pto.simd
def online_softmax_rows(
    s_tile: pto.Tile,          # UB, [Br, Bc]
    p_tile: pto.Tile,          # UB, [Br, Bc], output
    m_prev_tile: pto.Tile,     # UB, [Br, 1]
    l_prev_tile: pto.Tile,     # UB, [Br, 1]
    m_next_tile: pto.Tile,     # UB, [Br, 1], output
    l_next_tile: pto.Tile,     # UB, [Br, 1], output
    alpha_tile: pto.Tile,      # UB, [Br, 1], output
    beta_tile: pto.Tile,       # UB, [Br, 1], output
    row_start: pto.i32,
    row_stop: pto.i32,
    valid_cols: pto.i32,
):
    """
    Per-row online softmax update.

    For each active row::

        m_next = max(m_prev, row_max(S))
        P      = exp(S - m_next)
        l_next = l_prev * exp(m_prev - m_next) + row_sum(P)
        alpha  = l_prev * exp(m_prev - m_next) / l_next
        beta   = 1 / l_next

    ``alpha`` and ``beta`` are kept explicitly because the output update needs
    both the old accumulator and the newly computed ``P @ V`` contribution.
    """
    with pto.for_(row_start, row_stop, step=1) as row:
        col_mask = pto.make_mask(pto.f32, valid_cols)

        s_row = pto.vlds(s_tile[row, 0:])
        m_prev = scalar.load(m_prev_tile[row, 0])
        l_prev = scalar.load(l_prev_tile[row, 0])

        row_max = pto.vcgmax(s_row, col_mask)
        m_next = scalar.max(m_prev, row_max)

        s_shifted = pto.vsubs(s_row, m_next, col_mask)
        p_row = pto.vexp(s_shifted, col_mask)

        row_sum = pto.vcgadd(p_row, col_mask)
        l_scaled = l_prev * scalar.exp(m_prev - m_next)
        l_next = l_scaled + row_sum

        alpha = l_scaled / l_next
        beta = 1.0 / l_next

        pto.vsts(p_row, p_tile[row, 0:], col_mask)
        scalar.sts(m_next_tile[row, 0], m_next)
        scalar.sts(l_next_tile[row, 0], l_next)
        scalar.sts(alpha_tile[row, 0], alpha)
        scalar.sts(beta_tile[row, 0], beta)


@pto.simt
def blend_output_rows(
    o_prev_tile: pto.Tile,      # UB, [Br, dim]
    pv_tile: pto.Tile,          # UB, [Br, dim]
    alpha_tile: pto.Tile,       # UB, [Br, 1]
    beta_tile: pto.Tile,        # UB, [Br, 1]
    o_next_tile: pto.Tile,      # UB, [Br, dim], output
    row_start: pto.i32,
    row_stop: pto.i32,
    valid_dim: pto.i32,
):
    """
    Update the output accumulator with SIMT-style scalar element work::

        O_next[row, col] = alpha[row] * O_prev[row, col] + beta[row] * PV[row, col]

    This intentionally contrasts with ``online_softmax_rows``: the softmax step
    stays on the SIMD path because it is dominated by row-wise vector math,
    while the final blend is expressed here as explicit scalar work-items over
    the tile domain.
    """
    with pto.for_(row_start, row_stop, step=1) as row:
        alpha = scalar.load(alpha_tile[row, 0])
        beta = scalar.load(beta_tile[row, 0])

        with pto.for_(0, valid_dim, step=1) as col:
            o_prev = scalar.load(o_prev_tile[row, col])
            pv_val = scalar.load(pv_tile[row, col])

            o_next = alpha * o_prev + beta * pv_val
            scalar.sts(o_next_tile[row, col], o_next)


@pto.simt
def materialize_tile_bounds(
    meta_ptr: pto.ptr(pto.i32, pto.MemorySpace.UB),   # [out] {row_start, row_stop, valid_cols}
    valid_rows: pto.i32,
    valid_cols: pto.i32,
):
    """
    Materialize tile-local loop bounds for the current block.

    The SIMT kernel stays intentionally small here: it is responsible for
    scalar control metadata, not for rewriting the vector or cube logic.
    """
    scalar.sts(meta_ptr + 0, 0)
    scalar.sts(meta_ptr + 4, valid_rows)
    scalar.sts(meta_ptr + 8, valid_cols)


# ═══════════════════════════════════════════════════════════════════════════════
# Level 2: ukernel — one KV block worth of execution orchestration
# ═══════════════════════════════════════════════════════════════════════════════


@pto.ukernel
def kv_block_process(
    q_tile: pto.Tile,                # UB, reused across inner KV loop
    k_part: pto.PartitionTensorView, # GM view for current K block
    v_part: pto.PartitionTensorView, # GM view for current V block
    k_tile: pto.Tile,                # UB scratch
    v_tile: pto.Tile,                # UB scratch
    o_prev_tile: pto.Tile,           # UB state
    o_next_tile: pto.Tile,           # UB state
    m_prev_tile: pto.Tile,           # UB state
    l_prev_tile: pto.Tile,           # UB state
    m_next_tile: pto.Tile,           # UB state
    l_next_tile: pto.Tile,           # UB state
    s_tile: pto.Tile,                # UB scratch for QK^T
    p_tile: pto.Tile,                # UB scratch for probabilities
    pv_tile: pto.Tile,               # UB scratch for P@V
    alpha_tile: pto.Tile,            # UB scratch
    beta_tile: pto.Tile,             # UB scratch
    q_l0a: pto.Tile,                 # LEFT scratch for Q
    p_l0a: pto.Tile,                 # LEFT scratch for P
    rhs_l0b: pto.Tile,               # RIGHT scratch, reused by K/V
    qk_acc_tile: pto.Tile,           # ACC scratch for QK^T
    pv_acc_tile: pto.Tile,           # ACC scratch for P@V
    meta_ptr: pto.ptr(pto.i32, pto.MemorySpace.UB),
):
    """
    Process one KV block against an already-loaded Q tile.

    The ukernel owns:
    - staging the current K/V block into reusable UB scratch with explicit
      DMA-style micro-instructions,
    - synchronizing the hand-off between MTE, cube, simd, and simt stages,
    - wiring together the explicit state transition
      (prev -> next for m/l/o).
    """
    # Current-block GM->UB staging via MTE micro-instructions.
    pto.mte_load(k_part, k_tile)
    pto.mte_load(v_part, v_tile)
    pto.mem_bar(pto.BarrierType.SYNC)

    materialize_tile_bounds(
        meta_ptr,
        pto.tile_valid_rows(q_tile),
        pto.tile_valid_rows(k_tile),
    )
    row_start = scalar.load(meta_ptr + 0)
    row_stop = scalar.load(meta_ptr + 4)
    valid_cols = scalar.load(meta_ptr + 8)

    # 1. S = Q @ K^T
    qk_matmul(q_tile, k_tile, q_l0a, rhs_l0b, qk_acc_tile, s_tile)
    pto.mem_bar(pto.BarrierType.SYNC)

    # 2. Row-wise online softmax over S
    online_softmax_rows(
        s_tile,
        p_tile,
        m_prev_tile,
        l_prev_tile,
        m_next_tile,
        l_next_tile,
        alpha_tile,
        beta_tile,
        row_start,
        row_stop,
        valid_cols,
    )
    pto.mem_bar(pto.BarrierType.SYNC)

    # 3. PV = P @ V
    pv_matmul(p_tile, v_tile, p_l0a, rhs_l0b, pv_acc_tile, pv_tile)
    pto.mem_bar(pto.BarrierType.SYNC)

    # 4. O_next = alpha * O_prev + beta * PV
    blend_output_rows(
        o_prev_tile,
        pv_tile,
        alpha_tile,
        beta_tile,
        o_next_tile,
        row_start,
        row_stop,
        pto.tile_valid_cols(v_tile),
    )
    pto.mem_bar(pto.BarrierType.SYNC)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer summary
# ═══════════════════════════════════════════════════════════════════════════════
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │ L0  Python wrapper   flash_attention(...)                                 │
# │                                                                            │
# │   output allocation, shape/stride extraction, compile, launch             │
# │                                                                            │
# │   Key idea: user-facing tensor API, not IR authoring.                     │
# ├──────────────────────────────────────────────────────────────────────────┤
# │ L1  @pto.jit         compile + cache + launch + top-level orchestration   │
# │                                                                            │
# │   flash_attention_kernel[grid, stream](...)                               │
# │   TensorView metadata / alloc_tile / partition_view / tload / tstore      │
# │   outer Q loop + inner KV loop + ping-pong state ownership                │
# │                                                                            │
# │   Key idea: one launchable entry owns both runtime binding and logical     │
# │   tile scheduling.                                                         │
# ├──────────────────────────────────────────────────────────────────────────┤
# │ L2  @pto.ukernel     Per-block execution sandwich                         │
# │                                                                            │
# │   explicit mte_load(part, tile) staging for current K/V block, mem_bar,   │
# │   call cube/simd/simt sub-kernels,                                        │
# │   manage scratch/state hand-off                                            │
# │                                                                            │
# │   Key idea: one place owns the "how this block runs on hardware" story.   │
# ├──────────────────────────────────────────────────────────────────────────┤
# │ L3a @pto.cube       Matrix-product kernels                                 │
# │                                                                            │
# │   qk_matmul: Q @ K^T                                                       │
# │   pv_matmul: P @ V                                                         │
# │   explicit LEFT/RIGHT/ACC scratch + UB output                              │
# │                                                                            │
# │   Key idea: UB tiles are inputs/outputs; cube-local state is explicit.    │
# ├──────────────────────────────────────────────────────────────────────────┤
# │ L3b @pto.simd       Row-wise vector math                                   │
# │                                                                            │
# │   online_softmax_rows                                                      │
# │   vreg stays local; persistent state is written back to UB tiles           │
# │                                                                            │
# │   Key idea: no cross-kernel vreg values, only UB-backed state.            │
# ├──────────────────────────────────────────────────────────────────────────┤
# │ L3c @pto.simt       Scalar metadata and pointwise blend                    │
# │                                                                            │
# │   materialize_tile_bounds / blend_output_rows                              │
# │                                                                            │
# │   Key idea: SIMT handles scalar control facts and scalar tile walks.      │
# └──────────────────────────────────────────────────────────────────────────┘
#
#                       dataflow for one KV block
#
#   jit kernel alloc/schedule
#          │
#          ▼
#   ukernel loads K/V block and sequences the pipeline
#          │
#          ├─ cube:  Q + K  ───────────────► S
#          ├─ simd:  S + (m_prev, l_prev) ─► P, (m_next, l_next), alpha, beta
#          ├─ cube:  P + V  ───────────────► PV
#          └─ simt:  (o_prev, PV, alpha, beta) ─► o_next
#
#   After each KV block:
#     (m_prev, l_prev, o_prev) := (m_next, l_next, o_next)
#
# The important part for the redesign is not the exact helper spelling, but
# that every cross-stage dependency is visible in the surface language.
