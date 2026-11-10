# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Projection + reduce-scatter whose transfers run on the SDMA copy engines.

The projection GEMM writes each owner's rows into a fixed, pitched place in
this rank's producer buffer and marks every strip of 256 rows as it finishes.
An Iris triggered plan (``iris.experimental.plan``) has the copy engines send
each marked strip into its owner's inbox while the GEMM keeps computing, so
the owner sums local memory only. The sum order matches
``iris_pull_reduce_scatter``, so the shard carries the same bits for the same
partials.

Importing this module needs an Iris with triggered-plan destinations and
peers; the heap must be created with ``engines=("sdma",)`` (rocm-xio).
"""

import torch
from tokenspeed_kernel._triton import (
    gl,
    gluon,
    redirect_triton_to_tokenspeed_triton,
    triton,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.largem import (
    LARGEM_BLOCK_M,
    launch_gluon_mm_a16w16_prefill_mark_owner_rows_gfx950,
)

with redirect_triton_to_tokenspeed_triton():
    from iris.experimental.plan import Chunk, Region, TriggeredPlan
    from iris.experimental.plan.gluon import TriggeredView

STRIP_ROWS = LARGEM_BLOCK_M
REDUCE_PROGRAMS = 256
REDUCE_NUM_WARPS = 4
REDUCE_BLOCK = 8 * 64 * REDUCE_NUM_WARPS
# Polls of a strip's arrival before the kernel records a timeout and moves on;
# each poll is a system-scope atomic, so this is seconds, far past any copy.
REDUCE_POLL_LIMIT = 1 << 24
# error_word code of a strip that never arrived
STRIP_TIMED_OUT = gl.constexpr(1)


@gluon.jit(do_not_specialize=["ROWS", "OWNED_ROWS"])
def iris_triggered_reduce_scatter_rows(
    partials,
    inbox,
    output,
    handle,
    error_word,
    ROWS,
    OWNED_ROWS,
    POLL_LIMIT,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    OWNER_PITCH: gl.constexpr,
    WIDTH: gl.constexpr,
    STRIP: gl.constexpr,
    BLOCK: gl.constexpr,
    PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Sum this rank's owned rows from its own partials and its inbox.

    ``partials`` is ``[WORLD_SIZE * OWNER_PITCH, WIDTH]``: this rank's rows sit
    at ``RANK * OWNER_PITCH``. ``inbox[(q - RANK - 1) % WORLD_SIZE]`` holds rank
    ``q``'s partials of them, delivered strip by strip by ``handle``'s plan.
    Each strip waits for its chunks from every peer, then the rows are summed
    in FP32 in rank order 0..WORLD_SIZE-1 and rounded to BF16, as
    ``iris_pull_reduce_scatter`` does; rows past ``ROWS`` are written as zero.
    A strip that never arrives sets ``error_word`` instead of hanging.
    """
    STRIPS: gl.constexpr = OWNER_PITCH // STRIP
    CHUNKS: gl.constexpr = STRIPS * (WORLD_SIZE - 1)
    cv = TriggeredView.initialize(handle)
    pid = gl.program_id(0)
    elements = OWNED_ROWS * WIDTH
    valid = gl.minimum(gl.maximum(ROWS - RANK * OWNED_ROWS, 0), OWNED_ROWS) * WIDTH
    slice_elements: gl.constexpr = OWNER_PITCH * WIDTH
    own = partials + RANK * slice_elements
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    lanes = gl.arange(0, BLOCK, layout=layout)
    waited = 0
    for tile in range(pid, gl.cdiv(elements, BLOCK), PROGRAMS):
        # Tiles of a program only move forward, so each strip is waited once.
        need = (tile * BLOCK + BLOCK - 1) // (STRIP * WIDTH) + 1
        while waited < gl.minimum(need, STRIPS):
            for peer in gl.static_range(WORLD_SIZE):
                if peer != RANK:
                    # Peer's chunk for this strip toward this rank
                    chunk = (
                        peer * CHUNKS
                        + waited * (WORLD_SIZE - 1)
                        + (RANK - peer - 1 + WORLD_SIZE) % WORLD_SIZE
                    )
                    polls = 0
                    while (cv.arrived(chunk) == 0) & (polls < POLL_LIMIT):
                        polls += 1
                    if polls >= POLL_LIMIT:
                        gl.atomic_max(error_word, STRIP_TIMED_OUT, sem="relaxed")
            waited += 1
        offsets = tile * BLOCK + lanes
        mask = offsets < valid
        values = ()
        for q in gl.static_range(WORLD_SIZE):
            if q == RANK:
                source = own
            else:
                source = (
                    inbox + (q - RANK - 1 + WORLD_SIZE) % WORLD_SIZE * slice_elements
                )
            values += (
                gl.amd.cdna4.buffer_load(source, offsets, mask, 0, cache=".cg").to(
                    gl.float32
                ),
            )
        reduced = values[0]
        for q in gl.static_range(1, WORLD_SIZE):
            reduced += values[q]
        gl.amd.cdna4.buffer_store(
            reduced.to(gl.bfloat16), output, offsets, offsets < elements, cache=".wt"
        )


class TriggeredOwnedRowsReduceScatter:
    """``out = A @ B.T`` reduce-scattered over consecutive owner rows, on SDMA.

    Owns one triggered plan from this rank's producer buffer into every peer's
    inbox, used for one round per call. Every rank calls ``run`` the same
    number of times with the same row count, since each call ends with the
    plan's collective round fence. The plan has one row window, the row counts
    whose owned rows fill exactly ``owner_pitch // 256`` strips (``serves``):
    region sizes are fixed when the plan is created, and a plan's rounds must
    run in the order they are prepared. Other row counts stay with the caller's
    CU path.

    Args:
        ctx: Iris heap, created with ``engines=("sdma",)``.
        producer: This rank's symmetric BF16 producer buffer; its first
            ``world_size * owner_pitch * width`` elements hold the partials.
        owner_pitch: Rows reserved per owner, a multiple of 256.
        width: Output width.
        min_rows: Fewest rows the plan serves.
        timeout: Seconds the plan's waits and round fence wait before recording
            an error.
    """

    def __init__(
        self,
        ctx,
        producer: torch.Tensor,
        *,
        owner_pitch: int,
        width: int,
        min_rows: int,
        timeout: float,
    ):
        if owner_pitch % STRIP_ROWS:
            raise ValueError(
                f"owner_pitch {owner_pitch} is not a multiple of {STRIP_ROWS}"
            )
        self.rank = ctx.get_rank()
        self.world_size = ctx.get_num_ranks()
        self.owner_pitch = owner_pitch
        self.width = width
        self.strips = owner_pitch // STRIP_ROWS
        world, strips = self.world_size, self.strips
        numel = world * owner_pitch * width
        if producer.dtype != torch.bfloat16 or producer.numel() < numel:
            raise ValueError(
                f"producer needs {numel} BF16 elements, has {producer.numel()} {producer.dtype}"
            )
        self.partials = producer.view(-1)[:numel].view(world * owner_pitch, width)
        self.inbox = ctx.zeros((world - 1, owner_pitch, width), dtype=torch.bfloat16)
        chunks = strips * (world - 1)
        device = producer.device
        self.strip_arrivals = torch.zeros(chunks, dtype=torch.int32, device=device)
        self.error_word = torch.zeros(1, dtype=torch.int32, device=device)
        strip_bytes = STRIP_ROWS * width * producer.element_size()
        slice_bytes = owner_pitch * width * producer.element_size()
        schedule = []
        for strip in range(strips):
            for step in range(world - 1):
                owner = (self.rank + 1 + step) % world
                slot = (self.rank - owner - 1) % world
                region = Region(
                    offset_bytes=owner * slice_bytes + strip * strip_bytes,
                    row_bytes=strip_bytes,
                    dst_offset_bytes=slot * slice_bytes + strip * strip_bytes,
                )
                schedule.append(Chunk((region,), writers=1, peers=(owner,)))
        self.plan = TriggeredPlan(
            ctx, self.partials, schedule, timeout=timeout, destination=self.inbox
        )
        first = self.plan.chunk_id(self.rank, 0)
        self.counters = self.plan.counters[first : first + chunks]
        self.min_rows = max(min_rows, world * STRIP_ROWS * (strips - 1) + 1)
        self.max_rows = world * owner_pitch
        # The latest error state copied off the GPU without waiting for it.
        self._errors = torch.zeros(1 + 4, dtype=torch.int64, pin_memory=True)
        self._errors_copied = torch.cuda.Event()
        self._errors_pending = False

    def serves(self, rows: int) -> bool:
        """Whether ``rows`` fit the plan: every owner's rows fill all its strips."""
        return (
            self.min_rows <= rows <= self.max_rows
            and triton.cdiv(triton.cdiv(rows, self.world_size), STRIP_ROWS)
            == self.strips
        )

    def run(
        self, x: torch.Tensor, weight: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        """Reduce-scatter ``x @ weight.T`` into ``output`` (``[ceil(rows / world), width]``).

        Collective and stream-ordered: queues the plan's round on the current
        stream, with no host synchronization. Raises an error the GPU recorded
        in an earlier round, once that round's state has come back.
        """
        self.raise_recorded_errors()
        rows = x.shape[0]
        owned_rows = triton.cdiv(rows, self.world_size)
        if not self.serves(rows) or tuple(output.shape) != (owned_rows, self.width):
            raise ValueError(
                f"{rows} rows into {tuple(output.shape)} do not fit the plan"
            )
        handle = self.plan.start()
        if (
            launch_gluon_mm_a16w16_prefill_mark_owner_rows_gfx950(
                x,
                weight,
                self.partials,
                self.counters,
                self.strip_arrivals,
                owned_rows=owned_rows,
                owner_pitch=self.owner_pitch,
                rank=self.rank,
                world_size=self.world_size,
            )
            is None
        ):
            # Every rank sees the same operands, so every rank raises here.
            raise ValueError("operands unsupported by the strip-marking GEMM")
        iris_triggered_reduce_scatter_rows[(REDUCE_PROGRAMS,)](
            self.partials,
            self.inbox,
            output,
            handle,
            self.error_word,
            rows,
            owned_rows,
            REDUCE_POLL_LIMIT,
            RANK=self.rank,
            WORLD_SIZE=self.world_size,
            OWNER_PITCH=self.owner_pitch,
            WIDTH=self.width,
            STRIP=STRIP_ROWS,
            BLOCK=REDUCE_BLOCK,
            PROGRAMS=REDUCE_PROGRAMS,
            NUM_WARPS=REDUCE_NUM_WARPS,
            num_warps=REDUCE_NUM_WARPS,
        )
        # Queued: later writers of the producer buffer wait until peers hold
        # this rank's strips, and the fence frees the inbox for the next round.
        self.plan.wait(blocking=False)
        self.plan.next_round()
        self._copy_errors()
        return output

    def _copy_errors(self) -> None:
        if self._errors_pending:
            return
        # The plan's first recorded error, [code, round, index, value], is
        # private to Iris; no public accessor exists yet.
        state = torch.cat((self.error_word.to(torch.int64), self.plan._error))
        self._errors.copy_(state, non_blocking=True)
        self._errors_copied.record()
        self._errors_pending = True

    def raise_recorded_errors(self) -> None:
        """Raise if a round that has finished recorded an error. Never waits."""
        if not self._errors_pending or not self._errors_copied.query():
            return
        self._errors_pending = False
        strip, code, round_, index, value = self._errors.tolist()
        if strip:
            raise RuntimeError("triggered reduce-scatter: a strip never arrived")
        if code:
            raise RuntimeError(
                f"triggered reduce-scatter plan error {code} in round {round_} "
                f"(chunk or peer {index}, value {value}); close() raises its detail"
            )

    def warmup(self, x: torch.Tensor, weight: torch.Tensor) -> None:
        """Run one round at the largest row count and check it, before serving."""
        output = torch.empty(
            (self.owner_pitch, self.width), dtype=torch.bfloat16, device=x.device
        )
        self.run(x[: self.max_rows], weight, output)
        # The round's error state was copied after it; it has landed now.
        torch.cuda.synchronize()
        self.raise_recorded_errors()

    def close(self) -> None:
        """Collective: close the plan, raising any error it recorded."""
        self.plan.close()
