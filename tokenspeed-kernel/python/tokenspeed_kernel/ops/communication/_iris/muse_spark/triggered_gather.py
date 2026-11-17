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

"""The first sorted rows of Muse's intermediate gather, on the SDMA copy engines.

The gate/up GEMM runs over consecutive sorted-row ranges. After each range
below the split, a mark releases that chunk of an Iris triggered plan, whose
copy engines write this rank's slice of those rows into every peer's gathered
rows while later ranges compute. The CU gather then moves the remaining valid
rows (``iris_pull_allgather_sorted_mxfp4_slices`` from the split), and the
down projection reads the gathered rows after a wait on the plan's arrival
groups.

Importing this module needs an Iris with triggered-plan destinations; the
heap must be created with ``engines=("sdma",)`` (rocm-xio).
"""

import torch
from tokenspeed_kernel._triton import gl, gluon, redirect_triton_to_tokenspeed_triton

with redirect_triton_to_tokenspeed_triton():
    from iris.experimental.plan import Chunk, Region, TriggeredPlan
    from iris.experimental.plan.gluon import TriggeredView

# gfx950 dispatches workgroup i to XCD i mod 8. The copy engines read behind
# every XCD's L2, so a chunk is released once from each XCD, and the wait
# acquires on each XCD before the down projection reads.
_XCDS = 8
# Polls of a group's arrival before the wait records a timeout and moves on.
_POLL_LIMIT = 1 << 24
# error_word code of a chunk that never arrived
_CHUNK_TIMED_OUT = gl.constexpr(1)


@gluon.jit
def iris_triggered_gather_mark(handle, chunk):
    """Release plan chunk ``chunk``: one workgroup per XCD."""
    TriggeredView.initialize(handle).mark_ready(chunk)


@gluon.jit
def iris_triggered_gather_wait(handle, error_word, groups, poll_limit):
    """Wait for every group of this round on each XCD, or record a timeout."""
    cv = TriggeredView.initialize(handle)
    for group in range(groups):
        polls = 0
        while (cv.group_arrived(group) == 0) & (polls < poll_limit):
            polls += 1
        if polls >= poll_limit:
            gl.atomic_max(error_word, _CHUNK_TIMED_OUT, sem="relaxed")


class TriggeredSortedSliceGather:
    """Sends each rank's slice of the first sorted rows to every peer on SDMA.

    ``slices`` is this rank's symmetric uint8 arena: ``capacity`` sorted rows
    of ``slice_bytes``, then their scale panels of ``panel_bytes`` per 32
    rows. The plan writes them into every peer's gathered arena
    (``gathered``: rows of ``world_size * slice_bytes`` and panels of
    ``world_size * panel_bytes``, rank r's part at ``r * slice_bytes`` and
    ``r * panel_bytes``), one chunk of ``chunk_rows`` rows per range below
    ``split_rows``; every peer's chunk ``c`` arrives on group ``c``.

    Each use is one round: ``start``, the gate/up ranges with ``mark`` after
    each chunk's range, the caller's CU gather of the rows from the split and
    this rank's own slice, ``wait`` before the down projection, then
    ``finish``. Every rank runs the same rounds, since a round ends with the
    plan's collective fence. Plans sharing the copy queues must start their
    rounds in the order they were prepared, so a caller running another plan
    (the attention reduce-scatter) alternates rounds with it exactly.

    Args:
        ctx: Iris heap, created with ``engines=("sdma",)``.
        slices: This rank's symmetric uint8 slice arena.
        capacity: Sorted rows the arena holds.
        slice_bytes: Bytes of this rank's slice of a row.
        panel_bytes: Bytes of this rank's slice of a 32-row scale panel.
        split_rows: Rows the copy engines send, a multiple of ``chunk_rows``.
        chunk_rows: Rows per gate/up range and plan chunk, a multiple of 128.
        timeout: Seconds the plan's waits and round fence wait before
            recording an error.
    """

    def __init__(
        self,
        ctx,
        slices: torch.Tensor,
        *,
        capacity: int,
        slice_bytes: int,
        panel_bytes: int,
        split_rows: int,
        chunk_rows: int,
        timeout: float,
    ):
        if (
            chunk_rows % 128
            or split_rows % chunk_rows
            or not 0 < split_rows <= capacity
        ):
            raise ValueError(
                f"split {split_rows} must be whole {chunk_rows}-row chunks (a "
                f"multiple of 128) within {capacity} rows"
            )
        self.rank = ctx.get_rank()
        self.world_size = world = ctx.get_num_ranks()
        self.capacity = capacity
        self.split_rows = split_rows
        self.chunk_rows = chunk_rows
        panels = -(-capacity // 32)
        data_bytes = capacity * slice_bytes
        if (
            slices.dtype != torch.uint8
            or slices.numel() < data_bytes + panels * panel_bytes
        ):
            raise ValueError("the slice arena is smaller than its rows and panels")
        self.gathered = ctx.zeros(
            world * (data_bytes + panels * panel_bytes), dtype=torch.uint8
        )
        # The gathered rows and scale panels in the layout gate/up writes for
        # the full intermediate.
        self.gathered_rows = self.gathered[: world * data_bytes].view(
            capacity, world * slice_bytes
        )
        self.gathered_scales = self.gathered[world * data_bytes :].view(
            panels * 32, world * panel_bytes // 32
        )
        schedule = []
        for chunk, start in enumerate(range(0, split_rows, chunk_rows)):
            rows = Region(
                offset_bytes=start * slice_bytes,
                row_bytes=slice_bytes,
                rows=chunk_rows,
                dst_offset_bytes=start * world * slice_bytes + self.rank * slice_bytes,
                dst_stride_bytes=world * slice_bytes,
            )
            scales = Region(
                offset_bytes=data_bytes + start // 32 * panel_bytes,
                row_bytes=panel_bytes,
                rows=chunk_rows // 32,
                dst_offset_bytes=world * data_bytes
                + (start // 32 * world + self.rank) * panel_bytes,
                dst_stride_bytes=world * panel_bytes,
            )
            schedule.append(Chunk((rows, scales), writers=_XCDS, group=chunk))
        self.chunks = len(schedule)
        self.plan = TriggeredPlan(
            ctx,
            slices[: data_bytes + panels * panel_bytes],
            schedule,
            timeout=timeout,
            destination=self.gathered,
        )
        self._first_chunk = self.plan.chunk_id(self.rank, 0)
        self.error_word = torch.zeros(1, dtype=torch.int32, device=slices.device)
        # The latest error state copied off the GPU without waiting for it.
        self._errors = torch.zeros(1 + 4, dtype=torch.int64, pin_memory=True)
        self._errors_copied = torch.cuda.Event()
        self._errors_pending = False
        self._handle: torch.Tensor | None = None

    def ranges(self, sorted_rows: int) -> list[tuple[int, int, int | None]]:
        """Gate/up ranges ``(start, end, chunk)`` over ``sorted_rows`` sorted rows.

        The chunk ranges come first, ``chunk`` being the plan chunk to mark
        after the range; the rest is one range with ``chunk`` None.
        """
        if sorted_rows < self.split_rows:
            raise ValueError(
                f"{sorted_rows} sorted rows do not cover the {self.split_rows}-row split"
            )
        spans = [
            (start, start + self.chunk_rows, chunk)
            for chunk, start in enumerate(range(0, self.split_rows, self.chunk_rows))
        ]
        if sorted_rows > self.split_rows:
            spans.append((self.split_rows, sorted_rows, None))
        return spans

    def start(self) -> None:
        """Start a round; raise an error a finished earlier round recorded."""
        self.raise_recorded_errors()
        self._handle = self.plan.start()

    def mark(self, chunk: int) -> None:
        """Release ``chunk`` once the gate/up range before it on the stream is done."""
        iris_triggered_gather_mark[(_XCDS,)](
            self._handle, self._first_chunk + chunk, num_warps=1
        )

    def wait(self) -> None:
        """Make the stream wait until every peer's chunks have arrived here."""
        iris_triggered_gather_wait[(_XCDS,)](
            self._handle, self.error_word, self.chunks, _POLL_LIMIT, num_warps=1
        )

    def finish(self) -> None:
        """End the round after the gathered rows' reader is queued.

        Queued, so later writers of the slice arena wait until peers hold this
        rank's chunks, and the fence frees the gathered arena for the next
        round and prepares it.
        """
        self.plan.wait(blocking=False)
        self.plan.next_round()
        self._handle = None
        if not self._errors_pending:
            # The plan's first recorded error, [code, round, index, value],
            # is private to Iris; no public accessor exists yet.
            state = torch.cat((self.error_word.to(torch.int64), self.plan._error))
            self._errors.copy_(state, non_blocking=True)
            self._errors_copied.record()
            self._errors_pending = True

    def raise_recorded_errors(self) -> None:
        """Raise if a round that has finished recorded an error. Never waits."""
        if not self._errors_pending or not self._errors_copied.query():
            return
        self._errors_pending = False
        timed_out, code, round_, index, value = self._errors.tolist()
        if timed_out:
            raise RuntimeError("triggered gather: a chunk never arrived")
        if code:
            raise RuntimeError(
                f"triggered gather plan error {code} in round {round_} "
                f"(chunk or peer {index}, value {value}); close() raises its detail"
            )

    def close(self) -> None:
        """Collective: close the plan, raising any error it recorded."""
        self.plan.close()
