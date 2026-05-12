import math
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from quack.cute_dsl_utils import ParamsBase
from quack.layout_utils import reshape_acc_to_frgA, reshape_acc_to_mn

LOG2_E = math.log2(math.e)


@cute.jit
def thread_reduce_max(x: cute.TensorSSA, init_val: float | Float32 | None = None):
    """
    Max of thread's register slice.
    Original FA2 in Cute3.0 simply just max'd every element individually
    Can issue 4 at a time to improve ILP (~4 cycles/FLOP)
    """
    x_frag = cute.make_fragment(x.shape, Float32)
    x_frag.store(x)

    # thread max
    t_max = [x_frag[0], x_frag[1], x_frag[2], x_frag[3]]
    for i in cutlass.range(4, cute.size(x), 4):
        t_max[0] = cute.arch.fmax(t_max[0], x_frag[i])
        t_max[1] = cute.arch.fmax(t_max[1], x_frag[i + 1])
        t_max[2] = cute.arch.fmax(t_max[2], x_frag[i + 2])
        t_max[3] = cute.arch.fmax(t_max[3], x_frag[i + 3])

    t_max[0] = cute.arch.fmax(t_max[0], t_max[2])
    t_max[1] = cute.arch.fmax(t_max[1], t_max[3])
    t_max[0] = cute.arch.fmax(t_max[0], t_max[1])

    if cutlass.const_expr(init_val is not None):
        t_max[0] = cute.arch.fmax(t_max[0], init_val)

    return t_max[0]


def thread_reduce_max_dsl(x: cute.TensorSSA, init_val: Float32 | float | None = None):
    """
    Probably natively CuTe might only do 2-way ILP compared to above
    """
    if cutlass.const_expr(init_val is None):
        # check if this works
        init_val = -Float32.inf
    return x.reduce(cute.ReductionOp.MAX, init_val, 0)


def thread_reduce_sum(x: cute.TensorSSA, init_val: float | Float32 | None = None):
    if cutlass.const_expr(init_val is None):
        init_val = Float32.zero
    return x.reduce(cute.ReductionOp.ADD, init_val, 0)


def reduce_max(x: cute.TensorSSA, init_val: float | Float32 | None = None):
    thread_max = thread_reduce_max(x, init_val)
    # warp reduce
    return cute.arch.warp_reduction_max(thread_max, threads_in_group=4)


@dataclass
class Softmax(ParamsBase):
    scale_log2: Float32
    row_max: cute.Tensor
    row_sum: cute.Tensor

    @staticmethod
    def create(num_rows: cutlass.Constexpr[int], scale_log2):
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return Softmax(scale_log2, row_max, row_sum)

    @cute.jit
    def softmax_rescale_o(
        self,
        acc_S: cute.Tensor,
        acc_O: cute.Tensor,
        is_first: cutlass.Constexpr[bool] = False,
    ):
        scores = reshape_acc_to_mn(acc_S)
        output = reshape_acc_to_mn(acc_O)

        row_max = self.row_max
        row_sum = self.row_sum
        scale_log2 = self.scale_log2

        for r in cutlass.range(cute.size(row_max), unroll=True):
            scores_row = scores[r, None].load()
            output_row = output[r, None].load()

            r_max_new = reduce_max(
                scores_row,
                init_val=row_max[r] if cutlass.const_expr(not is_first) else None,
            )
            r_max_old = row_max[r]
            row_max[r] = r_max_new

            if cutlass.const_expr(is_first):
                r_max_new_scaled = scale_log2 * r_max_new
                # can give in full SSA row
                scores_row_exp = cute.math.exp2(
                    scores_row * scale_log2 - r_max_new_scaled, fastmath=True
                )
                r_sum = thread_reduce_sum(scores_row_exp)
            else:
                r_max_new_scaled = scale_log2 * r_max_new
                correction = cute.math.exp2(
                    scale_log2 * (r_max_old - r_max_new), fastmath=True
                )
                scores_row_exp = cute.math.exp2(
                    scores_row * scale_log2 - r_max_new_scaled, fastmath=True
                )
                r_sum = thread_reduce_sum(
                    scores_row_exp, init_val=correction * row_sum[r]
                )
                output_row *= correction
                output[r, None].store(output_row)

            row_sum[r] = r_sum
            scores[r, None].store(scores_row_exp)
