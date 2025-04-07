
from typing import Tuple

import numpy as np
from numba import njit, prange

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import MAX_DIMS, Index, Shape, Strides
from .tensor_functions import Function

#########################################
# Helper Functions for Conv Kernels (CPU)
#########################################
@njit(inline="always")
def _compute_multi_index_conv(linear_idx, shape, index_out):
    for idx in range(int(len(shape)) - 1, -1, -1):
        index_out[idx] = int(linear_idx % shape[idx])
        linear_idx //= shape[idx]


@njit(inline="always")
def _compute_offset_conv(index, strides, n):
    off = 0
    for i in range(n):
        off += index[i] * strides[i]
    return off


#########################################
# 1D Convolution Kernel
#########################################
def _tensor_conv1d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    for i in prange(out_size):
        out_index = np.zeros(MAX_DIMS, np.int16)
        _compute_multi_index_conv(i, out_shape, out_index)
        o = _compute_offset_conv(out_index, out_strides, len(out_shape))
        b, oc, w = out_index[0], out_index[1], out_index[2]

        for dw in range(kw):
            iw = w + dw if not reverse else w - dw
            if iw < 0 or iw >= width:
                continue

            for ic in range(in_channels):
                term1 = input[s1[0] * b + s1[1] * ic + s1[2] * iw]
                term2 = weight[s2[0] * oc + s2[1] * ic + s2[2] * dw]
                out[o] += term1 * term2


tensor_conv1d = njit(parallel=True)(_tensor_conv1d)


#########################################
# 1D Convolution Function (with autodiff)
#########################################
class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


#########################################
# 2D Convolution Kernel
#########################################
@njit(parallel=True, fastmath=True)
def _tensor_conv2d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    for i in prange(out_size):
        out_index = np.zeros(MAX_DIMS, np.int16)
        _compute_multi_index_conv(i, out_shape, out_index)
        o = _compute_offset_conv(out_index, out_strides, len(out_shape))
        b, oc, h, w = out_index[0], out_index[1], out_index[2], out_index[3]
        acc = 0.0
        order = -1 if reverse else 1
        for dh in prange(kh):
            ih = h + order * dh
            if ih < 0 or ih >= height:
                continue
            for dw in prange(kw):
                iw = w + order * dw
                if iw < 0 or iw >= width:
                    continue
                inner1 = s10 * b + s12 * ih + s13 * iw
                inner2 = s20 * oc + s22 * dh + s23 * dw
                for ic in prange(in_channels):
                    acc += input[inner1] * weight[inner2]
                    inner1 += s11
                    inner2 += s21
        out[o] = acc


tensor_conv2d = njit(parallel=True, fastmath=True)(_tensor_conv2d)


#########################################
# 2D Convolution Function (with autodiff)
#########################################
class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
