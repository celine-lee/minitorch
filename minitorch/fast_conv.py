
from typing import Tuple

import numpy as np
from numba import njit, prange

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import MAX_DIMS, Index, Shape, Strides
from .tensor_functions import Function


# Helper functions for use in the convolution kernels.
def _flat_to_index(i, shape, index):
    cur = i
    for idx in range(len(shape) - 1, -1, -1):
        index[idx] = int(cur % shape[idx])
        cur //= shape[idx]


def _index_to_pos(index, strides):
    pos = 0
    for ind, stride in zip(index, strides):
        pos += ind * stride
    return pos


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
        out_index: Index = np.zeros(MAX_DIMS, np.int16)
        _flat_to_index(i, out_shape, out_index)
        o = _index_to_pos(out_index, out_strides)
        b, oc, w = out_index[:3]
        for dw in range(kw):
            iw = w + dw
            if reverse:
                iw = w - dw
            if iw < 0 or iw >= width:
                continue
            for ic in range(in_channels):
                term1 = input[s1[0] * b + s1[1] * ic + s1[2] * iw]
                term2 = weight[s2[0] * oc + s2[1] * ic + s2[2] * dw]
                out[o] += term1 * term2


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
        out_index: Index = np.zeros(MAX_DIMS, np.int16)
        _flat_to_index(i, out_shape, out_index)
        o = _index_to_pos(out_index, out_strides)
        b, oc, h_pos, w = out_index[:4]
        acc = 0.0
        order = -1 if reverse else 1
        for dh in prange(kh):
            ih = h_pos + order * dh
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


tensor_conv1d = njit(parallel=True)(_tensor_conv1d)
tensor_conv2d = njit(parallel=True, fastmath=True)(_tensor_conv2d)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(*output.tuple(), output.size, *input.tuple(), *weight.tuple(), False)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(*grad_weight.tuple(), grad_weight.size, *new_input.tuple(), *new_grad_output.tuple(), False)
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(*grad_input.tuple(), grad_input.size, *grad_output.tuple(), *new_weight.tuple(), True)
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(*output.tuple(), output.size, *input.tuple(), *weight.tuple(), False)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(*grad_weight.tuple(), grad_weight.size, *new_input.tuple(), *new_grad_output.tuple(), False)
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(*grad_input.tuple(), grad_input.size, *grad_output.tuple(), *new_weight.tuple(), True)
        return grad_input, grad_weight

conv2d = Conv2dFun.apply
