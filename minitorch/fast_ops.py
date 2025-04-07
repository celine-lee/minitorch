
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from numba import njit, prange

from .tensor_data import MAX_DIMS
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional
    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

#------------------------------------------------------------------------------
# Helper functions for index conversion and broadcasting.
# These helpers are inlined so they can be used inside njitted functions.
#------------------------------------------------------------------------------

@njit(inline='always')
def _linear_to_multi_index(i, shape, out_index):
    # Given a linear index i and a shape array, fill out_index with the multi-index.
    cur = i
    for idx in range(len(shape) - 1, -1, -1):
        sh = shape[idx]
        out_index[idx] = cur % sh
        cur //= sh

@njit(inline='always')
def _multi_index_to_position(index, strides, n):
    # Given a multi-index (first n elements of index) and strides,
    # compute the linear position.
    pos = 0
    for j in range(n):
        pos += index[j] * strides[j]
    return pos

@njit(inline='always')
def _broadcast_index(out_index, src_shape, result_index):
    # Given an index for the output tensor (which is larger than the source),
    # compute the effective source index (broadcasting singleton dimensions as 0)
    offset = len(out_index) - len(src_shape)
    for i in range(len(src_shape)):
        if src_shape[i] > 1:
            result_index[i] = out_index[i + offset]
        else:
            result_index[i] = 0

@njit(inline='always')
def _broadcast_shapes(shape_a, shape_b):
    # Compute a broadcasted shape for two input shapes.
    m = max(len(shape_a), len(shape_b))
    c_rev = [0] * m
    a_rev = list(reversed(shape_a))
    b_rev = list(reversed(shape_b))
    for i in range(m):
        if i >= len(shape_a):
            c_rev[i] = b_rev[i]
        elif i >= len(shape_b):
            c_rev[i] = a_rev[i]
        else:
            c_rev[i] = max(a_rev[i], b_rev[i])
            if a_rev[i] != c_rev[i] and a_rev[i] != 1:
                raise RuntimeError("Broadcast failure " + str(shape_a) + " " + str(shape_b))
            if b_rev[i] != c_rev[i] and b_rev[i] != 1:
                raise RuntimeError("Broadcast failure " + str(shape_a) + " " + str(shape_b))
    return tuple(reversed(c_rev))


#------------------------------------------------------------------------------
# FastOps implementation
#------------------------------------------------------------------------------

class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        f = tensor_map(njit()(fn))
        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out
        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        f = tensor_zip(njit()(fn))
        def ret(a: Tensor, b: Tensor) -> Tensor:
            # If shapes differ, use the helper to compute a broadcasted shape.
            c_shape = a.shape if a.shape == b.shape else _broadcast_shapes(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out
        return ret

    @staticmethod
    def reduce(fn: Callable[[float, float], float], start: float = 0.0) -> Callable[[Tensor, int], Tensor]:
        f = tensor_reduce(njit()(fn))
        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start
            f(*out.tuple(), *a.tuple(), dim)
            return out
        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Ensure 3D multiplication (add dummy batch dimension if 2D)
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        # Compute batch broadcast shape using our helper.
        a_batch = a.shape[:-2]
        b_batch = b.shape[:-2]
        batch_shape = a_batch if a_batch == b_batch else _broadcast_shapes(a_batch, b_batch)
        ls = list(batch_shape)
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))
        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out

#------------------------------------------------------------------------------
# Low-level (NUMBA) implementations
#------------------------------------------------------------------------------

def tensor_map(fn: Callable[[float], float]) -> Callable[[Storage, np.ndarray, np.ndarray, Storage, np.ndarray, np.ndarray], None]:
    def _map(out: Storage, out_shape: np.ndarray, out_strides: np.ndarray,
             in_storage: Storage, in_shape: np.ndarray, in_strides: np.ndarray) -> None:
        # When shapes or strides differ, use broadcasted version.
        if (len(out_strides) != len(in_strides) or (out_strides != in_strides).any() or (out_shape != in_shape).any()):
            out_index = np.empty(MAX_DIMS, np.int32)
            in_index = np.empty(MAX_DIMS, np.int32)
            for i in prange(len(out)):
                _linear_to_multi_index(i, out_shape, out_index)
                _broadcast_index(out_index[:len(in_shape)], in_shape, in_index[:len(in_shape)])
                o = _multi_index_to_position(out_index, out_strides, len(out_shape))
                j = _multi_index_to_position(in_index, in_strides, len(in_shape))
                out[o] = fn(in_storage[j])
        else:
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
    return njit(parallel=True)(_map)

def tensor_zip(fn: Callable[[float, float], float]) -> Callable[
    [Storage, np.ndarray, np.ndarray,
     Storage, np.ndarray, np.ndarray,
     Storage, np.ndarray, np.ndarray], None]:
    def _zip(out: Storage, out_shape: np.ndarray, out_strides: np.ndarray,
             a_storage: Storage, a_shape: np.ndarray, a_strides: np.ndarray,
             b_storage: Storage, b_shape: np.ndarray, b_strides: np.ndarray) -> None:
        if (len(out_strides) != len(a_strides) or len(out_strides) != len(b_strides) or
            (out_strides != a_strides).any() or (out_strides != b_strides).any() or
            (out_shape != a_shape).any() or (out_shape != b_shape).any()):
            out_index = np.empty(MAX_DIMS, np.int32)
            a_index = np.empty(MAX_DIMS, np.int32)
            b_index = np.empty(MAX_DIMS, np.int32)
            for i in prange(len(out)):
                _linear_to_multi_index(i, out_shape, out_index)
                _broadcast_index(out_index[:len(a_shape)], a_shape, a_index[:len(a_shape)])
                _broadcast_index(out_index[:len(b_shape)], b_shape, b_index[:len(b_shape)])
                o = _multi_index_to_position(out_index, out_strides, len(out_shape))
                j = _multi_index_to_position(a_index, a_strides, len(a_shape))
                k = _multi_index_to_position(b_index, b_strides, len(b_shape))
                out[o] = fn(a_storage[j], b_storage[k])
        else:
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])
    return njit(parallel=True)(_zip)

def tensor_reduce(fn: Callable[[float, float], float]) -> Callable[
    [Storage, np.ndarray, np.ndarray, Storage, np.ndarray, np.ndarray, int], None]:
    def _reduce(out: Storage, out_shape: np.ndarray, out_strides: np.ndarray,
                a_storage: Storage, a_shape: np.ndarray, a_strides: np.ndarray, reduce_dim: int) -> None:
        for i in prange(len(out)):
            out_index = np.empty(MAX_DIMS, np.int32)
            _linear_to_multi_index(i, out_shape, out_index)
            o = _multi_index_to_position(out_index, out_strides, len(out_shape))
            accum = out[o]
            j = _multi_index_to_position(out_index, a_strides, len(out_shape))
            step = a_strides[reduce_dim]
            reduce_size = a_shape[reduce_dim]
            for s in range(reduce_size):
                accum = fn(accum, a_storage[j])
                j += step
            out[o] = accum
    return njit(parallel=True)(_reduce)

def _tensor_matrix_multiply(out: Storage, out_shape: np.ndarray, out_strides: np.ndarray,
                            a_storage: Storage, a_shape: np.ndarray, a_strides: np.ndarray,
                            b_storage: Storage, b_shape: np.ndarray, b_strides: np.ndarray) -> None:
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    for i1 in prange(out_shape[0]):
        for i2 in prange(out_shape[1]):
            for i3 in prange(out_shape[2]):
                a_inner = i1 * a_batch_stride + i2 * a_strides[1]
                b_inner = i1 * b_batch_stride + i3 * b_strides[2]
                acc = 0.0
                for _ in range(a_shape[2]):
                    acc += a_storage[a_inner] * b_storage[b_inner]
                    a_inner += a_strides[2]
                    b_inner += b_strides[1]
                o = i1 * out_strides[0] + i2 * out_strides[1] + i3 * out_strides[2]
                out[o] = acc
tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)
