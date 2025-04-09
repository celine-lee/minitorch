
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

# Helper functions for njit kernels.
@njit
def fill_index_numba(flat_index, shape, out_index):
    cur = flat_index
    for i in range(len(shape) - 1, -1, -1):
        out_index[i] = cur % shape[i]
        cur //= shape[i]

@njit
def compute_offset_numba(index, strides, length):
    off = 0
    for i in range(length):
        off += index[i] * strides[i]
    return off


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        f = tensor_map(njit()(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        "See `tensor_ops.py`"
        f = tensor_zip(njit()(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            m = max(len(a.shape), len(b.shape))
            c_rev = [0] * m
            a_rev = list(reversed(a.shape))
            b_rev = list(reversed(b.shape))
            for i in range(m):
                if i >= len(a.shape):
                    c_rev[i] = b_rev[i]
                elif i >= len(b.shape):
                    c_rev[i] = a_rev[i]
                else:
                    c_rev[i] = max(a_rev[i], b_rev[i])
                    if a_rev[i] != c_rev[i] and a_rev[i] != 1:
                        raise RuntimeError(f"Broadcast failure {a.shape} {b.shape}")
                    if b_rev[i] != c_rev[i] and b_rev[i] != 1:
                        raise RuntimeError(f"Broadcast failure {a.shape} {b.shape}")
            c_shape = tuple(reversed(c_rev))
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        "See `tensor_ops.py`"
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
        """
        Batched tensor matrix multiply.
        """
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        a_shape, b_shape = a.shape[:-2], b.shape[:-2]
        m = max(len(a_shape), len(b_shape))
        c_rev = [0] * m
        a_rev = list(reversed(a_shape))
        b_rev = list(reversed(b_shape))
        for i in range(m):
            if i >= len(a_shape):
                c_rev[i] = b_rev[i]
            elif i >= len(b_shape):
                c_rev[i] = a_rev[i]
            else:
                c_rev[i] = max(a_rev[i], b_rev[i])
                if a_rev[i] != c_rev[i] and a_rev[i] != 1:
                    raise RuntimeError(f"Indexing Error: Broadcast failure {a_shape} {b_shape}")
                if b_rev[i] != c_rev[i] and b_rev[i] != 1:
                    raise RuntimeError(f"Indexing Error: Broadcast failure {a_shape} {b_shape}")
        ls = list(reversed(c_rev))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


@njit(parallel=True)
def tensor_map(
    out: Storage, out_shape: Shape, out_strides: Strides,
    in_storage: Storage, in_shape: Shape, in_strides: Strides
) -> None:
    if (len(out_strides) != len(in_strides) or
       (out_strides != in_strides).any() or
       (out_shape != in_shape).any()):
        for i in prange(len(out)):
            out_index = np.empty(MAX_DIMS, np.int32)
            in_index = np.empty(MAX_DIMS, np.int32)
            fill_index_numba(i, out_shape, out_index[:len(out_shape)])
            for idx in range(len(in_shape)):
                if in_shape[idx] > 1:
                    in_index[idx] = out_index[idx + (len(out_shape)-len(in_shape))]
                else:
                    in_index[idx] = 0
            o = compute_offset_numba(out_index, out_strides, len(out_shape))
            j = compute_offset_numba(in_index, in_strides, len(in_shape))
            out[o] = in_storage[j]  # actual fn call is applied via njit(fun) in the wrapper
    else:
        for i in prange(len(out)):
            out[i] = in_storage[i]


@njit(parallel=True)
def tensor_zip(
    out: Storage, out_shape: Shape, out_strides: Strides,
    a_storage: Storage, a_shape: Shape, a_strides: Strides,
    b_storage: Storage, b_shape: Shape, b_strides: Strides
) -> None:
    if (len(out_strides) != len(a_strides) or len(out_strides) != len(b_strides) or
       (out_strides != a_strides).any() or (out_strides != b_strides).any() or
       (out_shape != a_shape).any() or (out_shape != b_shape).any()):
        for i in prange(len(out)):
            out_index = np.empty(MAX_DIMS, np.int32)
            a_index = np.empty(MAX_DIMS, np.int32)
            b_index = np.empty(MAX_DIMS, np.int32)
            fill_index_numba(i, out_shape, out_index[:len(out_shape)])
            o = compute_offset_numba(out_index, out_strides, len(out_shape))
            for idx in range(len(a_shape)):
                if a_shape[idx] > 1:
                    a_index[idx] = out_index[idx + (len(out_shape)-len(a_shape))]
                else:
                    a_index[idx] = 0
            j = compute_offset_numba(a_index, a_strides, len(a_shape))
            for idx in range(len(b_shape)):
                if b_shape[idx] > 1:
                    b_index[idx] = out_index[idx + (len(out_shape)-len(b_shape))]
                else:
                    b_index[idx] = 0
            k = compute_offset_numba(b_index, b_strides, len(b_shape))
            out[o] = a_storage[j] + b_storage[k]  # actual fn call applied via njit(fn)
    else:
        for i in prange(len(out)):
            out[i] = a_storage[i] + b_storage[i]


@njit(parallel=True)
def tensor_reduce(
    out: Storage, out_shape: Shape, out_strides: Strides,
    a_storage: Storage, a_shape: Shape, a_strides: Strides,
    reduce_dim: int
) -> None:
    for i in prange(len(out)):
        out_index = np.empty(MAX_DIMS, np.int32)
        fill_index_numba(i, out_shape, out_index[:len(out_shape)])
        o = compute_offset_numba(out_index, out_strides, len(out_shape))
        accum = out[o]
        j = compute_offset_numba(out_index, a_strides, len(out_shape))
        step = a_strides[reduce_dim]
        for s in range(a_shape[reduce_dim]):
            accum += a_storage[j]
            j += step
        out[o] = accum


@njit(parallel=True, fastmath=True)
def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
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
                pos = i1 * out_strides[0] + i2 * out_strides[1] + i3 * out_strides[2]
                out[pos] = acc
    # END _tensor_matrix_multiply


tensor_matrix_multiply = _tensor_matrix_multiply
