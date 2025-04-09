
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


# Helper functions for index computations.
def _flat_to_index(i: int, shape: np.ndarray, index: np.ndarray) -> None:
    cur = i
    for idx in range(len(shape) - 1, -1, -1):
        index[idx] = int(cur % shape[idx])
        cur //= shape[idx]


def _multi_index_to_pos(index: np.ndarray, strides: np.ndarray) -> int:
    pos = 0
    for ind, stride in zip(index, strides):
        pos += ind * stride
    return pos


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


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # When shapes and strides differ, use multi-index routing.
        if (len(out_strides) != len(in_strides)
            or (out_strides != in_strides).any()
            or (out_shape != in_shape).any()):
            for i in prange(len(out)):
                out_index: np.ndarray = np.empty(MAX_DIMS, np.int32)
                in_index: np.ndarray = np.empty(MAX_DIMS, np.int32)
                _flat_to_index(i, out_shape, out_index)
                for idx, s in enumerate(in_shape):
                    in_index[idx] = out_index[idx + (len(out_shape) - len(in_shape))] if s > 1 else 0
                o = _multi_index_to_pos(out_index, out_strides)
                j = _multi_index_to_pos(in_index, in_strides)
                out[o] = fn(in_storage[j])
        else:
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
    return njit(parallel=True)(_map)


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    def _zip(
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
        if (len(out_strides) != len(a_strides)
            or len(out_strides) != len(b_strides)
            or (out_strides != a_strides).any()
            or (out_strides != b_strides).any()
            or (out_shape != a_shape).any()
            or (out_shape != b_shape).any()):
            for i in prange(len(out)):
                out_index: np.ndarray = np.empty(MAX_DIMS, np.int32)
                a_index: np.ndarray = np.empty(MAX_DIMS, np.int32)
                b_index: np.ndarray = np.empty(MAX_DIMS, np.int32)
                _flat_to_index(i, out_shape, out_index)
                o = _multi_index_to_pos(out_index, out_strides)
                for idx, s in enumerate(a_shape):
                    a_index[idx] = out_index[idx + (len(out_shape) - len(a_shape))] if s > 1 else 0
                j = _multi_index_to_pos(a_index, a_strides)
                for idx, s in enumerate(b_shape):
                    b_index[idx] = out_index[idx + (len(out_shape) - len(b_shape))] if s > 1 else 0
                k = _multi_index_to_pos(b_index, b_strides)
                out[o] = fn(a_storage[j], b_storage[k])
        else:
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])
    return njit(parallel=True)(_zip)


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        for i in prange(len(out)):
            out_index: np.ndarray = np.empty(MAX_DIMS, np.int32)
            _flat_to_index(i, out_shape, out_index)
            o = _multi_index_to_pos(out_index, out_strides)
            accum = out[o]
            j = _multi_index_to_pos(out_index, a_strides)
            step = a_strides[reduce_dim]
            reduce_size = a_shape[reduce_dim]
            for s in range(reduce_size):
                accum = fn(accum, a_storage[j])
                j += step
            out[o] = accum
    return njit(parallel=True)(_reduce)


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
                out_position = i1 * out_strides[0] + i2 * out_strides[1] + i3 * out_strides[2]
                out[out_position] = acc
    # END ASSIGN3.2

tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)
