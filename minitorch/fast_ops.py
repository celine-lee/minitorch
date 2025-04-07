
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


# Helper functions for index calculations.
@njit(inline="always")
def _unravel_index(cur_ord: int, shape: np.ndarray, out_index: np.ndarray) -> None:
    for idx in range(len(shape) - 1, -1, -1):
        out_index[idx] = cur_ord % shape[idx]
        cur_ord //= shape[idx]


@njit(inline="always")
def _ravel_index(index: np.ndarray, strides: np.ndarray, n: int) -> int:
    pos = 0
    for i in range(n):
        pos += index[i] * strides[i]
    return pos


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
        Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
            a : tensor data a
            b : tensor data b

        Returns:
            New tensor data
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


# Implementations

def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, np.ndarray, np.ndarray, Storage, np.ndarray, np.ndarray], None]:
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.
    """
    def _map(
        out: Storage,
        out_shape: np.ndarray,
        out_strides: np.ndarray,
        in_storage: Storage,
        in_shape: np.ndarray,
        in_strides: np.ndarray,
    ) -> None:
        if (
            len(out_strides) != len(in_strides)
            or (out_strides != in_strides).any()
            or (out_shape != in_shape).any()
        ):
            out_index = np.empty(MAX_DIMS, np.int32)
            in_index = np.empty(MAX_DIMS, np.int32)
            for i in prange(len(out)):
                _unravel_index(i, out_shape, out_index)
                for s in range(len(in_shape)):
                    if in_shape[s] > 1:
                        in_index[s] = out_index[s + (len(out_shape) - len(in_shape))]
                    else:
                        in_index[s] = 0
                o = _ravel_index(out_index, out_strides, len(out_shape))
                j = _ravel_index(in_index, in_strides, len(in_shape))
                out[o] = fn(in_storage[j])
        else:
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
    return njit(parallel=True)(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, np.ndarray, np.ndarray, Storage, np.ndarray, np.ndarray, Storage, np.ndarray, np.ndarray], None
]:
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.
    """
    def _zip(
        out: Storage,
        out_shape: np.ndarray,
        out_strides: np.ndarray,
        a_storage: Storage,
        a_shape: np.ndarray,
        a_strides: np.ndarray,
        b_storage: Storage,
        b_shape: np.ndarray,
        b_strides: np.ndarray,
    ) -> None:
        if (
            len(out_strides) != len(a_strides)
            or len(out_strides) != len(b_strides)
            or (out_strides != a_strides).any()
            or (out_strides != b_strides).any()
            or (out_shape != a_shape).any()
            or (out_shape != b_shape).any()
        ):
            out_index = np.empty(MAX_DIMS, np.int32)
            a_index = np.empty(MAX_DIMS, np.int32)
            b_index = np.empty(MAX_DIMS, np.int32)
            for i in prange(len(out)):
                _unravel_index(i, out_shape, out_index)
                o = _ravel_index(out_index, out_strides, len(out_shape))
                for s in range(len(a_shape)):
                    if a_shape[s] > 1:
                        a_index[s] = out_index[s + (len(out_shape) - len(a_shape))]
                    else:
                        a_index[s] = 0
                j = _ravel_index(a_index, a_strides, len(a_shape))
                for s in range(len(b_shape)):
                    if b_shape[s] > 1:
                        b_index[s] = out_index[s + (len(out_shape) - len(b_shape))]
                    else:
                        b_index[s] = 0
                k = _ravel_index(b_index, b_strides, len(b_shape))
                out[o] = fn(a_storage[j], b_storage[k])
        else:
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])
    return njit(parallel=True)(_zip)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, np.ndarray, np.ndarray, Storage, np.ndarray, np.ndarray, int], None]:
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.
    """
    def _reduce(
        out: Storage,
        out_shape: np.ndarray,
        out_strides: np.ndarray,
        a_storage: Storage,
        a_shape: np.ndarray,
        a_strides: np.ndarray,
        reduce_dim: int,
    ) -> None:
        out_index = np.empty(MAX_DIMS, np.int32)
        reduce_size = a_shape[reduce_dim]
        for i in prange(len(out)):
            _unravel_index(i, out_shape, out_index)
            o = _ravel_index(out_index, out_strides, len(out_shape))
            accum = out[o]
            for s in range(reduce_size):
                out_index[reduce_dim] = s
                j = _ravel_index(out_index, a_strides, len(a_shape))
                accum = fn(accum, a_storage[j])
            out[o] = accum
    return njit(parallel=True)(_reduce)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: np.ndarray,
    out_strides: np.ndarray,
    a_storage: Storage,
    a_shape: np.ndarray,
    a_strides: np.ndarray,
    b_storage: Storage,
    b_shape: np.ndarray,
    b_strides: np.ndarray,
) -> None:
    """
    NUMBA tensor matrix multiply function.
    """
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
                out_position = (
                    i1 * out_strides[0] + i2 * out_strides[1] + i3 * out_strides[2]
                )
                out[out_position] = acc
tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)
