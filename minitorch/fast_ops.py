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

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.



class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"

        # This line JIT compiles your tensor_map
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

            # Other values when not sum.
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

        # Make these always be a 3 dimensional multiply
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

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # ASSIGN3.1
        if (
            len(out_strides) != len(in_strides)
            or (out_strides != in_strides).any()
            or (out_shape != in_shape).any()
        ):
            for i in prange(len(out)):
                out_index: Index = np.empty(MAX_DIMS, np.int32)
                in_index: Index = np.empty(MAX_DIMS, np.int32)
                cur_ord = i + 0
                for idx in range(len(out_shape) - 1, -1, -1):
                    sh = out_shape[idx]
                    out_index[idx] = int(cur_ord % sh)
                    cur_ord = cur_ord // sh
                for idx, s in enumerate(in_shape):
                    if s > 1:
                        in_index[idx] = out_index[idx + (len(out_shape) - len(in_shape))]
                    else:
                        in_index[idx] = 0
                o = 0
                for ind, stride in zip(out_index, out_strides):
                    o += ind * stride
                j = 0
                for ind, stride in zip(in_index, in_strides):
                    j += ind * stride
                out[o] = fn(in_storage[j])
        else:
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
        # END ASSIGN3.1

    return njit(parallel=True)(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
        fn: function maps two floats to float to apply.

    Returns:
        Tensor zip function.
    """

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
        # ASSIGN3.1
        if (
            len(out_strides) != len(a_strides)
            or len(out_strides) != len(b_strides)
            or (out_strides != a_strides).any()
            or (out_strides != b_strides).any()
            or (out_shape != a_shape).any()
            or (out_shape != b_shape).any()
        ):
            for i in prange(len(out)):
                out_index: Index = np.empty(MAX_DIMS, np.int32)
                a_index: Index = np.empty(MAX_DIMS, np.int32)
                b_index: Index = np.empty(MAX_DIMS, np.int32)
                cur_ord = i + 0
                for idx in range(len(out_shape) - 1, -1, -1):
                    sh = out_shape[idx]
                    out_index[idx] = int(cur_ord % sh)
                    cur_ord = cur_ord // sh
                o = 0
                for ind, stride in zip(out_index, out_strides):
                    o += ind * stride
                for idx, s in enumerate(a_shape):
                    if s > 1:
                        a_index[idx] = out_index[idx + (len(out_shape) - len(a_shape))]
                    else:
                        a_index[idx] = 0
                j = 0
                for ind, stride in zip(a_index, a_strides):
                    j += ind * stride
                for idx, s in enumerate(b_shape):
                    if s > 1:
                        b_index[idx] = out_index[idx + (len(out_shape) - len(b_shape))]
                    else:
                        b_index[idx] = 0
                k = 0
                for ind, stride in zip(b_index, b_strides):
                    k += ind * stride
                out[o] = fn(a_storage[j], b_storage[k])
        else:
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])
        # END ASSIGN3.1


    return njit(parallel=True)(_zip)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
        fn: reduction function mapping two floats to float.

    Returns:
        Tensor reduce function
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # ASSIGN3.1
        for i in prange(len(out)):
            out_index: Index = np.empty(MAX_DIMS, np.int32)
            reduce_size = a_shape[reduce_dim]
            cur_ord = i + 0
            for idx in range(len(out_shape) - 1, -1, -1):
                sh = out_shape[idx]
                out_index[idx] = int(cur_ord % sh)
                cur_ord = cur_ord // sh
            o = 0
            for ind, stride in zip(out_index, out_strides):
                o += ind * stride
            accum = out[o]
            j = 0
            for ind, stride in zip(out_index, a_strides):
                j += ind * stride
            step = a_strides[reduce_dim]
            for s in range(reduce_size):
                accum = fn(accum, a_storage[j])
                j += step
            out[o] = accum
        # END ASSIGN3.1


    return njit(parallel=True)(_reduce)  # type: ignore


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
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # ASSIGN3.2
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
    # END ASSIGN3.2


tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)