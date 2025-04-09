
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type, Sequence

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import MAX_DIMS, shape_broadcast

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# Helper functions for CPU kernels.
def fill_index(flat: int, shape: Sequence[int], out_index: np.ndarray) -> None:
    cur = flat
    for idx in range(len(shape) - 1, -1, -1):
        sh = shape[idx]
        out_index[idx] = int(cur % sh)
        cur //= sh

def compute_position(index: np.ndarray, strides: Sequence[int]) -> int:
    pos = 0
    for ind, s in zip(index, strides):
        pos += ind * s
    return pos

def fill_broadcast_index(out_index: np.ndarray, out_shape: Sequence[int],
                         in_shape: Sequence[int], in_index: np.ndarray) -> None:
    offset = len(out_shape) - len(in_shape)
    for i, s in enumerate(in_shape):
        in_index[i] = out_index[i + offset] if s > 1 else 0


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ...) -> Tensor:
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        pass

    @staticmethod
    def cmap(fn: Callable[[float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        pass

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """
        Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
            ops : tensor operations object see `tensor_ops.py`

        Returns :
            A collection of tensor functions
        """
        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.id_cmap = ops.cmap(operators.id)
        self.inv_map = ops.map(operators.inv)
        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)
        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        # Call the helper kernel generator.
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            # Call low‐level kernel.
            f(*out.tuple(), *a.tuple())
            return out
        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                # Use the provided helper in tensor_data if possible.
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out
        return ret

    @staticmethod
    def reduce(fn: Callable[[float, float], float], start: float = 0.0) -> Callable[["Tensor", int], "Tensor"]:
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start
            f(*out.tuple(), *a.tuple(), dim)
            return out
        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Low-level implementations.
def tensor_map(fn: Callable[[float], float]) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    Low-level implementation of tensor map.
    """
    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = np.zeros(MAX_DIMS, np.int16)
        in_index = np.zeros(MAX_DIMS, np.int16)
        for i in range(len(out)):
            fill_index(i, out_shape, out_index)
            fill_broadcast_index(out_index, out_shape, in_shape, in_index)
            o = compute_position(out_index, out_strides)
            j = compute_position(in_index, in_strides)
            out[o] = fn(in_storage[j])
    return _map


def tensor_zip(fn: Callable[[float, float], float]) -> Callable[
        [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    Low-level implementation of tensor zip.
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
        out_index = np.zeros(MAX_DIMS, np.int32)
        a_index = np.zeros(MAX_DIMS, np.int32)
        b_index = np.zeros(MAX_DIMS, np.int32)
        for i in range(len(out)):
            fill_index(i, out_shape, out_index)
            a_offset = len(out_shape) - len(a_shape)
            b_offset = len(out_shape) - len(b_shape)
            for s_i, s in enumerate(a_shape):
                a_index[s_i] = out_index[s_i + a_offset] if s > 1 else 0
            for s_i, s in enumerate(b_shape):
                b_index[s_i] = out_index[s_i + b_offset] if s > 1 else 0
            o = compute_position(out_index, out_strides)
            j = compute_position(a_index, a_strides)
            k = compute_position(b_index, b_strides)
            out[o] = fn(a_storage[j], b_storage[k])
    return _zip


def tensor_reduce(fn: Callable[[float, float], float]) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    Low-level implementation of tensor reduce.
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
        out_index = np.zeros(MAX_DIMS, np.int32)
        reduce_size = a_shape[reduce_dim]
        for i in range(len(out)):
            fill_index(i, out_shape, out_index)
            o = compute_position(out_index, out_strides)
            for s in range(reduce_size):
                out_index[reduce_dim] = s
                j = compute_position(out_index, a_strides)
                out[o] = fn(out[o], a_storage[j])
    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
