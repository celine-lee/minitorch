
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import MAX_DIMS

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
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
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float]
    ) -> Callable[[Tensor, Tensor], Tensor]:
        f = tensor_zip(fn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            if a.shape != b.shape:
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
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        f = tensor_reduce(fn)

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
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


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
        out_index = np.zeros(MAX_DIMS, np.int16)
        in_index = np.zeros(MAX_DIMS, np.int16)
        for i in range(len(out)):
            cur_ord = i
            for idx in range(len(out_shape) - 1, -1, -1):
                sh = out_shape[idx]
                out_index[idx] = int(cur_ord % sh)
                cur_ord //= sh
            for s_i, s in enumerate(in_shape):
                if s > 1:
                    in_index[s_i] = out_index[s_i + (len(out_shape) - len(in_shape))]
                else:
                    in_index[s_i] = 0
            o = 0
            for ind, stride in zip(out_index, out_strides):
                o += ind * stride
            j = 0
            for ind, stride in zip(in_index, in_strides):
                j += ind * stride
            out[o] = fn(in_storage[j])
    return _map


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
        out_index = np.zeros(MAX_DIMS, np.int32)
        a_index = np.zeros(MAX_DIMS, np.int32)
        b_index = np.zeros(MAX_DIMS, np.int32)
        for i in range(len(out)):
            cur_ord = i
            for idx in range(len(out_shape) - 1, -1, -1):
                sh = out_shape[idx]
                out_index[idx] = int(cur_ord % sh)
                cur_ord //= sh
            o = 0
            for ind, stride in zip(out_index, out_strides):
                o += ind * stride
            for s_i, s in enumerate(a_shape):
                if s > 1:
                    a_index[s_i] = out_index[s_i + (len(out_shape) - len(a_shape))]
                else:
                    a_index[s_i] = 0
            j = 0
            for ind, stride in zip(a_index, a_strides):
                j += ind * stride
            for s_i, s in enumerate(b_shape):
                if s > 1:
                    b_index[s_i] = out_index[s_i + (len(out_shape) - len(b_shape))]
                else:
                    b_index[s_i] = 0
            k = 0
            for ind, stride in zip(b_index, b_strides):
                k += ind * stride
            out[o] = fn(a_storage[j], b_storage[k])
    return _zip


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
        out_index = np.zeros(MAX_DIMS, np.int32)
        reduce_size = a_shape[reduce_dim]
        for i in range(len(out)):
            cur_ord = i
            for idx in range(len(out_shape) - 1, -1, -1):
                sh = out_shape[idx]
                out_index[idx] = int(cur_ord % sh)
                cur_ord //= sh
            o = 0
            for ind, stride in zip(out_index, out_strides):
                o += ind * stride
            for s in range(reduce_size):
                out_index[reduce_dim] = s
                j = 0
                for ind, stride in zip(out_index, a_strides):
                    j += ind * stride
                out[o] = fn(out[o], a_storage[j])
    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
