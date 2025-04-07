
from typing import Callable, Optional

import numba
from numba import cuda

from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
)
from .tensor_ops import MapProto, TensorOps

THREADS_PER_BLOCK = 32

####################################################
# Helper device functions for CUDA kernels
####################################################
@cuda.jit(device=True, inline=True)
def _cuda_compute_multi_index(i, shape, index_out):
    # Note: shape and index_out are treated as arrays.
    for idx in range(int(len(shape)) - 1, -1, -1):
        index_out[idx] = int(i % shape[idx])
        i //= shape[idx]


@cuda.jit(device=True, inline=True)
def _cuda_compute_offset(index, strides, n):
    off = 0
    for j in range(n):
        off += index[j] * strides[j]
    return off


##############################################
#  CudaOps Class (public API)
##############################################
class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        f = tensor_map(cuda.jit(device=True)(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        f = tensor_zip(cuda.jit(device=True)(fn))

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
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        f = tensor_reduce(cuda.jit(device=True)(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))
            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )
            return out_a

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

        a_s, b_s = a.shape[:-2], b.shape[:-2]
        m = max(len(a_s), len(b_s))
        c_rev = [0] * m
        a_rev = list(reversed(a_s))
        b_rev = list(reversed(b_s))
        for i in range(m):
            if i >= len(a_s):
                c_rev[i] = b_rev[i]
            elif i >= len(b_s):
                c_rev[i] = a_rev[i]
            else:
                c_rev[i] = max(a_rev[i], b_rev[i])
                if a_rev[i] != c_rev[i] and a_rev[i] != 1:
                    raise RuntimeError(f"indexing error: Broadcast failure {a_s} {b_s}")
                if b_rev[i] != c_rev[i] and b_rev[i] != 1:
                    raise RuntimeError(f"indexing error: Broadcast failure {a_s} {b_s}")
        ls = list(reversed(c_rev))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


##############################################
# CUDA Kernel implementations using device helpers
##############################################
def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, int, Storage, Shape, Strides], None]:
    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < out_size:
            cur = i
            _cuda_compute_multi_index(cur, out_shape, out_index)
            o = _cuda_compute_offset(out_index, out_strides, len(out_shape))
            for s_i in range(len(in_shape)):
                if in_shape[s_i] > 1:
                    in_index[s_i] = out_index[s_i + (len(out_shape) - len(in_shape))]
                else:
                    in_index[s_i] = 0
            j = _cuda_compute_offset(in_index, in_strides, len(in_shape))
            out[o] = fn(in_storage[j])
    return cuda.jit()(_map)


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, int, Storage, Shape, Strides, Storage, Shape, Strides],
    None,
]:
    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < out_size:
            cur = i
            _cuda_compute_multi_index(cur, out_shape, out_index)
            o = _cuda_compute_offset(out_index, out_strides, len(out_shape))
            for s_i in range(len(a_shape)):
                if a_shape[s_i] > 1:
                    a_index[s_i] = out_index[s_i + (len(out_shape) - len(a_shape))]
                else:
                    a_index[s_i] = 0
            j = _cuda_compute_offset(a_index, a_strides, len(a_shape))
            for s_i in range(len(b_shape)):
                if b_shape[s_i] > 1:
                    b_index[s_i] = out_index[s_i + (len(out_shape) - len(b_shape))]
                else:
                    b_index[s_i] = 0
            k = _cuda_compute_offset(b_index, b_strides, len(b_shape))
            out[o] = fn(a_storage[j], b_storage[k])
    return cuda.jit()(_zip)


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, int, Storage, Shape, Strides, int, float], None
]:
    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = int(cuda.blockIdx.x)
        pos = int(cuda.threadIdx.x)
        cache[pos] = reduce_value
        if out_pos < out_size:
            cur = out_pos
            _cuda_compute_multi_index(cur, out_shape, out_index)
            o = _cuda_compute_offset(out_index, out_strides, len(out_shape))
            out_index[reduce_dim] = out_index[reduce_dim] * BLOCK_DIM + pos
            if out_index[reduce_dim] < a_shape[reduce_dim]:
                in_a = _cuda_compute_offset(out_index, a_strides, len(a_shape))
                cache[pos] = a_storage[in_a]
                cuda.syncthreads()
                x = 0
                while 2**x < BLOCK_DIM:
                    j = 2**x
                    if pos % (j * 2) == 0:
                        cache[pos] = fn(cache[pos], cache[pos + j])
                    cuda.syncthreads()
                    x += 1
            if pos == 0:
                out[o] = cache[0]
    return cuda.jit()(_reduce)


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    batch = int(cuda.blockIdx.z)

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    i = int(cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x)
    j = int(cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y)
    
    pi = int(cuda.threadIdx.x)
    pj = int(cuda.threadIdx.y)

    accum = 0.0
    for k_start in range(0, a_shape[2], BLOCK_DIM):
        k = k_start + pj
        if i < a_shape[1] and k < a_shape[2]:
            a_shared[pi, pj] = a_storage[
                a_batch_stride * batch + a_strides[1] * i + a_strides[2] * k
            ]
        k = k_start + pi
        if j < b_shape[2] and k < b_shape[1]:
            b_shared[pi, pj] = b_storage[
                b_batch_stride * batch + b_strides[1] * k + b_strides[2] * j
            ]
        cuda.syncthreads()

        for k in range(BLOCK_DIM):
            if (k_start + k) < a_shape[2]:
                accum += a_shared[pi, k] * b_shared[k, pj]
    if i < out_shape[1] and j < out_shape[2]:
        out[a_batch_stride * batch + out_strides[1] * i + out_strides[2] * j] = accum
    return

tensor_matrix_multiply = cuda.jit(_tensor_matrix_multiply)
