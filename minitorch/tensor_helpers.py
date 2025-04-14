
from .tensor_data import MAX_DIMS
import numpy as np

def fill_index(flat_index: int, shape: np.ndarray, out_index: np.ndarray) -> None:
    """
    Fills the provided out_index array (of length len(shape)) with the multidimensional
    index corresponding to the flat index and shape.
    """
    cur = flat_index
    for idx in range(len(shape) - 1, -1, -1):
        out_index[idx] = int(cur % shape[idx])
        cur //= shape[idx]

def compute_offset(index: np.ndarray, strides: np.ndarray) -> int:
    """
    Computes the offset within a storage array given a multi‐dimensional index and strides.
    """
    offset = 0
    for i, stride in zip(index, strides):
        offset += int(i) * int(stride)
    return offset

def fill_broadcast_index(out_index: np.ndarray, in_shape: np.ndarray, in_index: np.ndarray) -> None:
    """
    Given an already computed out_index (for the output tensor with shape out_shape),
    fills in the in_index corresponding to an input tensor whose shape in_shape is broadcast
    against the out_shape.
    """
    offset = len(out_index) - len(in_shape)
    for i, s in enumerate(in_shape):
        in_index[i] = out_index[i + offset] if s > 1 else 0



from .tensor_data import MAX_DIMS
import numpy as np

def fill_index(flat_index: int, shape: np.ndarray, out_index: np.ndarray) -> None:
    """
    Fills the provided out_index array (of length len(shape)) with the multidimensional
    index corresponding to the flat index and shape.
    """
    cur = flat_index
    for idx in range(len(shape) - 1, -1, -1):
        out_index[idx] = int(cur % shape[idx])
        cur //= shape[idx]

def compute_offset(index: np.ndarray, strides: np.ndarray) -> int:
    """
    Computes the offset within a storage array given a multi‐dimensional index and strides.
    """
    offset = 0
    for i, stride in zip(index, strides):
        offset += int(i) * int(stride)
    return offset

def fill_broadcast_index(out_index: np.ndarray, in_shape: np.ndarray, in_index: np.ndarray) -> None:
    """
    Given an already computed out_index (for the output tensor with shape out_shape),
    fills in the in_index corresponding to an input tensor whose shape in_shape is broadcast
    against the out_shape.
    """
    offset = len(out_index) - len(in_shape)
    for i, s in enumerate(in_shape):
        in_index[i] = out_index[i + offset] if s > 1 else 0
