import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def im2col_nd(data, block_size, overlap):
    """
    Returns a 2d array of shape flat(block_size) by data.shape/block_size made
    from blocks of a nd array.
    """

    block_size = np.array(block_size, dtype=np.int32)[:3]
    overlap = np.array(overlap, dtype=np.int32)[:3]

    if len(overlap) < 3 or len(block_size) < 3 or data.ndim < 3:
        raise ValueError('only 3D or 4D is supported!')

    if (overlap < 0).any() or (block_size < overlap).any():
        raise ValueError(f'Invalid overlap = {overlap} value, it must lie between 0 and {np.min(block_size)-1}')

    data = padding(data, block_size)

    if data.ndim == 4:
        block_size = np.append(block_size, data.shape[-1])
        overlap = np.append(overlap, 0)

    skip = block_size - overlap
    out = sliding_window_view(data, block_size)
    out = out[::skip[0], ::skip[1], ::skip[2]].reshape(-1, np.prod(block_size)).T
    return out


def col2im_nd(data, block_size, end_shape, overlap, weights=None, dtype=np.float32):
    """
    Returns a nd array of shape end_shape from a 2D array made of flatenned
    block that had originally a shape of block_size.
    Inverse function of im2col_nd.
    """

    block_size = np.array(block_size, dtype=np.int32)[:3]
    overlap = np.array(overlap, dtype=np.int32)[:3]
    end_shape = np.array(end_shape, dtype=np.int32)

    if len(overlap) < 3 or len(block_size) < 3 or len(end_shape) < 3:
        raise ValueError('only 3D or 4D is supported!')

    if (overlap < 0).any() or ((block_size < overlap).any()):
        raise ValueError(f'Invalid overlap value, it must lie between 0 and {min(block_size) - 1}', overlap, block_size)

    if weights is None:
        weights = np.ones(data.shape[1], dtype=np.float32)

    if weights.shape[0] != data.shape[1]:
        error = f'weights array shape {weights.shape} does not fit with array shape {data.shape}'
        raise ValueError(error)

    if np.any(np.mod(end_shape[:3], block_size)):
        error = f'Padding is wrong and does not fit {end_shape} evenly, check the shape for the resulting array'
        raise ValueError(error, block_size)

    out = np.zeros(end_shape, dtype=dtype)
    div = np.zeros(end_shape[:3], dtype=dtype)

    if out.ndim == 4:
        block_size = np.append(block_size, out.shape[-1])
        overlap = np.append(overlap, 0)

    skip = block_size - overlap

    out_view = sliding_window_view(out, block_size,     writeable=True)[::skip[0], ::skip[1], ::skip[2]]
    div_view = sliding_window_view(div, block_size[:3], writeable=True)[::skip[0], ::skip[1], ::skip[2]]


    shape = out_view.shape[:4]
    if len(end_shape) == 3:
        shape = shape[:3]

    # The views will index at the right place in the original underlying arrays magically
    out_view += data.T.reshape(out_view.shape)
    div_view += weights.reshape(out_view.shape[:3] + (1,1,1))

    if out.ndim == 4:
        div = div[..., None]

    with np.errstate(divide='ignore'):
        out /= div # We need to properly unpad later on
    return out


def padding(data, block_size):
    """
    Pad A at the end so that block_size will cut an integer number of blocks
    across all dimensions. A is padded with 0s.
    """

    shape = data.shape[:3]
    block_size = np.array(block_size)[:3]

    if np.all(np.mod(shape, block_size) == 0):
        return data

    ceil = (block_size - np.mod(shape, block_size)) % block_size
    padding = (0, int(ceil[0])), (0, int(ceil[1])), (0, int(ceil[2]))

    if data.ndim == 4:
        padding += ((0, 0),) # no padding in diffusion dimension
    padded = np.pad(data, padding)

    return padded


def unpad(data, original_shape):
    if data.ndim != len(original_shape):
        error = f'data shape {data.shape} does not fit with with the unpad shape {original_shape}'
        raise ValueError(error)

    return data[:original_shape[0], :original_shape[1], :original_shape[2]]
