import numpy as np
import numba as nb

@nb.njit(parallel=True)
def rearrange_numba_kernel(data, out, perm, in_shape, out_shape):
    n = data.size
    for i in nb.prange(n):
        coords = np.unravel_index(i, in_shape)
        new_coords = tuple(coords[p] for p in perm)
        out_idx = np.ravel_multi_index(new_coords, out_shape)
        out.flat[out_idx] = data.flat[i]

def rearrange_numba(tensor: np.ndarray, input_spec: list, output_spec: list,
                    axes_lengths: dict) -> np.ndarray:
    from .core import Rearrange
    r = Rearrange(use_numba=False) 
    axis_sizes = r._determine_axis_sizes(tensor.shape, input_spec, axes_lengths)

    intermediate_shape = []
    input_axis_order = []
    shape_iter = iter(tensor.shape)
    ellipsis_dims = 0

    for item in input_spec:
        if item == '...':
            ellipsis_dims = len(tensor.shape) - (len(input_spec) - 1)
            intermediate_shape.extend(tensor.shape[:ellipsis_dims])
            input_axis_order.extend([f'batch_{j}' for j in range(ellipsis_dims)])
        elif item == '1':
            size = next(shape_iter)
            if size != 1:
                raise ValueError(f"Expected singleton axis, got size {size}")
            new_size = axes_lengths[output_spec[input_spec.index(item)]]
            intermediate_shape.append(new_size)  
            input_axis_order.append(output_spec[input_spec.index(item)])
        elif isinstance(item, tuple):
            group_size = next(shape_iter)
            group_axes = list(item)
            intermediate_shape.extend(axes_lengths[ax] for ax in group_axes)
            input_axis_order.extend(group_axes)
        else:
            size = next(shape_iter)
            intermediate_shape.append(size)
            input_axis_order.append(item)

    tensor = tensor.reshape(intermediate_shape)

    output_axis_order = []
    for item in output_spec:
        if item == '...':
            output_axis_order.extend([f'batch_{j}' for j in range(ellipsis_dims)])
        elif isinstance(item, tuple):
            output_axis_order.extend(item)
        else:
            output_axis_order.append(item)

    perm = np.array([input_axis_order.index(ax) for ax in output_axis_order], dtype=np.int32)
    out_shape = tuple(intermediate_shape[p] for p in perm)

    out = np.empty(out_shape, dtype=tensor.dtype)
    rearrange_numba_kernel(tensor, out, perm, tuple(intermediate_shape), out_shape)

    final_shape = []
    for item in output_spec:
        if item == '...':
            final_shape.extend(out_shape[:ellipsis_dims])
        elif isinstance(item, tuple):
            size = np.prod([axis_sizes[ax] for ax in item])
            final_shape.append(size)
        else:
            final_shape.append(axis_sizes[item])

    return out.reshape(final_shape)