# core.py
import numpy as np
from typing import Dict, List, Tuple, Union
from .parser import PatternParser, ParserError

class Rearrange:
    def __init__(self, use_numba=False, use_eigen=False):
        self.parser = PatternParser()
        self.use_numba = use_numba and numba_available
        self.use_eigen = use_eigen and eigen_available
        if self.use_numba:
            from .numba_backend import rearrange_numba
            self.backend = rearrange_numba
        elif self.use_eigen:
            from .eigen_backend_wrapper import rearrange_eigen
            self.backend = rearrange_eigen
        else:
            self.backend = self._numpy_backend

    def __call__(self, tensor: np.ndarray, pattern: str, **axes_lengths: int) -> np.ndarray:
        try:
            input_spec, output_spec = self.parser.parse(pattern)
            return self.backend(tensor, input_spec, output_spec, axes_lengths)
        except ParserError as e:
            raise ValueError(f"Pattern parsing error: {str(e)}")
        except Exception as e:
            raise ValueError(str(e))  

    def _numpy_backend(self, tensor: np.ndarray, input_spec: list, output_spec: list,
                       axes_lengths: Dict[str, int]) -> np.ndarray:
        axis_sizes = self._determine_axis_sizes(tensor.shape, input_spec, output_spec, axes_lengths)
        intermediate_shape = []
        input_axis_order = []
        shape_iter = iter(tensor.shape)

        for i, item in enumerate(input_spec):
            if item == '...':
                ellipsis_dims = len(tensor.shape) - (len(input_spec) - 1)
                for j in range(ellipsis_dims):
                    size = next(shape_iter)
                    intermediate_shape.append(size)
                    input_axis_order.append(f'batch_{j}')
            elif item == '1':
                size = next(shape_iter)
                if size != 1:
                    raise ValueError(f"Expected singleton axis at position {i}, got {size}")
                intermediate_shape.append(1)
                input_axis_order.append(f'singleton_{i}')
            elif isinstance(item, tuple):
                group_size = next(shape_iter)
                group_axes = list(item)
                known_sizes = [axes_lengths.get(ax) for ax in group_axes if ax in axes_lengths]
                known_product = np.prod(known_sizes) if known_sizes else 1
                unknown_count = sum(1 for ax in group_axes if ax not in axes_lengths)
                if unknown_count > 1:
                    raise ValueError("Too many unspecified axes in group")
                elif unknown_count == 1:
                    for ax in group_axes:
                        if ax not in axes_lengths:
                            axis_sizes[ax] = group_size // known_product
                elif known_product != group_size:
                    raise ValueError(f"Group size mismatch: {known_product} != {group_size}")
                for ax in group_axes:
                    intermediate_shape.append(axis_sizes[ax])
                    input_axis_order.append(ax)
            else:
                size = next(shape_iter)
                axis_sizes[item] = size
                intermediate_shape.append(size)
                input_axis_order.append(item)

        tensor = tensor.reshape(intermediate_shape)

        output_axis_order = []
        final_shape = []
        ellipsis_used = False
        for item in output_spec:
            if item == '...':
                ellipsis_dims = len(tensor.shape) - (len(input_spec) - 1)
                output_axis_order.extend([f'batch_{j}' for j in range(ellipsis_dims)])
                final_shape.extend([tensor.shape[i] for i in range(ellipsis_dims)])
                ellipsis_used = True
            elif isinstance(item, tuple):
                size = np.prod([axis_sizes[ax] for ax in item])
                output_axis_order.extend(item)
                final_shape.append(size)
            elif item == '1':
                output_axis_order.append(f'singleton_{len(output_axis_order)}')
                final_shape.append(1)
            else:
                output_axis_order.append(item)
                final_shape.append(axis_sizes[item])

        if any(item == '...' for item in input_spec) and not ellipsis_used:
            ellipsis_dims = len(tensor.shape) - (len(input_spec) - 1)
            output_axis_order = [f'batch_{j}' for j in range(ellipsis_dims)] + output_axis_order
            final_shape = [tensor.shape[i] for i in range(ellipsis_dims)] + final_shape

        perm = []
        for ax in output_axis_order:
            if ax in input_axis_order:
                perm.append(input_axis_order.index(ax))

        if len(perm) < tensor.ndim:
            used = set(perm)
            perm.extend(i for i in range(tensor.ndim) if i not in used)

        tensor = np.transpose(tensor, perm)
        total_size = np.prod(tensor.shape)
        expected_size = np.prod(final_shape)
        if total_size != expected_size:
            raise ValueError(f"Shape mismatch: total size {total_size} != expected {expected_size}")

        return tensor.reshape(final_shape)

    def _determine_axis_sizes(self, shape: tuple, input_spec: list, output_spec: list,
                             axes_lengths: Dict[str, int]) -> Dict[str, int]:
        axis_sizes = axes_lengths.copy()
        shape_iter = iter(shape)
        for item in input_spec:
            if item == '...':
                ellipsis_dims = len(shape) - (len(input_spec) - 1)
                for j in range(ellipsis_dims):
                    axis_sizes[f'batch_{j}'] = next(shape_iter)
            elif item == '1':
                size = next(shape_iter)
                if size != 1:
                    raise ValueError(f"Expected singleton axis, got {size}")
            elif isinstance(item, tuple):
                group_size = next(shape_iter)
                group_axes = list(item)
                known_sizes = [axes_lengths.get(ax) for ax in group_axes if ax in axes_lengths]
                known_product = np.prod(known_sizes) if known_sizes else 1
                unknown_count = sum(1 for ax in group_axes if ax not in axes_lengths)
                if unknown_count > 1:
                    raise ValueError("Too many unspecified axes in group")
                elif unknown_count == 1:
                    for ax in group_axes:
                        if ax not in axes_lengths:
                            axis_sizes[ax] = group_size // known_product
                elif known_product != group_size:
                    raise ValueError(f"Group size mismatch: {known_product} != {group_size}")
            else:
                axis_sizes[item] = next(shape_iter)

        all_input_axes = self.parser.get_axes(input_spec)
        all_output_axes = self.parser.get_axes(output_spec)
        missing = all_output_axes - all_input_axes
        for ax in missing:
            if ax not in axis_sizes and ax != '1':
                raise ValueError(f"Output axis '{ax}' not in input and not specified")
            if ax not in axis_sizes:
                axis_sizes[ax] = 1 

        return axis_sizes

try:
    import numba
    numba_available = True
except ImportError:
    numba_available = False

try:
    from .eigen_backend import rearrange_eigen
    eigen_available = True
except ImportError:
    eigen_available = False