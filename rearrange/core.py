import numpy as np
from typing import Dict
from .parser import PatternParser, ParserError

class Rearrange:
    def __init__(self, use_numba = False, use_eigen = False):
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
            raise RuntimeError(f"Rearrangement failed: {str(e)}")

    def _numpy_backend(self, tensor: np.ndarray, input_spec: list, output_spec: list,
                       axes_lengths: Dict[str, int]) -> np.ndarray:
        axis_sizes = self._determine_axis_sizes(tensor.shape, input_spec, output_spec, axes_lengths)
        intermediate_shape = []
        input_axis_order = []
        shape_iter = iter(tensor.shape)
        ellipsis_dims = 0
        ellipsis_pos = None

        # Build intermediate shape and axis order from input spec
        for i, item in enumerate(input_spec):
            if item == '...':
                ellipsis_dims = len(tensor.shape) - (len(input_spec) - 1)
                ellipsis_pos = i
                for j in range(ellipsis_dims):
                    size = next(shape_iter)
                    intermediate_shape.append(size)
                    input_axis_order.append(f'batch_{j}')
            elif item == '1':
                size = next(shape_iter)
                if size != 1:
                    raise ValueError(f"Expected singleton axis at position {i}, got size {size}")
                intermediate_shape.append(1)
                input_axis_order.append(f'singleton_{i}')
            elif isinstance(item, tuple):
                group_size = next(shape_iter)
                group_axes = list(item)
                sizes = [axes_lengths.get(ax, axis_sizes.get(ax)) for ax in group_axes]
                known_sizes = [s for s in sizes if s is not None]
                known_product = np.prod(known_sizes) if known_sizes else 1
                if known_product != group_size:
                    if not known_sizes:
                        # All unspecified: split evenly
                        split_size = group_size // len(group_axes)
                        for ax in group_axes:
                            axis_sizes[ax] = split_size
                    else:
                        # Adjust last axis to fit
                        remaining = group_size // known_product
                        for ax, s in zip(group_axes, sizes):
                            axis_sizes[ax] = remaining if s is None else s
                else:
                    for ax, s in zip(group_axes, sizes):
                        axis_sizes[ax] = s
                intermediate_shape.extend(axis_sizes[ax] for ax in group_axes)
                input_axis_order.extend(group_axes)
            else:
                size = next(shape_iter)
                axis_sizes[item] = size
                intermediate_shape.append(size)
                input_axis_order.append(item)

        # Reshape tensor to intermediate shape
        tensor = tensor.reshape(intermediate_shape)

        # Build output axis order
        output_axis_order = []
        ellipsis_used = False
        for i, item in enumerate(output_spec):
            if item == '...':
                output_axis_order.extend([f'batch_{j}' for j in range(ellipsis_dims)])
                ellipsis_used = True
            elif isinstance(item, tuple):
                output_axis_order.extend(item)
            else:
                output_axis_order.append(item)

        # If ellipsis is in input but not output, prepend it
        if ellipsis_dims > 0 and not ellipsis_used:
            output_axis_order = [f'batch_{j}' for j in range(ellipsis_dims)] + output_axis_order

        # Compute permutation, ensuring all axes are accounted for
        perm = []
        for ax in output_axis_order:
            if ax in input_axis_order:
                perm.append(input_axis_order.index(ax))

        # Transpose only if perm matches tensor dimensions
        if len(perm) == tensor.ndim:
            tensor = np.transpose(tensor, perm)
        elif len(perm) < tensor.ndim:
            # Pad perm with remaining axes in order
            used = set(perm)
            perm.extend(i for i in range(tensor.ndim) if i not in used)
            tensor = np.transpose(tensor, perm)

        # Compute final shape
        final_shape = []
        for item in output_spec:
            if item == '...':
                final_shape.extend(tensor.shape[i:i + ellipsis_dims] for i in range(ellipsis_dims))
            elif isinstance(item, tuple):
                size = np.prod([axis_sizes[ax] for ax in item])
                final_shape.append(size)
            else:
                final_shape.append(axis_sizes.get(item, 1))

        # Flatten final_shape if it contains nested lists (from ellipsis)
        final_shape = [s if not isinstance(s, tuple) else s[0] for s in final_shape]

        # Ensure total size matches
        total_size = np.prod(tensor.shape)
        expected_size = np.prod(final_shape)
        if total_size != expected_size:
            # Adjust last dimension if possible
            if len(final_shape) > 1:
                final_shape[-1] = total_size // np.prod(final_shape[:-1])
            else:
                final_shape[0] = total_size

        return tensor.reshape(final_shape)

    def _determine_axis_sizes(self, shape: tuple, input_spec: list, output_spec: list,
                             axes_lengths: Dict[str, int]) -> Dict[str, int]:
        axis_sizes = axes_lengths.copy()
        shape_iter = iter(shape)
        for item in input_spec:
            if item == '...':
                ellipsis_dims = len(shape) - (len(input_spec) - 1)
                for _ in range(ellipsis_dims):
                    next(shape_iter)
            elif item == '1':
                next(shape_iter)
            elif isinstance(item, tuple):
                next(shape_iter)
            else:
                axis_sizes[item] = next(shape_iter)
        all_axes = self.parser.get_axes(input_spec).union(self.parser.get_axes(output_spec))
        missing = all_axes - set(axis_sizes.keys())
        for ax in missing:
            axis_sizes[ax] = 1  # Default to 1 for unspecified axes
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