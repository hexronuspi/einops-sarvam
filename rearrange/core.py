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
            if "Repeated dimension name" in str(e):
                raise ValueError("Repeated dimension name")
            raise ValueError(str(e))

    @staticmethod
    def _effective_tokens(tokens: List[Union[str, Tuple[str, ...]]], output_spec: List[Union[str, Tuple[str, ...]]]) -> int:
        count = 0
        for t in tokens:
            if t == '...':
                count += 1
            elif isinstance(t, str) and t.isdigit() and t not in output_spec:
                continue
            else:
                count += 1
        return count

    def _numpy_backend(self, tensor: np.ndarray, input_spec: list, output_spec: list,
                       axes_lengths: Dict[str, int]) -> np.ndarray:
        axis_sizes = self._determine_axis_sizes(tensor.shape, input_spec, output_spec, axes_lengths)
        
        pos = 0
        intermediate_shape = []
        input_axis_order = []
        shp = tensor.shape  

        for i, token in enumerate(input_spec):
            if token == '...':
                remaining = self._effective_tokens(input_spec[i+1:], output_spec)
                ellipsis_dims = tensor.ndim - pos - remaining
                if ellipsis_dims < 0:
                    raise ValueError("Not enough axes for ellipsis")
                for j in range(ellipsis_dims):
                    size = shp[pos]
                    intermediate_shape.append(size)
                    input_axis_order.append(f'batch_{j}')
                    pos += 1
            elif token == '1':
                size = shp[pos]
                if size != 1:
                    raise ValueError(f"Expected singleton axis at position {i}, got {size}")
                intermediate_shape.append(1)
                input_axis_order.append(f'singleton_{i}')
                pos += 1
            elif isinstance(token, tuple):
                size = shp[pos]
                group_axes = list(token)
                known_sizes = [axes_lengths.get(ax) for ax in group_axes if ax in axes_lengths]
                known_product = np.prod(known_sizes) if known_sizes else 1
                unknown_count = sum(1 for ax in group_axes if ax not in axes_lengths)
                if unknown_count > 1:
                    raise ValueError("Too many unspecified axes in group")
                elif unknown_count == 1:
                    for ax in group_axes:
                        if ax not in axes_lengths:
                            axis_sizes[ax] = size // known_product
                elif known_product != size:
                    raise ValueError(f"Group size mismatch: {known_product} != {size}")
                for ax in group_axes:
                    intermediate_shape.append(axis_sizes[ax])
                    input_axis_order.append(ax)
                pos += 1
            elif isinstance(token, str) and token.isdigit():
                size = shp[pos]
                if int(token) != size:
                    raise ValueError(f"Literal dimension mismatch: expected {token}, got {size}")
                if token not in output_spec:
                    tensor = np.take(tensor, 0, axis=pos)
                    shp = tensor.shape
                else:
                    intermediate_shape.append(size)
                    input_axis_order.append(token)
                    pos += 1
            else:
                size = shp[pos]
                axis_sizes[token] = size
                intermediate_shape.append(size)
                input_axis_order.append(token)
                pos += 1

        if pos != tensor.ndim:
            raise ValueError("Mismatch between consumed axes and tensor dimensions")
        tensor = tensor.reshape(intermediate_shape)

        output_axis_order = []
        final_shape = []
        for token in output_spec:
            if token == '...':
                batch_axes = [ax for ax in input_axis_order if ax.startswith('batch_')]
                output_axis_order.extend(batch_axes)
                final_shape.extend([axis_sizes[ax] for ax in batch_axes])
            elif isinstance(token, tuple):
                size = np.prod([axis_sizes[ax] for ax in token])
                output_axis_order.extend(token)
                final_shape.append(size)
            elif token == '1':
                new_name = f'singleton_{len(output_axis_order)}'
                output_axis_order.append(new_name)
                final_shape.append(1)
            else:
                output_axis_order.append(token)
                final_shape.append(axis_sizes[token])

        perm = [input_axis_order.index(ax) for ax in output_axis_order if ax in input_axis_order]
        new_order = perm + [i for i in range(len(input_axis_order)) if i not in perm]
        try:
            tensor = np.transpose(tensor, new_order)
        except Exception as e:
            if "axes don't match array" in str(e):
                raise ValueError("Repeated dimension name")
            raise ValueError(str(e))
        kept_ndim = len(perm)
        new_shape = tensor.shape[:kept_ndim]
        tensor = tensor.reshape(new_shape)
        total_size = np.prod(new_shape)
        expected_size = np.prod(final_shape)
        if total_size != expected_size:
            raise ValueError(f"Shape mismatch: total size {total_size} != expected {expected_size}")
        return tensor.reshape(final_shape)

    def _determine_axis_sizes(self, shape: tuple, input_spec: list, output_spec: list,
                                axes_lengths: Dict[str, int]) -> Dict[str, int]:
        axis_sizes = axes_lengths.copy()
        pos = 0
        shp = shape
        for token in input_spec:
            if token == '...':
                remaining = self._effective_tokens(input_spec[input_spec.index(token)+1:], output_spec)
                ellipsis_dims = len(shp) - pos - remaining
                if ellipsis_dims < 0:
                    raise ValueError("Not enough axes for ellipsis")
                for j in range(ellipsis_dims):
                    axis_sizes[f'batch_{j}'] = shp[pos+j]
                pos += ellipsis_dims
            elif token == '1':
                size = shp[pos]
                if size != 1:
                    raise ValueError(f"Expected singleton axis, got {size}")
                pos += 1
            elif isinstance(token, tuple):
                size = shp[pos]
                group_axes = list(token)
                known_sizes = [axes_lengths.get(ax) for ax in group_axes if ax in axes_lengths]
                known_product = np.prod(known_sizes) if known_sizes else 1
                unknown_count = sum(1 for ax in group_axes if ax not in axes_lengths)
                if unknown_count > 1:
                    raise ValueError("Too many unspecified axes in group")
                elif unknown_count == 1:
                    for ax in group_axes:
                        if ax not in axes_lengths:
                            axis_sizes[ax] = size // known_product
                elif known_product != size:
                    raise ValueError(f"Group size mismatch: {known_product} != {size}")
                pos += 1
            elif isinstance(token, str) and token.isdigit():
                size = shp[pos]
                if int(token) != size:
                    raise ValueError(f"Literal dimension mismatch: expected {token}, got {size}")
                if token in output_spec:
                    axis_sizes[token] = size
                pos += 1
            else:
                axis_sizes[token] = shp[pos]
                pos += 1
        if pos != len(shp):
            raise ValueError("Mismatch between consumed axes and tensor dimensions")
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
