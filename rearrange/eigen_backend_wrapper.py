import numpy as np
from .eigen_backend import rearrange_eigen

def rearrange_eigen(tensor: np.ndarray, input_spec: list, output_spec: list,
                    axes_lengths: dict) -> np.ndarray:
    tensor = np.ascontiguousarray(tensor)
    input_spec_str = ' '.join(str(x) for x in input_spec)
    output_spec_str = ' '.join(str(x) for x in output_spec)
    return rearrange_eigen(tensor, input_spec_str, output_spec_str, axes_lengths)