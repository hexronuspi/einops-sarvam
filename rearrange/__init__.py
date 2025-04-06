from .core import Rearrange
def rearrange(tensor, pattern, **axes_lengths):
    return Rearrange()(tensor, pattern, **axes_lengths)
__all__ = ['rearrange']