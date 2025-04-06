import numpy as np
from rearrange import rearrange 
import pytest

def test_basic_group_split():
    x = np.random.rand(12, 10)
    result = rearrange(x, '(h w) c -> h w c', h=3)
    assert result.shape == (3, 4, 10), "Basic group split failed"
    assert np.allclose(result.reshape(12, 10), x), "Data integrity compromised"

def test_fully_specified_group():
    x = np.random.rand(12, 10)
    result = rearrange(x, '(h w) c -> h w c', h=3, w=4)
    assert result.shape == (3, 4, 10), "Fully specified group failed"
    assert np.allclose(result.reshape(12, 10), x), "Data integrity compromised"

def test_invalid_group_size():
    x = np.random.rand(12, 10)
    with pytest.raises(ValueError, match="Group size mismatch"):
        rearrange(x, '(h w) c -> h w c', h=5, w=3) 

def test_too_many_unknowns():
    x = np.random.rand(12, 10)
    with pytest.raises(ValueError, match="Too many unspecified axes"):
        rearrange(x, '(h w x) c -> h w x c')  

def test_ellipsis_basic():
    x = np.random.rand(2, 3, 4)
    result = rearrange(x, '... c -> c ...')
    assert result.shape == (4, 2, 3), "Ellipsis basic failed"
    assert np.allclose(result.transpose(1, 2, 0), x), "Data integrity compromised"

def test_ellipsis_with_group():
    x = np.random.rand(2, 12, 10)
    result = rearrange(x, 'b (h w) c -> b h w c', h=3)
    assert result.shape == (2, 3, 4, 10), "Ellipsis with group failed"
    assert np.allclose(result.reshape(2, 12, 10), x), "Data integrity compromised"

def test_singleton_axis():
    x = np.ones((1, 5, 1))
    result = rearrange(x, '1 h 1 -> h')
    assert result.shape == (5,), "Singleton axis failed"
    assert np.allclose(result, np.ones(5)), "Data integrity compromised"

def test_invalid_singleton():
    x = np.random.rand(2, 5, 3)
    with pytest.raises(ValueError, match="Expected singleton axis"):
        rearrange(x, '1 h c -> h c')

def test_empty_tensor():
    x = np.random.rand(0, 3, 4)
    result = rearrange(x, 'b h w -> w (h b)')
    assert result.shape == (4, 0), "Empty tensor failed"

def test_complex_numbers():
    x = np.ones((2, 3), dtype=complex)
    result = rearrange(x, 'h w -> w h')
    assert result.shape == (3, 2), "Complex numbers failed"
    assert result.dtype == complex, "Complex dtype not preserved"
    assert np.allclose(result, np.ones((3, 2), dtype=complex)), "Data integrity compromised"

def test_non_contiguous_memory():
    x = np.ones((3, 4, 5))[:, ::2, :]
    result = rearrange(x, 'h w c -> c (w h)')
    assert result.shape == (5, 6), "Non-contiguous memory failed"
    assert np.allclose(result.reshape(5, 3, 2).transpose(1, 2, 0), x), "Data integrity compromised"

def test_large_dimensions():
    x = np.random.rand(10, 20, 30)
    result = rearrange(x, 'a b c -> (a b) c')
    assert result.shape == (200, 30), "Large dimensions failed"
    assert np.allclose(result.reshape(10, 20, 30), x), "Data integrity compromised"

def test_full_flattening():
    x = np.random.rand(2, 3, 4)
    result = rearrange(x, 'a b c -> (a b c)')
    assert result.shape == (24,), "Full flattening failed"
    assert np.allclose(result.reshape(2, 3, 4), x), "Data integrity compromised"

def test_adding_singleton():
    x = np.ones((2, 3))
    result = rearrange(x, 'h w -> h w 1')
    assert result.shape == (2, 3, 1), "Adding singleton failed"
    assert np.allclose(result.squeeze(-1), x), "Data integrity compromised"

def test_asymmetric_flatten():
    x = np.ones((3, 5, 7))
    result = rearrange(x, 'a b c -> a (b c)')
    assert result.shape == (3, 35), "Asymmetric flatten failed"
    assert np.allclose(result.reshape(3, 5, 7), x), "Data integrity compromised"

def test_no_op():
    x = np.random.rand(5, 6)
    result = rearrange(x, 'h w -> h w')
    assert result.shape == (5, 6), "No-op failed"
    assert np.allclose(result, x), "Data integrity compromised"

def test_high_dimensional():
    x = np.random.rand(2, 3, 4, 5, 6)
    result = rearrange(x, 'a b c d e -> e (d c b a)')
    assert result.shape == (6, 120), "High-dimensional failed"
    assert np.allclose(result.reshape(6, 5, 4, 3, 2).transpose(4, 3, 2, 1, 0), x), "Data integrity compromised"

if __name__ == "__main__":
    pytest.main(["-v", __file__])