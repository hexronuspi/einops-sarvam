import numpy as np
from rearrange import rearrange
import pytest

def test_rearrange_basic_group_split():
    x = np.random.rand(12, 10)
    result = rearrange(x, '(h w) c -> h w c', h=3)
    assert result.shape == (3, 4, 10)
    assert np.allclose(result.reshape(12, 10), x)

def test_rearrange_fully_specified_group():
    x = np.random.rand(12, 10)
    result = rearrange(x, '(h w) c -> h w c', h=3, w=4)
    assert result.shape == (3, 4, 10)
    assert np.allclose(result.reshape(12, 10), x)

def test_rearrange_invalid_group_size():
    x = np.random.rand(12, 10)
    with pytest.raises(ValueError, match="Group size mismatch"):
        rearrange(x, '(h w) c -> h w c', h=5, w=3)

def test_rearrange_too_many_unknowns():
    x = np.random.rand(12, 10)
    with pytest.raises(ValueError, match="Too many unspecified axes"):
        rearrange(x, '(h w x) c -> h w x c')

def test_rearrange_ellipsis_basic():
    x = np.random.rand(2, 3, 4)
    result = rearrange(x, '... c -> c ...')
    assert result.shape == (4, 2, 3)
    assert np.allclose(result.transpose(1, 2, 0), x)

def test_rearrange_ellipsis_with_group():
    x = np.random.rand(2, 12, 10)
    result = rearrange(x, 'b (h w) c -> b h w c', h=3)
    assert result.shape == (2, 3, 4, 10)
    assert np.allclose(result.reshape(2, 12, 10), x)

def test_rearrange_singleton_axis():
    x = np.ones((1, 5, 1))
    result = rearrange(x, '1 h 1 -> h')
    assert result.shape == (5,)
    assert np.allclose(result, np.ones(5))

def test_rearrange_invalid_singleton():
    x = np.random.rand(2, 5, 3)
    with pytest.raises(ValueError, match="Expected singleton axis"):
        rearrange(x, '1 h c -> h c')

def test_rearrange_empty_tensor():
    x = np.random.rand(0, 3, 4)
    result = rearrange(x, 'b h w -> w (h b)')
    assert result.shape == (4, 0)

def test_rearrange_complex_numbers():
    x = np.ones((2, 3), dtype=complex)
    result = rearrange(x, 'h w -> w h')
    assert result.shape == (3, 2)
    assert result.dtype == complex
    assert np.allclose(result, np.ones((3, 2), dtype=complex))

def test_rearrange_non_contiguous_memory():
    x = np.ones((3, 4, 5))[:, ::2, :]
    result = rearrange(x, 'h w c -> c (w h)')
    assert result.shape == (5, 6)
    assert np.allclose(result.reshape(5, 3, 2).transpose(1, 2, 0), x)

def test_rearrange_large_dimensions():
    x = np.random.rand(10, 20, 30)
    result = rearrange(x, 'a b c -> (a b) c')
    assert result.shape == (200, 30)
    assert np.allclose(result.reshape(10, 20, 30), x)

def test_rearrange_full_flattening():
    x = np.random.rand(2, 3, 4)
    result = rearrange(x, 'a b c -> (a b c)')
    assert result.shape == (24,)
    assert np.allclose(result.reshape(2, 3, 4), x)

def test_rearrange_adding_singleton():
    x = np.ones((2, 3))
    result = rearrange(x, 'h w -> h w 1')
    assert result.shape == (2, 3, 1)
    assert np.allclose(result.squeeze(-1), x)

def test_rearrange_asymmetric_flatten():
    x = np.ones((3, 5, 7))
    result = rearrange(x, 'a b c -> a (b c)')
    assert result.shape == (3, 35)
    assert np.allclose(result.reshape(3, 5, 7), x)

def test_rearrange_no_op():
    x = np.random.rand(5, 6)
    result = rearrange(x, 'h w -> h w')
    assert result.shape == (5, 6)
    assert np.allclose(result, x)

def test_rearrange_high_dimensional():
    x = np.random.rand(2, 3, 4, 5, 6)
    result = rearrange(x, 'a b c d e -> e (d c b a)')
    assert result.shape == (6, 120)
    assert np.allclose(result.reshape(6, 5, 4, 3, 2).transpose(4, 3, 2, 1, 0), x)

def test_rearrange_reverse_ellipsis():
    x = np.random.rand(4, 2, 3)
    result = rearrange(x, 'c ... -> ... c')
    assert result.shape == (2, 3, 4)
    assert np.allclose(result.transpose(2, 0, 1), x)

def test_rearrange_nested_groups():
    x = np.random.rand(2 * 3 * 4, 5)
    result = rearrange(x, '(a b c) d -> a b c d', a=2, b=3, c=4)
    assert result.shape == (2, 3, 4, 5)
    assert np.allclose(result.reshape(24, 5), x)

def test_rearrange_multiple_singletons_and_inference():
    x = np.random.rand(1, 5, 1, 2)
    result = rearrange(x, '1 h 1 c -> h c')
    assert result.shape == (5, 2)
    assert np.allclose(result, x.reshape(5, 2))

def test_rearrange_mixed_numbers_and_named_dims():
    x = np.random.rand(2, 3, 4)
    result = rearrange(x, '2 b c -> b c', b=3)
    assert result.shape == (3, 4)
    assert np.allclose(result, x[0])

def test_rearrange_repeated_variable_names():
    x = np.random.rand(2, 2, 3)
    with pytest.raises(ValueError, match="Repeated dimension name"):
        rearrange(x, 'a a b -> b (a a)')

def test_rearrange_dtype_preservation_int():
    x = np.ones((2, 3), dtype=np.int32)
    result = rearrange(x, 'h w -> (h w)')
    assert result.shape == (6,)
    assert result.dtype == np.int32
    assert np.all(result == 1)

def test_rearrange_transpose_high_dim():
    x = np.random.rand(2, 3, 4, 5)
    result = rearrange(x, 'a b c d -> d c b a')
    assert result.shape == (5, 4, 3, 2)
    assert np.allclose(result, np.transpose(x, (3, 2, 1, 0)))

def test_rearrange_broadcast_and_squeeze():
    x = np.ones((1, 1, 3))
    result = rearrange(x, '1 1 c -> c')
    assert result.shape == (3,)
    assert np.allclose(result, [1, 1, 1])

def test_deep_nested_inference():
    x = np.random.rand(2 * 3 * 4 * 5, 6)
    with pytest.raises(ValueError, match="tuple index out of range"):
        rearrange(x, '((a b) c d) e -> a b c d e', a=2, b=3)

def test_complex_ellipsis_squeeze_expand():
    x = np.ones((1, 2, 1, 3, 1))
    with pytest.raises(ValueError, match="tuple index out of range"):
        rearrange(x, '1 ... 1 -> ...')

def test_overlapping_groups_disallowed():
    x = np.random.rand(6, 10)
    with pytest.raises(ValueError, match="Too many unspecified axes in group"):
        rearrange(x, '(a b) (b c) -> a b c')

def test_redundant_identity_transform():
    x = np.random.rand(3, 4, 5)
    with pytest.raises(ValueError, match="Pattern parsing error: Pattern must contain exactly one '->'"):
        rearrange(x, 'a b c -> (a b) c -> a b c', a=3, b=4)

def test_reuse_in_output_only():
    x = np.random.rand(2, 3)
    with pytest.raises(ValueError, match="Output axis 'c' not in input and not specified"):
        rearrange(x, 'a b -> a b c')

def test_multiple_ellipsis_fails():
    x = np.random.rand(2, 3, 4)
    with pytest.raises(ValueError, match="Not enough axes for ellipsis"):
        rearrange(x, '... a ... -> a ...')

def test_flatten_and_unflatten_mixed_order():
    x = np.random.rand(3, 4, 5)
    flat = rearrange(x, 'a b c -> (c a b)')
    out = rearrange(flat, '(c a b) -> a b c', a=3, b=4)
    assert np.allclose(out, x)

def test_high_dim_mixed_ellipsis_and_named():
    x = np.random.rand(2, 3, 4, 5, 6, 7)
    result = rearrange(x, 'a ... b -> b ... a')
    assert result.shape == (7, 3, 4, 5, 6, 2)
    expected = np.moveaxis(x, [0, -1], [-1, 0])
    assert np.allclose(result, expected)

def test_large_dim_with_broadcast_singleton():
    x = np.ones((1, 1000, 1))
    result = rearrange(x, '1 h 1 -> h')
    assert result.shape == (1000,)
    assert np.all(result == 1)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
