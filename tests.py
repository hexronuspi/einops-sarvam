import numpy as np
from rearrange import rearrange
import time

def run_tests():
    timings = {}
    
    start = time.time()
    tensor = np.ones((2, 3, 4))
    result = rearrange(tensor, 'a b c -> c (b a)')
    assert result.shape == (4, 6), "Test 1 failed"
    timings['Test 1'] = time.time() - start
    print(f"Test 1 passed: Basic 3D rearrangement - {timings['Test 1']:.6f} seconds")

    start = time.time()
    tensor = np.random.rand(3, 4)
    result = rearrange(tensor, 'i j -> j i')
    assert result.shape == (4, 3), "Test 2 failed"
    timings['Test 2'] = time.time() - start
    print(f"Test 2 passed: 2D transposition - {timings['Test 2']:.6f} seconds")

    start = time.time()
    tensor = np.ones((2, 3, 4, 5))
    result = rearrange(tensor, 'a b c d -> (c d) (a b)')
    assert result.shape == (20, 6), "Test 3 failed"
    timings['Test 3'] = time.time() - start
    print(f"Test 3 passed: 4D complex rearrangement - {timings['Test 3']:.6f} seconds")

    start = time.time()
    tensor = np.ones((5,))
    result = rearrange(tensor, 'a -> a')
    assert result.shape == (5,), "Test 4 failed"
    timings['Test 4'] = time.time() - start
    print(f"Test 4 passed: 1D identity - {timings['Test 4']:.6f} seconds")

    start = time.time()
    tensor = np.ones((0, 3, 4))
    result = rearrange(tensor, 'a b c -> c (b a)')
    assert result.shape == (4, 0), "Test 5 failed"
    timings['Test 5'] = time.time() - start
    print(f"Test 5 passed: Empty dimension - {timings['Test 5']:.6f} seconds")

    start = time.time()
    tensor = np.ones((10, 20, 30))
    result = rearrange(tensor, 'a b c -> (a b) c')
    assert result.shape == (200, 30), "Test 6 failed"
    timings['Test 6'] = time.time() - start
    print(f"Test 6 passed: Large dimensions - {timings['Test 6']:.6f} seconds")

    start = time.time()
    tensor = np.eye(4)
    result = rearrange(tensor, 'i j -> j i')
    assert result.shape == (4, 4), "Test 7 failed"
    timings['Test 7'] = time.time() - start
    print(f"Test 7 passed: Square matrix transposition - {timings['Test 7']:.6f} seconds")

    start = time.time()
    tensor = np.ones((2, 3, 4, 5, 6))
    result = rearrange(tensor, 'a b c d e -> e (d c b a)')
    assert result.shape == (6, 120), "Test 8 failed"
    timings['Test 8'] = time.time() - start
    print(f"Test 8 passed: 5D rearrangement - {timings['Test 8']:.6f} seconds")

    start = time.time()
    tensor = np.ones((1, 1, 1))
    result = rearrange(tensor, 'a b c -> c (b a)')
    assert result.shape == (1, 1), "Test 9 failed"
    timings['Test 9'] = time.time() - start
    print(f"Test 9 passed: Singleton dimensions - {timings['Test 9']:.6f} seconds")

    start = time.time()
    tensor = np.ones((3, 4, 5))[:, ::2, :]
    result = rearrange(tensor, 'a b c -> c (b a)')
    assert result.shape == (5, 6), "Test 10 failed"
    timings['Test 10'] = time.time() - start
    print(f"Test 10 passed: Non-contiguous memory - {timings['Test 10']:.6f} seconds")

    start = time.time()
    tensor = np.ones((2, 3), dtype=complex)
    result = rearrange(tensor, 'a b -> b a')
    assert result.shape == (3, 2), "Test 11 failed"
    timings['Test 11'] = time.time() - start
    print(f"Test 11 passed: Complex numbers - {timings['Test 11']:.6f} seconds")

    start = time.time()
    tensor = np.ones((5, 2, 3))
    result = rearrange(tensor, 'b i j -> b (j i)')
    assert result.shape == (5, 6), "Test 12 failed"
    timings['Test 12'] = time.time() - start
    print(f"Test 12 passed: Batch dimension - {timings['Test 12']:.6f} seconds")

    start = time.time()
    tensor = np.ones((2, 3, 4))
    result = rearrange(tensor, 'a b c -> (a b c)')
    assert result.shape == (24,), "Test 13 failed"
    timings['Test 13'] = time.time() - start
    print(f"Test 13 passed: Full flattening - {timings['Test 13']:.6f} seconds")

    start = time.time()
    tensor = np.ones((2, 3))
    result = rearrange(tensor, 'a b -> a b 1')
    assert result.shape == (2, 3, 1), "Test 14 failed"
    timings['Test 14'] = time.time() - start
    print(f"Test 14 passed: Adding singleton - {timings['Test 14']:.6f} seconds")

    start = time.time()
    tensor = np.ones((3, 5, 7))
    result = rearrange(tensor, 'a b c -> a (b c)')
    assert result.shape == (3, 35), "Test 15 failed"
    timings['Test 15'] = time.time() - start
    print(f"Test 15 passed: Asymmetric partial flatten - {timings['Test 15']:.6f} seconds")

    avg_time = sum(timings.values()) / len(timings)
    min_time = min(timings.values())
    max_time = max(timings.values())
    print(f"\nAll tests completed!")
    print(f"Average time: {avg_time:.6f} seconds")
    print(f"Min time: {min_time:.6f} seconds")
    print(f"Max time: {max_time:.6f} seconds")

if __name__ == "__main__":
    run_tests()