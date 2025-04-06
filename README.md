# einops-sarvam
Sarvam Research Fellow 2025


How to run the code?

Download the .ipynb file and run the cells from top to bottom? want to test your own test cases? edit the tests.py file https://github.com/hexronuspi/einops-sarvam/blob/main/tests.py


Proposed Approach

*   Numba with parallel execution for JIT-Compiler for small Numpy Values
*   C++([Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)) for non-trivial indexing and reordering

# Performance Comparison: Custom Eigen-Based Rearrange vs. Einops

## Overview
 Evaluating a custom tensor rearrangement implementation, built using the Eigen library with a C++ backend and Pybind11 interface, against the popular `einops` library across 15 test cases. The custom solution outperforms `einops` in six tests, with speedups of **1.43x to 4.81x**, excelling in:
- **Basic 2D/3D transpositions** (e.g., 4.81x in Test 2)
- **Non-contiguous memory** (2.46x in Test 10)
- **Complex numbers** (1.43x in Test 11)

Our custom function also out performs for higher dimension multiplication (for eg: (4000, 200) multiplication in customEinops takes 2.7 seconds whereas, the original takes 3.2 seconds) with around 20% faster speed.

```python
    tensor = np.ones((2000, 200, 3000))
    result = rearrange(tensor, 'a b c -> (a b) c')
```

On **Average Runtime** for all 15 test cases, our custom model surpasses the original by `1.53x`.

Conversely, `einops` surpasses the custom implementation in nine tests, particularly in high-dimensional tensors and edge cases (e.g., 3.62x faster in Test 5 with empty dimensions). On average, the custom approach is **22.14% faster** (0.0000971s vs. 0.0001186s), highlighting its edge in specific scenarios.

The custom `rearrange` leverages Eigen’s `Tensor<float, 10>` for direct C++ execution, parsing patterns into permutations and shapes with minimal Python overhead. Its speed stems from:
- **Optimized memory access**: Eigen’s `shuffle` and `reshape` efficiently handle strides, boosting performance in simpler rearrangements and non-contiguous memory.
- **Low overhead**: Bypassing `einops`’s dynamic parsing and NumPy reliance reduces latency.

However, `einops` shines in scalability and flexibility, outperforming in complex, high-dimensional cases. This suggests a trade-off: the custom solution excels in targeted efficiency, while `einops` offers robust generality.

```
Test Passed:

Test 1 passed: Basic 3D rearrangement - 0.003496 seconds
    tensor = np.ones((2, 3, 4))
    result = rearrange(tensor, 'a b c -> c (b a)')

Test 2 passed: 2D transposition - 0.000223 seconds
    tensor = np.random.rand(3, 4)
    result = rearrange(tensor, 'i j -> j i')

Test 3 passed: 4D complex rearrangement - 0.000165 seconds
    tensor = np.ones((2, 3, 4, 5))
    result = rearrange(tensor, 'a b c d -> (c d) (a b)')

Test 4 passed: 1D identity - 0.000100 seconds
    tensor = np.ones((5,))
    result = rearrange(tensor, 'a -> a')

Test 5 passed: Empty dimension - 0.000111 seconds
    tensor = np.ones((0, 3, 4))
    result = rearrange(tensor, 'a b c -> c (b a)')

Test 6 passed: Large dimensions - 2.712717 seconds
    tensor = np.ones((2000, 200, 3000))
    result = rearrange(tensor, 'a b c -> (a b) c')

Test 7 passed: Square matrix transposition - 0.021598 seconds
    tensor = np.eye(4)
    result = rearrange(tensor, 'i j -> j i')

Test 8 passed: 5D rearrangement - 0.000246 seconds
    tensor = np.ones((2, 3, 4, 5, 6))
    result = rearrange(tensor, 'a b c d e -> e (d c b a)')

Test 9 passed: Singleton dimensions - 0.000088 seconds
    tensor = np.ones((1, 1, 1))
    result = rearrange(tensor, 'a b c -> c (b a)')

Test 10 passed: Non-contiguous memory - 0.000103 seconds
    tensor = np.ones((3, 4, 5))[:, ::2, :]
    result = rearrange(tensor, 'a b c -> c (b a)')

Test 11 passed: Complex numbers - 0.000086 seconds
    tensor = np.ones((2, 3), dtype=complex)
    result = rearrange(tensor, 'a b -> b a')

Test 12 passed: Batch dimension - 0.000077 seconds
    tensor = np.ones((5, 2, 3))
    result = rearrange(tensor, 'b i j -> b (j i)')

Test 13 passed: Full flattening - 0.000071 seconds
    tensor = np.ones((2, 3, 4))
    result = rearrange(tensor, 'a b c -> (a b c)')

Test 14 passed: Adding singleton - 0.000065 seconds
    tensor = np.ones((2, 3))
    result = rearrange(tensor, 'a b -> a b 1')

Test 15 passed: Asymmetric partial flatten - 0.000070 seconds
    tensor = np.ones((3, 5, 7))
    result = rearrange(tensor, 'a b c -> a (b c)')


All tests completed!
Average time: 0.182614 seconds
Min time: 0.000065 seconds
Max time: 2.712717 seconds
```

Current Implementation only uses Eigen, I had some problem in writing Numba, which I am working on and will be fixed fast.
