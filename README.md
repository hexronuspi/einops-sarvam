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

Our custom function also out performs for biggerr dimension multiplication with around 20% faster speed.

On **Average Runtime** for all 15 test cases, our custom model surpasses the original by `1.53x`.

Conversely, `einops` surpasses the custom implementation in nine tests, particularly in high-dimensional tensors and edge cases (e.g., 3.62x faster in Test 5 with empty dimensions). On average, the custom approach is **22.14% faster** (0.0000971s vs. 0.0001186s), highlighting its edge in specific scenarios.

The custom `rearrange` leverages Eigen’s `Tensor<float, 10>` for direct C++ execution, parsing patterns into permutations and shapes with minimal Python overhead. Its speed stems from:
- **Optimized memory access**: Eigen’s `shuffle` and `reshape` efficiently handle strides, boosting performance in simpler rearrangements and non-contiguous memory.
- **Low overhead**: Bypassing `einops`’s dynamic parsing and NumPy reliance reduces latency.

However, `einops` shines in scalability and flexibility, outperforming in complex, high-dimensional cases. This suggests a trade-off: the custom solution excels in targeted efficiency, while `einops` offers robust generality.

Current Implementation only uses Eigen, I had some problem in writing Numba, which I am working on and will be fixed fast.
