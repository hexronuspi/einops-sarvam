# einops-sarvam
**Sarvam Research Fellow 2025**

---

[Email](mailto:hexronus@gmail.com) - hexronus@gmail.com \
[Portfolio](https://hexronus.vercel.app/) - https://hexronus.vercel.app \
[LinkedIn](https://www.linkedin.com/in/hexronus/) - https://www.linkedin.com/in/hexronus/


## Getting Started: Running the Code  

To test the code, download the `.ipynb` notebook from the repository, open it in google colab(for preinstalled libraries) and execute the cells sequentially from top to bottom. For custom test cases, modify the `tests.py` file, available at:  
[**https://github.com/hexronuspi/einops-sarvam/blob/main/tests.py**](https://github.com/hexronuspi/einops-sarvam/blob/main/tests.py). This `tests.py` file also gets downloded when the cells are run in sequential manner.

an easier alternative, here's the [colab](https://colab.research.google.com/drive/1i1BLGdvP5knlcfmA8M9wn-teBlnq2smk?usp=sharing) link, just select run all.

---

## Proposed Approach  

Our methodology uses two paradigms to redefine tensor rearrangement efficiency:  

- **Numba with Parallel Execution**: Leveraging JIT compilation to accelerate operations on small NumPy arrays, harnessing parallelism for optimal performance.  
- **C++ with Eigen**: Employing the Eigen library ([https://eigen.tuxfamily.org](https://eigen.tuxfamily.org)) for sophisticated indexing and reordering, particularly suited to non-trivial tensor manipulations.  

This strategy balances acceleration with low-level C++ execution.

---

## Performance Comparison: Custom Eigen-Based Rearrange vs. Einops  

### Overview  

We performed a rigorous evaluation of tensor rearrangement implementation with the Eigen library, a C++ backend, and a Pybind11 interface—against the widely adopted `einops` library. Our work tested 15 test cases.  

The custom solution surpasses `einops` for closed testing on custom tests, achieving speedups ranging from **1.43x to 4.81x**, with standout performance in:  
- **Basic 2D/3D Transpositions**: Up to 4.81x faster, showcasing exceptional efficiency in fundamental operations.  
- **Non-Contiguous Memory Access**: A 2.46x speedup, highlighting adept handling of complex memory layouts.  
- **Complex Number Operations**: A 1.43x advantage, demonstrating robustness across data types.  

Moreover, our approach excels in higher-dimensional tensor multiplications. For instance, a `(4000, 200)` multiplication completes in **2.7 seconds** with our custom implementation, compared to **3.2 seconds** with `einops`—a **1.28x** speedup. 

## Performance Comparison: Custom Eigen-Based Rearrange vs. Einops  

### Overview  

We performed a rigorous evaluation of tensor rearrangement implementation with the Eigen library, a C++ backend, and a Pybind11 interface—against the widely adopted `einops` library. Our work tested 15 test cases.  

The custom solution surpasses `einops` for closed testing on custom tests, achieving speedups ranging from **1.43x to 4.81x**, with standout performance in:  
- **Basic 2D/3D Transpositions**: Up to 4.81x faster, showcasing exceptional efficiency in fundamental operations.  
- **Non-Contiguous Memory Access**: A 2.46x speedup, highlighting adept handling of complex memory layouts.  
- **Complex Number Operations**: A 1.43x advantage, demonstrating robustness across data types.  

Moreover, our approach excels in higher-dimensional tensor multiplications. For instance, a `(4000, 200)` multiplication completes in **2.7 seconds** with our custom implementation, compared to **3.2 seconds** with `einops`—a **1.28x** speedup. 

The custom `rearrange` leverages Eigen’s `Tensor<float, 10>` for direct C++ execution, parsing patterns into permutations and shapes with minimal Python overhead. Its speed stems from:
- **Optimized memory access**: Eigen’s `shuffle` and `reshape` efficiently handle strides, boosting performance in simpler rearrangements and non-contiguous memory.
- **Low overhead**: Bypassing `einops`’s dynamic parsing and NumPy reliance reduces latency.

However, `einops` shines in scalability and flexibility, outperforming in complex, high-dimensional cases. This suggests a trade-off: the custom solution excels in targeted efficiency, while `einops` offers robust generality.

The code was tested again the doc test cases and other test cases, both result screenshots are displayed here, the other test cases are also present in the github repository as tests.py,

#### Given Tests in docs
![given_test_run](https://github.com/hexronuspi/einops-sarvam/blob/main/test_run_images/given_test_run.png)

#### Hidden Tests taken on own
![hidden_test_run](https://github.com/hexronuspi/einops-sarvam/blob/main/test_run_images/hidden_test_run.png)


#### Error fallback
![error_test_run](https://github.com/hexronuspi/einops-sarvam/blob/main/test_run_images/error_fall_back.png)

