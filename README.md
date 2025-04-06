# einops-sarvam
**Sarvam Research Fellow 2025**  

---

## Getting Started: Running the Code  

To test the code, download the `.ipynb` notebook from the repository and execute the cells sequentially from top to bottom. For custom test cases, modify the `tests.py` file, available at:  
[**https://github.com/hexronuspi/einops-sarvam/blob/main/tests.py**](https://github.com/hexronuspi/einops-sarvam/blob/main/tests.py). This `tests.py` file also gets downloded when the cells are run in sequential manner.

This streamlined workflow ensures reproducibility while offering flexibility for experimentation.

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

The custom solution surpasses `einops`, achieving speedups ranging from **1.43x to 4.81x**, with standout performance in:  
- **Basic 2D/3D Transpositions**: Up to 4.81x faster, showcasing exceptional efficiency in fundamental operations.  
- **Non-Contiguous Memory Access**: A 2.46x speedup, highlighting adept handling of complex memory layouts.  
- **Complex Number Operations**: A 1.43x advantage, demonstrating robustness across data types.  

Moreover, our approach excels in higher-dimensional tensor multiplications. For instance, a `(4000, 200)` multiplication completes in **2.7 seconds** with our custom implementation, compared to **3.2 seconds** with `einops`—a **1.28x** speedup. 

The code was tested again the doc test cases and other test cases, both result screenshots are displayed here, the other test cases are also present in the github repository as tests.py,

![given_test_run](https://github.com/hexronuspi/einops-sarvam/blob/main/test_run_images/given_test_run.png)

![hidden_test_run](https://github.com/hexronuspi/einops-sarvam/blob/main/test_run_images/hidden_test_run.png)


Note: Currently `Numba` does not work currently and I am working on it, I will try my best to update the Github repository by 7th April 2025 11:00pm, the deadline for the task.
