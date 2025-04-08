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

Our methodology uses:  

- **Numpy**: Numpy execution  
- **Numba with Parallel Execution**: Leveraging JIT compilation to accelerate operations on small NumPy arrays, harnessing parallelism for optimal performance.  
- **C++ with Eigen**: Employing the Eigen library ([https://eigen.tuxfamily.org](https://eigen.tuxfamily.org)) for sophisticated indexing and reordering, particularly suited to non-trivial tensor manipulations.  

This strategy balances acceleration with low-level C++ execution.

---

## Performance Comparison: Custom Eigen-Based Rearrange vs. Einops  

### Tests

*  All tests validate shape, value equivalence (np.allclose), and dtype preservation.
*  Fallbacks are fail-fast with clear error messages.
*  Support for inference, ellipses, singleton handling, and nested flatten/unflatten.

### Overview  

We performed an evaluation of tensor rearrangement implementation. Our work tested 34 test cases.  

The custom solution passed all closed testing on custom tests, achieving speedups ranging from **1.43x to 4.81x**, with standout performance in:  
- **Basic 2D/3D Transpositions**: Up to 4.81x faster, showcasing exceptional efficiency in fundamental operations.  
- **Non-Contiguous Memory Access**: A 2.46x speedup, highlighting adept handling of complex memory layouts.  
- **Complex Number Operations**: A 1.43x advantage, demonstrating robustness across data types.  


The code was tested again the doc test cases and other test cases, both result screenshots are displayed here, the other test cases are also present in the github repository as tests.py,

#### Given Tests in docs
![given_test_run](https://github.com/hexronuspi/einops-sarvam/blob/main/test_run_images/given_test_run.png)

#### Hidden Tests taken on own
![hidden_test_run](https://github.com/hexronuspi/einops-sarvam/blob/main/test_run_images/hidden_test_run.png)


#### Error fallback
![error_test_run](https://github.com/hexronuspi/einops-sarvam/blob/main/test_run_images/error_fall_back.png)

