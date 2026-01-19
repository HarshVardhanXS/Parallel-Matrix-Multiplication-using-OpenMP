# Parallel Matrix Multiplication using OpenMP

## Project Overview
This project implements both **serial and parallel matrix multiplication** using OpenMP in **C** for execution in **GitHub Codespaces (VS Code Web)** environment using **GCC**. It demonstrates the use of dynamic memory allocation for square matrices (NxN) and measures execution time for performance analysis.

## Project Requirements Met
✓ Implement both serial and parallel matrix multiplication
✓ Use OpenMP to parallelize the outer loop with `#pragma omp parallel for`
✓ Use dynamically allocated square matrices (NxN)
✓ Initialize matrices with sample values
✓ Measure execution time using `omp_get_wtime()`
✓ Print serial execution time, parallel execution time, and speedup
✓ Ensure correct shared/private variable usage (no race conditions)
✓ Code compiles and runs with GCC using OpenMP in Codespaces
✓ Add clear comments suitable for an academic HPC project
✓ Verify correctness by comparing serial and parallel results

## Files
- `matrix_openmp.c` - Main C source file containing both serial and parallel implementations

## Compilation in GitHub Codespaces

### Using the provided command:
```bash
gcc matrix_openmp.c -fopenmp -o matrix
```

### Compilation Flags:
- `matrix_openmp.c` - Source file
- `-fopenmp` - Enables OpenMP support in GCC
- `-o matrix` - Output executable name

## Execution

Run the compiled executable:
```bash
./matrix
```

### Output Example:
```
========================================================
Parallel Matrix Multiplication using OpenMP
========================================================

Matrix size: 1000 x 1000
Total elements per matrix: 1000000

Initializing matrices...
Matrices initialized successfully.

========== SERIAL EXECUTION ==========
Serial execution time: 2.345678 seconds

========== PARALLEL EXECUTION ==========
Parallel execution time: 0.687543 seconds
Number of threads used: 4

========== PERFORMANCE ANALYSIS ==========
Speedup: 3.4128 x
Efficiency: 85.32 %

========== CORRECTNESS VERIFICATION ==========
Comparing serial and parallel results...
✓ SUCCESS: Serial and parallel results match!
✓ Matrix multiplication is correct.

========== CLEANUP ==========
Freeing dynamically allocated memory...
Memory freed successfully.

========================================================
Program completed successfully!
========================================================
```

## Algorithm Details

### Serial Algorithm
- Three nested loops: O(N³) time complexity
- No parallelization
- Baseline for performance comparison
- Each element C[i][j] = sum of A[i][k] * B[k][j] for k = 0 to N-1

### Parallel Algorithm
- Parallelizes the outer two loops using `#pragma omp parallel for collapse(2)`
- Distributes row and column iterations across threads
- Each thread works on independent matrix elements (no race conditions)
- Shared variables: A, B, C, N (accessible to all threads)
- Private variables: i, j, k (each thread has its own copy)

### OpenMP Directives
- `#pragma omp parallel for collapse(2)` - Parallelizes nested loops
- No explicit synchronization needed as each thread works on unique data

## Performance Metrics

The program outputs:
1. **Serial Execution Time** - Baseline performance
2. **Parallel Execution Time** - Performance with multiple threads
3. **Speedup** - Ratio of serial time to parallel time (Serial Time / Parallel Time)
4. **Efficiency** - Speedup / Number of Threads × 100%

### Expected Performance
- Speedup typically ranges from 3-4x on 4-thread systems
- Efficiency around 75-100% depending on system load
- Larger matrices (N > 1000) show better parallelization benefits

## Customization

### Adjust Matrix Size
Modify the `N` value in `main()`:
```c
int N = 1000;  // Change to desired size
```

### Adjust Number of Threads
Modify the `num_threads` value in `main()`:
```c
int num_threads = 4;  // Change to desired number
omp_set_num_threads(num_threads);
```

## Memory Management
- Dynamic memory allocation using `malloc()` and `free()`
- All allocated matrices are properly freed after execution
- No memory leaks

## Academic Use
This project is suitable for:
- High-Performance Computing (HPC) coursework
- Parallel Programming assignments
- OpenMP tutorial and learning
- Performance analysis studies
- Optimization research

## References
- [OpenMP Official Documentation](https://www.openmp.org/)
- [GCC OpenMP Support](https://gcc.gnu.org/projects/gomp/)
- [GitHub Codespaces Documentation](https://docs.github.com/en/codespaces)

## Author Notes
This implementation follows best practices for parallel programming:
- Clear separation of concerns (functions for each operation)
- Comprehensive comments explaining algorithms and directives
- Proper memory management and cleanup
- Correctness verification comparing serial and parallel results
- Performance metrics for analysis

## License
This project is provided for educational purposes.
