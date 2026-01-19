/*
 * Parallel Matrix Multiplication using OpenMP
 * Author: HPC Developer
 * Description: This program implements both serial and parallel matrix multiplication
 *              using OpenMP for parallel processing in a GitHub Codespaces environment.
 * Compilation: gcc matrix_openmp.c -fopenmp -o matrix
 * Execution: ./matrix
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

/*
 * SERIAL MATRIX MULTIPLICATION
 * Description: Performs matrix multiplication C = A * B without parallelization
 * Parameters:
 *   - A: Pointer to first input matrix (N x N)
 *   - B: Pointer to second input matrix (N x N)
 *   - C: Pointer to output matrix (N x N)
 *   - N: Size of the square matrices
 * Algorithm:
 *   For each element C[i][j]:
 *   1. Initialize C[i][j] = 0
 *   2. Iterate through all k values from 0 to N-1
 *   3. Accumulate product: C[i][j] += A[i][k] * B[k][j]
 * Time Complexity: O(N^3)
 * Space Complexity: O(1) (excluding input and output matrices)
 */
void serialMatrixMultiplication(int** A, int** B, int** C, int N) {
    // Outer loop: iterate through rows of matrix A
    for (int i = 0; i < N; i++) {
        // Middle loop: iterate through columns of matrix B
        for (int j = 0; j < N; j++) {
            // Initialize result element
            C[i][j] = 0;
            
            // Inner loop: compute dot product of row i of A and column j of B
            for (int k = 0; k < N; k++) {
                // Multiply and accumulate
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

/*
 * PARALLEL MATRIX MULTIPLICATION
 * Description: Performs matrix multiplication C = A * B with OpenMP parallelization
 * Parameters:
 *   - A: Pointer to first input matrix (N x N)
 *   - B: Pointer to second input matrix (N x N)
 *   - C: Pointer to output matrix (N x N)
 *   - N: Size of the square matrices
 * OpenMP Directives:
 *   - #pragma omp parallel for: Distributes iterations of outer loop across threads
 *   - Shared variables: A, B, C, N (read/write accessible to all threads)
 *   - Private variables: i, j, k (each thread has its own copy)
 * Speedup: Theoretical speedup is approximately P (number of processors),
 *          practical speedup depends on system load and thread efficiency
 * Time Complexity: O(N^3 / P) where P is the number of threads
 */
void parallelMatrixMultiplication(int** A, int** B, int** C, int N) {
    /*
     * #pragma omp parallel for
     * - Parallelizes the outer loop (i loop)
     * - Each thread processes different rows of the result matrix
     * - Since each iteration (row) is independent, no race conditions occur
     * - The j and k loops execute sequentially within each thread
     * - Shared: A, B, C, N are shared by all threads
     * - Private: i, j, k are private to each thread (automatically)
     */
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // Initialize result element (private to each thread)
            C[i][j] = 0;
            
            // Inner loop: compute dot product (sequential within each thread)
            for (int k = 0; k < N; k++) {
                // Multiply and accumulate (no race condition as C[i][j] is unique per iteration)
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

/*
 * UTILITY FUNCTION: Allocate dynamic 2D matrix
 * Returns: Pointer to dynamically allocated N x N matrix
 */
int** allocateMatrix(int N) {
    int** matrix = (int**)malloc(N * sizeof(int*));
    for (int i = 0; i < N; i++) {
        matrix[i] = (int*)malloc(N * sizeof(int));
    }
    return matrix;
}

/*
 * UTILITY FUNCTION: Initialize matrix with sample values
 * Parameters: matrix (pointer to matrix), N (size), seed (for random values)
 */
void initializeMatrix(int** matrix, int N, int seed) {
    srand(seed);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = rand() % 100;  // Random values between 0 and 99
        }
    }
}

/*
 * UTILITY FUNCTION: Compare two matrices for correctness
 * Returns: 1 if matrices are identical, 0 otherwise
 */
int compareMatrices(int** matrix1, int** matrix2, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (matrix1[i][j] != matrix2[i][j]) {
                printf("Mismatch at position [%d][%d]: %d vs %d\n",
                       i, j, matrix1[i][j], matrix2[i][j]);
                return 0;
            }
        }
    }
    return 1;
}

/*
 * UTILITY FUNCTION: Free dynamically allocated matrix
 */
void freeMatrix(int** matrix, int N) {
    for (int i = 0; i < N; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

/*
 * MAIN PROGRAM
 */
int main() {
    printf("========================================================\n");
    printf("Parallel Matrix Multiplication using OpenMP\n");
    printf("========================================================\n\n");

    // Matrix size (NxN)
    int N = 1000;
    printf("Matrix size: %d x %d\n", N, N);
    printf("Total elements per matrix: %d\n\n", N * N);

    // Allocate memory for matrices
    int** A = allocateMatrix(N);
    int** B = allocateMatrix(N);
    int** C_serial = allocateMatrix(N);
    int** C_parallel = allocateMatrix(N);

    // Initialize input matrices with sample values
    printf("Initializing matrices...\n");
    initializeMatrix(A, N, 42);
    initializeMatrix(B, N, 24);
    printf("Matrices initialized successfully.\n\n");

    // ========== SERIAL EXECUTION ==========
    printf("========== SERIAL EXECUTION ==========\n");
    double serial_start = omp_get_wtime();
    serialMatrixMultiplication(A, B, C_serial, N);
    double serial_end = omp_get_wtime();
    double serial_time = serial_end - serial_start;
    printf("Serial execution time: %.6f seconds\n\n", serial_time);

    // ========== PARALLEL EXECUTION ==========
    printf("========== PARALLEL EXECUTION ==========\n");
    
    // Set number of threads
    int num_threads = 4;  // Set to desired number of threads
    omp_set_num_threads(num_threads);
    
    double parallel_start = omp_get_wtime();
    parallelMatrixMultiplication(A, B, C_parallel, N);
    double parallel_end = omp_get_wtime();
    double parallel_time = parallel_end - parallel_start;
    
    printf("Parallel execution time: %.6f seconds\n");
    printf("Number of threads used: %d\n", num_threads);
    printf("(Actual threads may vary based on system configuration)\n\n");

    // ========== PERFORMANCE ANALYSIS ==========
    printf("========== PERFORMANCE ANALYSIS ==========\n");
    double speedup = serial_time / parallel_time;
    double efficiency = (speedup / num_threads) * 100.0;
    
    printf("Speedup: %.4f x\n", speedup);
    printf("Efficiency: %.2f %%\n", efficiency);
    printf("(Ideal speedup for %d threads: %.2f x)\n\n", num_threads, (double)num_threads);

    // ========== CORRECTNESS VERIFICATION ==========
    printf("========== CORRECTNESS VERIFICATION ==========\n");
    printf("Comparing serial and parallel results...\n");
    
    if (compareMatrices(C_serial, C_parallel, N)) {
        printf("✓ SUCCESS: Serial and parallel results match!\n");
        printf("✓ Matrix multiplication is correct.\n\n");
    } else {
        printf("✗ ERROR: Results do not match!\n");
        printf("✗ There is a problem with the parallel implementation.\n\n");
        freeMatrix(A, N);
        freeMatrix(B, N);
        freeMatrix(C_serial, N);
        freeMatrix(C_parallel, N);
        return 1;
    }

    // ========== CLEANUP ==========
    printf("========== CLEANUP ==========\n");
    printf("Freeing dynamically allocated memory...\n");
    freeMatrix(A, N);
    freeMatrix(B, N);
    freeMatrix(C_serial, N);
    freeMatrix(C_parallel, N);
    printf("Memory freed successfully.\n\n");

    printf("========================================================\n");
    printf("Program completed successfully!\n");
    printf("========================================================\n");

    return 0;
}
