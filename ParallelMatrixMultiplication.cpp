#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <ctime>


































}    }        }            }                C[i][j] += A[i][k] * B[k][j]; // Multiply and accumulate            for (int k = 0; k < N; k++) {            C[i][j] = 0; // Initialize the result matrix element        for (int j = 0; j < N; j++) {    for (int i = 0; i < N; i++) {    #pragma omp parallel for    // Parallelize the outer loop using OpenMPvoid parallelMatrixMultiplication(int** A, int** B, int** C, int N) {// Parallel matrix multiplication function}    }        }            }                C[i][j] += A[i][k] * B[k][j]; // Multiply and accumulate            for (int k = 0; k < N; k++) {            // Loop through each element of the row of A and column of B            C[i][j] = 0; // Initialize the result matrix element        for (int j = 0; j < N; j++) {        // Loop through each column of B    for (int i = 0; i < N; i++) {    // Loop through each row of Avoid serialMatrixMultiplication(int** A, int** B, int** C, int N) {// Serial matrix multiplication function// OpenMP directives are used to manage parallel execution and thread allocation.// The parallel algorithm uses OpenMP to parallelize the outer loop of the multiplication.// The serial algorithm multiplies two matrices without parallelization.// This program implements both serial and parallel matrix multiplication using OpenMP.
void serialMatrixMultiplication(int** A, int** B, int** C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void parallelMatrixMultiplication(int** A, int** B, int** C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    int N = 1000; // Size of the matrices
    int** A = new int*[N];
    int** B = new int*[N];
    int** C_serial = new int*[N];
    int** C_parallel = new int*[N];

    // Initialize matrices
    for (int i = 0; i < N; i++) {
        A[i] = new int[N];
        B[i] = new int[N];
        C_serial[i] = new int[N];
        C_parallel[i] = new int[N];
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }
    }

    // Serial multiplication
    double start_time = omp_get_wtime();
    serialMatrixMultiplication(A, B, C_serial, N);
    double serial_time = omp_get_wtime() - start_time;

    // Parallel multiplication
    omp_set_num_threads(4); // Set number of threads
    start_time = omp_get_wtime();
    parallelMatrixMultiplication(A, B, C_parallel, N);
    double parallel_time = omp_get_wtime() - start_time;

    // Display results
    std::cout << "Serial execution time: " << serial_time << " seconds" << std::endl;
    std::cout << "Parallel execution time: " << parallel_time << " seconds" << std::endl;
    std::cout << "Number of threads used: 4" << std::endl;
    std::cout << "Speedup: " << serial_time / parallel_time << std::endl;

    // Verify correctness
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (C_serial[i][j] != C_parallel[i][j]) {
                std::cout << "Error: Matrices do not match at position (" << i << ", " << j << ")" << std::endl;
                return 1;
            }
        }
    }

    std::cout << "Matrices are correct!" << std::endl;

    // Clean up
    for (int i = 0; i < N; i++) {
        delete[] A[i];
        delete[] B[i];
        delete[] C_serial[i];
        delete[] C_parallel[i];
    }
    delete[] A;
    delete[] B;
    delete[] C_serial;
    delete[] C_parallel;

    return 0;
}