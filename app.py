#!/usr/bin/env python3
"""
Flask Web Application for Parallel Matrix Multiplication
Provides a GUI frontend to interact with the C program
"""

from flask import Flask, render_template, request, jsonify
import subprocess
import os
import sys

app = Flask(__name__)

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_matrix_multiplication():
    """
    API endpoint to run matrix multiplication
    Receives: matrix_size (int), num_threads (int)
    Returns: Execution results as JSON
    """
    try:
        data = request.json
        matrix_size = int(data.get('matrix_size', 1000))
        num_threads = int(data.get('num_threads', 4))
        
        # Validate inputs
        if matrix_size < 10 or matrix_size > 5000:
            return jsonify({'error': 'Matrix size must be between 10 and 5000'}), 400
        if num_threads < 1 or num_threads > 32:
            return jsonify({'error': 'Number of threads must be between 1 and 32'}), 400
        
        # Create a modified C program with custom parameters
        c_code = generate_c_program(matrix_size, num_threads)
        
        # Write to temporary file
        temp_file = '/tmp/matrix_temp.c'
        with open(temp_file, 'w') as f:
            f.write(c_code)
        
        # Compile the program
        compile_result = subprocess.run(
            ['gcc', temp_file, '-fopenmp', '-o', '/tmp/matrix_temp'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if compile_result.returncode != 0:
            return jsonify({
                'error': 'Compilation failed',
                'details': compile_result.stderr
            }), 500
        
        # Run the compiled program
        run_result = subprocess.run(
            ['/tmp/matrix_temp'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if run_result.returncode != 0:
            return jsonify({
                'error': 'Program execution failed',
                'details': run_result.stderr
            }), 500
        
        # Parse the output
        output = run_result.stdout
        results = parse_output(output, matrix_size, num_threads)
        
        return jsonify({
            'success': True,
            'results': results,
            'raw_output': output
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Execution timeout - matrix too large'}), 500
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

def generate_c_program(matrix_size, num_threads):
    """Generate customized C program with specific parameters"""
    return f'''
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void serialMatrixMultiplication(int** A, int** B, int** C, int N) {{
    for (int i = 0; i < N; i++) {{
        for (int j = 0; j < N; j++) {{
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {{
                C[i][j] += A[i][k] * B[k][j];
            }}
        }}
    }}
}}

void parallelMatrixMultiplication(int** A, int** B, int** C, int N) {{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {{
        for (int j = 0; j < N; j++) {{
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {{
                C[i][j] += A[i][k] * B[k][j];
            }}
        }}
    }}
}}

int** allocateMatrix(int N) {{
    int** matrix = (int**)malloc(N * sizeof(int*));
    for (int i = 0; i < N; i++) {{
        matrix[i] = (int*)malloc(N * sizeof(int));
    }}
    return matrix;
}}

void initializeMatrix(int** matrix, int N, int seed) {{
    srand(seed);
    for (int i = 0; i < N; i++) {{
        for (int j = 0; j < N; j++) {{
            matrix[i][j] = rand() % 100;
        }}
    }}
}}

int compareMatrices(int** matrix1, int** matrix2, int N) {{
    for (int i = 0; i < N; i++) {{
        for (int j = 0; j < N; j++) {{
            if (matrix1[i][j] != matrix2[i][j]) {{
                return 0;
            }}
        }}
    }}
    return 1;
}}

void freeMatrix(int** matrix, int N) {{
    for (int i = 0; i < N; i++) {{
        free(matrix[i]);
    }}
    free(matrix);
}}

int main() {{
    int N = {matrix_size};
    int num_threads = {num_threads};
    
    int** A = allocateMatrix(N);
    int** B = allocateMatrix(N);
    int** C_serial = allocateMatrix(N);
    int** C_parallel = allocateMatrix(N);

    initializeMatrix(A, N, 42);
    initializeMatrix(B, N, 24);

    // Serial execution
    double serial_start = omp_get_wtime();
    serialMatrixMultiplication(A, B, C_serial, N);
    double serial_end = omp_get_wtime();
    double serial_time = serial_end - serial_start;

    // Parallel execution
    omp_set_num_threads(num_threads);
    double parallel_start = omp_get_wtime();
    parallelMatrixMultiplication(A, B, C_parallel, N);
    double parallel_end = omp_get_wtime();
    double parallel_time = parallel_end - parallel_start;

    // Results
    printf("MATRIX_SIZE:%d\\n", N);
    printf("NUM_THREADS:%d\\n", num_threads);
    printf("SERIAL_TIME:%.6f\\n", serial_time);
    printf("PARALLEL_TIME:%.6f\\n", parallel_time);
    printf("SPEEDUP:%.4f\\n", serial_time / parallel_time);
    printf("EFFICIENCY:%.2f\\n", (serial_time / parallel_time) / num_threads * 100);
    
    if (compareMatrices(C_serial, C_parallel, N)) {{
        printf("VERIFICATION:SUCCESS\\n");
    }} else {{
        printf("VERIFICATION:FAILED\\n");
    }}

    freeMatrix(A, N);
    freeMatrix(B, N);
    freeMatrix(C_serial, N);
    freeMatrix(C_parallel, N);

    return 0;
}}
'''

def parse_output(output, matrix_size, num_threads):
    """Parse the program output and extract results"""
    results = {
        'matrix_size': matrix_size,
        'num_threads': num_threads,
        'serial_time': 0,
        'parallel_time': 0,
        'speedup': 0,
        'efficiency': 0,
        'verification': 'UNKNOWN'
    }
    
    lines = output.split('\n')
    for line in lines:
        if 'SERIAL_TIME:' in line:
            results['serial_time'] = float(line.split(':')[1])
        elif 'PARALLEL_TIME:' in line:
            results['parallel_time'] = float(line.split(':')[1])
        elif 'SPEEDUP:' in line:
            results['speedup'] = float(line.split(':')[1])
        elif 'EFFICIENCY:' in line:
            results['efficiency'] = float(line.split(':')[1])
        elif 'VERIFICATION:' in line:
            results['verification'] = line.split(':')[1].strip()
    
    return results

if __name__ == '__main__':
    print("Starting Flask web server...")
    print("Open your browser and navigate to http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
