//===============================================================================
// Name        : MatrixRotate.cpp
// Author      : Soumil Datta
// Version     : 1.0
// Description : CUDA program to rotate an NxN matrix by 90 degrees to the right
//===============================================================================

#include <iostream>
    using std::cout; using std::endl;
#include <cuda_runtime.h>

unsigned int dimension { 1u };

__global__ void transpose(float *matrix, unsigned int dimension);
__global__ void reverse(float *matrix, unsigned int dimension);

bool CPUSolveCheck(float *originalMatrix, float *solvedMatrix);
void printMatrix(const float *matrix);

int main(int argc, char* argv[]) {
    if(argc != 2) {
        cout << "Error: Enter dimension as argument" << endl;
        exit(EXIT_FAILURE);
    }

    cout << "Rotating matrix of dimension " << argv[1] << endl;

    dimension = atoi(argv[1]);
    const size_t size { (dimension * dimension) * sizeof(float) };
    float *h_matrix { (float *)malloc(size) };

    if(h_matrix == nullptr) {
        cout << "Host matrix memory allocation unsuccessful" << endl;
        exit(EXIT_FAILURE);
    }

    // Fill matrix
    for(auto i { 0u }; i < dimension * dimension; ++i) {
        h_matrix[i] = rand()/(float)RAND_MAX;
    }

    // Copy array to be used while checking output
    float *h_matrix_copy { (float *)malloc(size) };
    memcpy(h_matrix_copy, h_matrix, size);

    float *d_matrix = nullptr;
    cudaMalloc((void **)&d_matrix, size);
    cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);

    const dim3 threadsPerBlock(16, 16);
    const dim3 blocksPerGrid((dimension / threadsPerBlock.x) + 1, (dimension / threadsPerBlock.y) + 1);

    transpose<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, dimension);
    cudaDeviceSynchronize();

    reverse<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, dimension);
    cudaDeviceSynchronize();

   cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost);

   cudaFree(d_matrix);

    cout << endl << endl;
    if(CPUSolveCheck(h_matrix_copy, h_matrix)) cout << "GPU Rotate Successful" << endl;
    else cout << "GPU Rotate Unsuccessful" << endl;
    cout << "Program complete" << endl;

    free(h_matrix);
    free(h_matrix_copy);
    return 0;
}

__global__ void transpose(float *matrix, unsigned int dimension) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < (dimension) && j < (dimension) && j > i) {

        const auto index { dimension * i + j };
        const auto invIndex { dimension * j + i };

        const auto temp { matrix[index] };
        matrix[index] = matrix[invIndex];
        matrix[invIndex] = temp;
    }
}

__global__ void reverse(float *matrix, unsigned int dimension) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < (dimension) && j < (dimension / 2)) {
        const auto index { dimension * i + j };
        const auto revIndex { (dimension * i) + dimension - 1 - j };

        const auto temp { matrix[index] };
        matrix[index] = matrix[revIndex];
        matrix[revIndex] = temp;
    }
}

bool CPUSolveCheck(float *originalMatrix, float *solvedMatrix) {

    // Solve CPU-side with OriginalMatrix
    for(auto i { 0u }; i < dimension; ++i) {
        for(auto j { i + 1 }; j < dimension; ++j) {
            const auto index { dimension * i + j };
            const auto invIndex { dimension * j + i };

            const auto temp { originalMatrix[index] };
            originalMatrix[index] = originalMatrix[invIndex];
            originalMatrix[invIndex] = temp;
        }
    }
    for(auto i { 0u }; i < dimension; ++i) {
        for(auto j { 0u }; j < dimension / 2; ++j) {
            const auto index { dimension * i + j };
            const auto revIndex { (dimension * i) + dimension - 1 - j };

            const auto temp { originalMatrix[index] };
            originalMatrix[index] = originalMatrix[revIndex];
            originalMatrix[revIndex] = temp;
        }
    }

    // Check GPU output vs cpu output
    for(auto i { 0u }; i < dimension; ++i) {
        for(auto j { 0u }; j < dimension / 2; ++j) {
            const auto index { dimension * i + j };
            if(!(originalMatrix[index] == solvedMatrix[index])) return false;
        }
    }

    return true;
}

// Utility
void printMatrix(const float *matrix) {
    for(int i = 0; i < dimension * dimension; ++i) {
        if(i != 0 && i % dimension == 0) cout << endl;
        cout << matrix[i] << "\t";
    }
    cout << endl;
}
