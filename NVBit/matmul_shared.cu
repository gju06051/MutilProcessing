#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void matMul_kernel_shared(float *_A, float *_B, float *_C, int rows, int cols)
{
    int row = threadIdx.y;
    int col = threadIdx.x;
    int index = row * blockDim.x + col;

    __shared__ float sA[32][32];
    __shared__ float sB[32][32];

    float sum = 0;
    for (int tile = 0; tile < cols / 32; ++tile) {
        sA[row][col] = _A[row * cols + tile * 32 + col];
        sB[row][col] = _B[(tile * 32 + row) * cols + col];
        __syncthreads();

        for (int k = 0; k < 32; ++k)
            sum += sA[row][k] * sB[k][col];
        __syncthreads();
    }

    _C[index] = sum;
}

int main()
{
    int rows = 32;
    int cols = 32;
    int matSize = rows * cols;

    float *A, *B, *C;
    float *dA, *dB, *dC;

    A = (float *)malloc(sizeof(float) * matSize);
    B = (float *)malloc(sizeof(float) * matSize);
    C = (float *)malloc(sizeof(float) * matSize);

    for (int i = 0; i < matSize; ++i) {
        A[i] = static_cast<float>(i % rows);
        B[i] = static_cast<float>(i % cols);
        C[i] = 0.0f;
    }

    cudaMalloc((void **)&dA, sizeof(float) * matSize);
    cudaMalloc((void **)&dB, sizeof(float) * matSize);
    cudaMalloc((void **)&dC, sizeof(float) * matSize);

    cudaMemcpy(dA, A, sizeof(float) * matSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float) * matSize, cudaMemcpyHostToDevice);

    dim3 gridDim(1, 1);
    dim3 blockDim(32, 32);

    matMul_kernel_shared<<<gridDim, blockDim>>>(dA, dB, dC, rows, cols);

    cudaMemcpy(C, dC, sizeof(float) * matSize, cudaMemcpyDeviceToHost);

    // 결과 출력 또는 확인

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(A);
    free(B);
    free(C);

    return 0;
}
