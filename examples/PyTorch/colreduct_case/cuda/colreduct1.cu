#include <cuda_runtime.h>
// #include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// #define M 8192
// #define N 2560
// #define BLOCK_X 256
// #define TILE_SIZE 128

#define WARMUPS 100
#define ITERS 100

#define CHECK(call)                                          \
  {                                                          \
    const cudaError_t error = call;                          \
    if (error != cudaSuccess) {                              \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
      fprintf(stderr, "code: %d, reason: %s\n", error,       \
              cudaGetErrorString(error));                    \
      exit(1);                                               \
    }                                                        \
  }

#define CpuElapse(base, start) \
  base += ((double)(clock() - start)) * 1000 / CLOCKS_PER_SEC;

#define GpuElapse(start, stop, elapsed, total) \
  cudaEventRecord(stop, 0);                    \
  cudaEventSynchronize(stop);                  \
  cudaEventElapsedTime(&elapsed, start, stop); \
  total += elapsed;

// check result
// | (real - expected) / expected |
void check_result(float* host_ref, float* gpu_ref, int N) {
  double epsilon = 1.0E-4;
  bool match = 1;
  for (int i = 0; i < N; i++) {
  gpu_ref[i] /= 200;

    if (abs((host_ref[i] - gpu_ref[i]) / host_ref[i]) > epsilon) {
      match = 0;
      printf("wrong!\n");
      printf("host %5.8f gpu %5.8f at index %d, error %5.8f\n", host_ref[i],
             gpu_ref[i], i, abs((host_ref[i] - gpu_ref[i]) / host_ref[i]));
      break;
    }
  }
  if (match) printf("Match\n");
}
// column reduction on host
void column_reduce_host(float* matrix, float* result, int M, int N) {
  for (int col = 0; col < N; col++) {
    float accum = 0.0;
    for (int row = 0; row < M; row++) {
      accum += matrix[row * N + col];
    }
    result[col] = accum;
  }
}

// colume reduction for a mtrix, 1D grid and 1D block
// each thread will reduce a column in a block
// each accum of a block is reduce by atomicAdd
__global__ void column_reduce(float* data_in, float* data_out, int M, int N, int tile_size) {
  // get the column index
  // int col = blockIdx.x * blockDim.x + threadIdx.x;
  int block_x = blockIdx.x % ((N + blockDim.x - 1) / blockDim.x);
  int block_y = blockIdx.x / ((N + blockDim.x - 1) / blockDim.x);
  int col = block_x * blockDim.x + threadIdx.x;

  float accum = 0.0;

  for (int i = 0; i < tile_size; i++) {
    int row = block_y * tile_size + i;
    if (row < M) accum += data_in[row * N + col];
  }

  atomicAdd(&data_out[col], accum);
}

int main(int argc, char* argv[]) {

  if(argc != 5){
    // printf("4 arguments required: M N BLOCK_X TILE_SIZE\n");
    exit(1);
  }
  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int BLOCK_X = atoi(argv[3]);
  int TILE_SIZE = atoi(argv[4]);

  // allocate memory for matrix and result on host
  float* matrix = (float*)malloc(M * N * sizeof(float));
  float* result = (float*)malloc(N * sizeof(float));    // gpu result
  float* h_result = (float*)malloc(N * sizeof(float));  // validation

  // initialize matrix
  for (int i = 0; i < M * N; i++) {
    matrix[i] = (float)rand() / RAND_MAX;
    // matrix[i] = 1.0;
  }

  // allocate memory for matrix and result on device
  float *d_matrix, *d_result;
  CHECK(cudaMalloc((void**)&d_matrix, M * N * sizeof(float)));
  CHECK(cudaMalloc((void**)&d_result, N * sizeof(float)));

  CHECK(cudaMemcpy(d_matrix, matrix, M * N * sizeof(float),
                   cudaMemcpyHostToDevice));

  dim3 block(BLOCK_X, 1);
  int grid_y = (M + TILE_SIZE - 1) / TILE_SIZE;
  dim3 grid((N + BLOCK_X - 1) / BLOCK_X * grid_y, 1);

  for (int i = 0; i < WARMUPS; ++i)
    column_reduce<<<grid, block>>>(d_matrix, d_result, M, N, TILE_SIZE);

  float total = 0.;
  float elapsed = 0.;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  for (int i = 0; i < ITERS; ++i)
    column_reduce<<<grid, block>>>(d_matrix, d_result, M, N, TILE_SIZE);

  GpuElapse(start, stop, elapsed, total);
  printf("Time %f us\n", elapsed / ITERS * 1000);

  CHECK(cudaMemcpy(result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaDeviceReset());

  // column reduction on host
  column_reduce_host(matrix, h_result, M, N);

  // check result
  check_result(h_result, result, N);

  free(matrix);
  free(result);
  free(h_result);
  cudaFree(d_matrix);
  cudaFree(d_result);

  return 0;
}