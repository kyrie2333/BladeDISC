#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// #define BLOCK_X 4
// #define BLOCK_Y 256  // reduce axis
// #define TILE_SIZE (M / BLOCK_Y)

#define WARMUPS 100
#define ITERS 200

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

// elapsed time in millisecond
#define CpuElapse(base, start) \
  base += ((double)(clock() - start)) * 1000 / CLOCKS_PER_SEC;

#define GpuElapse(start, stop, elapsed, total) \
  cudaEventRecord(stop, 0);                    \
  cudaEventSynchronize(stop);                  \
  cudaEventElapsedTime(&elapsed, start, stop); \
  total += elapsed;

// check result
// | (real - expected) / expected |
void check_result(float* host_ref, float* gpu_ref) {
  double epsilon = 1.0E-5;
  bool match = 1;
  for (int i = 0; i < 8192; i++) {
    // gpu_ref[i] /= (WARMUPS + ITERS);
    if (abs((host_ref[i] - gpu_ref[i]) / host_ref[i]) >
        epsilon) {
      match = 0;
      printf("Arrays do not match!\n");
      printf("host %5.8f gpu %5.8f at index %d, error %5.8f\n", host_ref[i],
             gpu_ref[i], i, abs((host_ref[i] - gpu_ref[i]) / host_ref[i]));
      break;
    }
  }
  if (match) printf("Results match!\n");
}

// column reduction on host
void column_reduce_host(float* matrix, float* result) {
  for (int col = 0; col < 8192; col++) {
    float accum = 0.0;
    for (int row = 0; row < 8192; row++) {
      accum += matrix[row * 8192 + col];
    }
    result[col] = accum;
  }
}
// colume reduction for a mtrix, 1D grid and 2D block
__global__ void colreduct_disc(float* data_in, float* data_out, int M, int N) {
  // int tid = threadIdx.x + blockDim.x * threadIdx.y;
  // 
  __shared__ float shm[32][8];
  // printf("blockIdx.x %d blockIdx.y %d threadIdx.x %d threadIdx.y %d\n",blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
  int block_x = blockIdx.x % ((N + blockDim.x - 1) / blockDim.x);
  // printf("block_x %d\n", block_x);
  int block_y = blockIdx.x / ((N + blockDim.x - 1) / blockDim.x);
  // printf("block_y %d\n", block_y);
  int row_index = block_y * blockDim.y + threadIdx.y;
  int col_index = block_x * blockDim.x + threadIdx.x;
  // printf("row_index %d col_index %d\n", row_index, col_index);
  // int is_valid = (row_index < M) && (col_index < N);

  if (row_index < M && col_index < N) {
    shm[threadIdx.y][threadIdx.x] = data_in[row_index * N + col_index];
    // printf("1 shm %f\n", shm[threadIdx.y][threadIdx.x]);
  }
    __syncthreads();
  // printf("blockIdx.x %d blockIdx.y %d threadIdx.x %d threadIdx.y %d shm
  // %f\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
  // shm[blockIdx.y][blockIdx.x]);

  for (int stride = 32 / 2; stride > 1; stride /= 2) {
    __syncthreads();
    if (threadIdx.y < stride && row_index + stride < M) {
      shm[threadIdx.y][threadIdx.x] += shm[stride + threadIdx.y][threadIdx.x];
  // printf("2 shm %f\n", shm[threadIdx.y][threadIdx.x]);
    }
  }

  __syncthreads();

  if (threadIdx.y == 0) {
    if (row_index < M && col_index < N) {
      float partial_result =
          shm[threadIdx.y][threadIdx.x] + shm[threadIdx.y + 1][threadIdx.x];
      atomicAdd(&data_out[col_index], partial_result);
    }
  }
}

int main() {
  // allocate memory for matrix and result on host
  int M = 8192;
  int N = 8192;
  float* matrix = (float*)malloc(M * N * sizeof(float));
  float* result = (float*)malloc(N * sizeof(float));
  float* h_result = (float*)malloc(N * sizeof(float));

  // initialize matrix
  for (int i = 0; i < M * N; i++) {
      matrix[i] = (float)rand() / RAND_MAX;
    // matrix[i] = 1.0f;
  }

  // allocate memory for matrix and result on device
  float *d_matrix, *d_result;
  CHECK(cudaMalloc((void**)&d_matrix, M * N * sizeof(float)));
  CHECK(cudaMalloc((void**)&d_result, N * sizeof(float)));
  CHECK(cudaMemcpy(d_matrix, matrix, M * N * sizeof(float),
                   cudaMemcpyHostToDevice));

  dim3 block(8, 32);
  int grid_y = (M + block.y - 1) / block.y;
  dim3 grid((N + block.x - 1) / block.x * grid_y, 1);
  printf("grid.x %d grid.y %d\n", (N + block.x - 1) / block.x, grid_y);
  printf("block.x %d block.y %d\n", block.x, block.y);

  // for (int i = 0; i < WARMUPS; ++i)
  // colreduct_disc<<<grid, block>>>(d_matrix, d_result, M, N);

  float total = 0.;
  float elapsed = 0.;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // for (int i = 0; i < ITERS; ++i)
  colreduct_disc<<<grid, block>>>(d_matrix, d_result, M, N);

  GpuElapse(start, stop, elapsed, total);
  printf("column_reduce Time elapsed %f us\n", elapsed / ITERS * 1000);

  // CHECK(
  cudaMemcpy(result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);
  CHECK(cudaDeviceReset());

  // column reduction on host
  column_reduce_host(matrix, h_result);

  // check result
  check_result(h_result, result);

  free(matrix);
  free(result);
  free(h_result);
  cudaFree(d_matrix);
  cudaFree(d_result);

  return 0;
}
