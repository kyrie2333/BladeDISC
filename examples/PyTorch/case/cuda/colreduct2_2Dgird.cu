#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define M 8192
#define N 2560
// #define M 128
// #define N 32
#define BLOCK_X 40
#define BLOCK_Y 32
#define TILE_SIZE 16

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
  for (int i = 0; i < N; i++) {
  gpu_ref[i] /= 300;

    if (abs((host_ref[i] - gpu_ref[i]) / host_ref[i]) > epsilon) {
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
  for (int col = 0; col < N; col++) {
    float accum = 0.0;
    for (int row = 0; row < M; row++) {
      accum += matrix[row * N + col];
    }
    result[col] = accum;
  }
}

__device__ void warp_reduce(volatile float* sdata, int tid) {
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

// colume reduction for a mtrix, 1D grid and 2D block
__global__ void column_reduce_trans(float* data_in, float* data_out) {
  __shared__ float sdata[BLOCK_X][BLOCK_Y];

  int block_x = blockIdx.x % ((N + BLOCK_X - 1) / BLOCK_X);
  int block_y = blockIdx.x / ((N + BLOCK_X - 1) / BLOCK_X);

  int col_g = block_x * blockDim.x + threadIdx.x;
  // int col = blockIdx.x * blockDim.x + threadIdx.x;
  // int tid = threadIdx.x + threadIdx.y * blockDim.x;

  // local reduction
  float accum = 0.0f;
  for (int i = 0; i < TILE_SIZE; i++) {
    int row_g = i + threadIdx.y * TILE_SIZE + block_y * TILE_SIZE * BLOCK_Y;
    accum += data_in[row_g * N + col_g];
    // accum += data_in[(i + TILE_SIZE * threadIdx.y) * N + col];
  }
  sdata[threadIdx.x][threadIdx.y] = accum;
  __syncthreads();

  // printf("1 thread.x = %d, thread.y = %d, sdata[x][y] = %f\n", \
        threadIdx.x, threadIdx.y, sdata[threadIdx.x][threadIdx.y]);
  
  for (int stride = BLOCK_Y / 2; stride > 0; stride >>= 1) {
    if (threadIdx.y < stride) {
      sdata[threadIdx.x][threadIdx.y] +=
          sdata[threadIdx.x][threadIdx.y + stride];
    }
    __syncthreads();
  }

  // printf("2 blockIDx = %d, block_x= %d, block_y = %d,thread.x = %d, thread.y = %d, sdata[x][y] = %f\n", \
  //       blockIdx.x, block_x, block_y, threadIdx.x, threadIdx.y, sdata[threadIdx.x][threadIdx.y]);


  if (threadIdx.y == 0) {
    // data_out[col] = sdata[threadIdx.x][0];
    atomicAdd(&data_out[col_g], sdata[threadIdx.x][0]);
  }
}

int main() {
  // allocate memory for matrix and result on host
  float* matrix = (float*)malloc(M * N * sizeof(float));
  float* result = (float*)malloc(N * sizeof(float));
  float* h_result = (float*)malloc(N * sizeof(float));

  // initialize matrix
  for (int i = 0; i < M * N; i++) {
    // matrix[i] = (float)rand() / RAND_MAX;
    matrix[i] = 1.0f;
  }

  // allocate memory for matrix and result on device
  float *d_matrix, *d_result;
  CHECK(cudaMalloc((void**)&d_matrix, M * N * sizeof(float)));
  CHECK(cudaMalloc((void**)&d_result, N * sizeof(float)));
  CHECK(cudaMemcpy(d_matrix, matrix, M * N * sizeof(float),
                   cudaMemcpyHostToDevice));

  dim3 block(BLOCK_X, BLOCK_Y);
  //   dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  // dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) /
  // BLOCK_SIZE / TILE_SIZE);
  int grid_y = M / (TILE_SIZE * BLOCK_Y);
  dim3 grid((N + BLOCK_X - 1) / BLOCK_X * grid_y, 1);

  for (int i = 0; i < WARMUPS; ++i)
    column_reduce_trans<<<grid, block>>>(d_matrix, d_result);

  float total = 0.;
  float elapsed = 0.;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  for (int i = 0; i < ITERS; ++i)
    column_reduce_trans<<<grid, block>>>(d_matrix, d_result);

  GpuElapse(start, stop, elapsed, total);
  printf("column_reduce Time elapsed %f us\n", elapsed / ITERS * 1000);

  CHECK(
      cudaMemcpy(result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost));
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
