#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define M 8192
#define N 2560
#define BLOCK_X 4
#define BLOCK_Y 256  // reduce axis
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
  for (int i = 0; i < N; i++) {
    if (abs((host_ref[i] - gpu_ref[i] / (WARMUPS + ITERS)) / host_ref[i]) >
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
__global__ void column_reduce(float* data_in, float* data_out) {
  // LOCK TILE IMPL: tile_w*tile_h threads load tile_w*tile_h elements
  //   from gmem to SHM(tile_w vectorizes load), do reduction in SHM and
  //   atomic to gmem
  var_tile_w = 8 var_tile_h = 32 num_block_col = ceil(var_cols() / var_tile_w);
  num_block_row = ceil(var_rows() / var_tile_h);
  for (m = 0; m < num_block_col * num_block_row; ++m) {
    for (n = 0; n < var_threads; ++n) {
      local_row_index = n / var_tile_w;
      local_col_index = n % var_tile_w;
      block_row_index = m / num_block_col;
      block_col_index = m % num_block_col;
      row_index = block_row_index * var_tile_h + local_row_index;
      col_index = block_col_index * var_tile_w + local_col_index;
      is_valid = (row_index < var_rows()) && (col_index < var_cols());
      if (is_valid) {
        shm[n] = sum + global[row_index, col_index];
      } else {
        shm[n] = sum;
      }
      for (int stride = var_tile_h / 2; stride > 1; stride /= 2) {
        __syncthreads();
        if (local_row_index < stride && row_index + stride < var_rows) {
          shm[n] += shm[stride * var_tile_w + n];
        }
      }
      __syncthreads();
      if (local_row_index == 0) {
        if (is_valid) {
          partial_result = shm[n] + shm[var_tile_w + n];
          atomicAdd(&global[col_index], partial_result);
        }
      }
    }
  }
}

int main() {
  // allocate memory for matrix and result on host
  float* matrix = (float*)malloc(M * N * sizeof(float));
  float* result = (float*)malloc(N * sizeof(float));
  float* h_result = (float*)malloc(N * sizeof(float));

  // initialize matrix
  for (int i = 0; i < M * N; i++) {
    //   matrix[i] = (float)rand() / RAND_MAX;
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
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

  for (int i = 0; i < WARMUPS; ++i)
    column_reduce<<<grid, block>>>(d_matrix, d_result);

  float total = 0.;
  float elapsed = 0.;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  for (int i = 0; i < ITERS; ++i)
    column_reduce<<<grid, block>>>(d_matrix, d_result);

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
