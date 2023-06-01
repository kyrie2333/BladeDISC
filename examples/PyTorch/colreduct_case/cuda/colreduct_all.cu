#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// #define M 8192
// #define N 2560
// #define BLOCK_X 40
// #define BLOCK_Y 32
// #define TILE_SIZE 16

// #define WARMUPS 100
// #define ITERS 100

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

#define GpuElapse(start, stop, elapsed) \
  cudaEventRecord(stop, 0);             \
  cudaEventSynchronize(stop);           \
  cudaEventElapsedTime(&elapsed, start, stop);
// total += elapsed;

__global__ void colreduct1_256x32(float* data_in, float* data_out, int M, int N,
                                  int tile_size) {
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

__global__ void colreduct1_512x32(float* data_in, float* data_out, int M, int N,
                                  int tile_size) {
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

__global__ void colreduct2_32x8x64(float* data_in, float* data_out, int M,
                                   int N, int tile_size) {
  __shared__ float sdata[32][8];

  int block_x = blockIdx.x % ((N + blockDim.x - 1) / blockDim.x);
  int block_y = blockIdx.x / ((N + blockDim.x - 1) / blockDim.x);

  int col_g = block_x * blockDim.x + threadIdx.x;

  // local reduction
  float accum = 0.0f;
  for (int i = 0; i < tile_size; i++) {
    int row_g = i + threadIdx.y * tile_size + block_y * tile_size * blockDim.y;
    if (row_g < M && col_g < N)
      accum += data_in[row_g * N + col_g];
    else
      accum += 0.0f;
  }
  sdata[threadIdx.x][threadIdx.y] = accum;
  __syncthreads();

  for (int stride = blockDim.y / 2; stride > 0; stride >>= 1) {
    if (threadIdx.y < stride) {
      sdata[threadIdx.x][threadIdx.y] +=
          sdata[threadIdx.x][threadIdx.y + stride];
    }
    __syncthreads();
  }

  if (threadIdx.y == 0) {
    atomicAdd(&data_out[col_g], sdata[threadIdx.x][0]);
  }
}

__global__ void colreduct_disc(float* data_in, float* data_out, int M, int N) {
  __shared__ float shm[32][8];
  int block_x = blockIdx.x % ((N + blockDim.x - 1) / blockDim.x);
  int block_y = blockIdx.x / ((N + blockDim.x - 1) / blockDim.x);
  int row_index = block_y * blockDim.y + threadIdx.y;
  int col_index = block_x * blockDim.x + threadIdx.x;

  if (row_index < M && col_index < N) {
    shm[threadIdx.y][threadIdx.x] = data_in[row_index * N + col_index];
  }
  __syncthreads();

  for (int stride = 32 / 2; stride > 1; stride /= 2) {
    __syncthreads();
    if (threadIdx.y < stride && row_index + stride < M) {
      shm[threadIdx.y][threadIdx.x] += shm[stride + threadIdx.y][threadIdx.x];
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

// __global__ void column_reduce(float* data_in, float* data_out, int m,
//                                    int n, int tile_size) {
//   // LOCK TILE IMPL: tile_w*tile_h threads load tile_w*tile_h elements
//   //   from gmem to SHM(tile_w vectorizes load), do reduction in SHM and
//   //   atomic to gmem
//   var_tile_w = 8;
//   var_tile_h = 32;
//   num_block_col = ceil(var_cols() / var_tile_w);
//   num_block_row = ceil(var_rows() / var_tile_h);
//   for (m = 0; m < num_block_col * num_block_row; ++m) {
//     for (n = 0; n < var_threads; ++n) {
//       local_row_index = n / var_tile_w;
//       local_col_index = n % var_tile_w;
//       block_row_index = m / num_block_col;
//       block_col_index = m % num_block_col;
//       row_index = block_row_index * var_tile_h + local_row_index;
//       col_index = block_col_index * var_tile_w + local_col_index;
//       is_valid = (row_index < var_rows()) && (col_index < var_cols());
//       if (is_valid) {
//         shm[n] = sum + data_in[row_index, col_index];
//       } else {
//         shm[n] = sum;
//       }
//       for (int stride = var_tile_h / 2; stride > 1; stride /= 2) {
//         __syncthreads();
//         if (local_row_index < stride && row_index + stride < var_rows) {
//           shm[n] += shm[stride * var_tile_w + n];
//         }
//       }
//       __syncthreads();
//       if (local_row_index == 0) {
//         if (is_valid) {
//           partial_result = shm[n] + shm[var_tile_w + n];
//           atomicAdd(&global[col_index], partial_result);
//         }
//       }
//     }
//   }
// }
// void check_result(float* host_ref, float* gpu_ref) {
//   double epsilon = 1.0E-5;
//   bool match = 1;
//   for (int i = 0; i < 8192; i++) {
//     // gpu_ref[i] /= (WARMUPS + ITERS);
//     if (abs((host_ref[i] - gpu_ref[i]) / host_ref[i]) > epsilon) {
//       match = 0;
//       printf("Arrays do not match!\n");
//       printf("host %5.8f gpu %5.8f at index %d, error %5.8f\n", host_ref[i],
//              gpu_ref[i], i, abs((host_ref[i] - gpu_ref[i]) / host_ref[i]));
//       break;
//     }
//   }
//   if (match) printf("Results match!\n");
// }

// void column_reduce_host(float* matrix, float* result) {
//   for (int col = 0; col < 8192; col++) {
//     float accum = 0.0;
//     for (int row = 0; row < 512; row++) {
//       accum += matrix[row * 8192 + col];
//     }
//     result[col] = accum;
//   }
// }
int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("2 arguments required: M, N\n");
    exit(1);
  }

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);

  float* matrix = (float*)malloc(M * N * sizeof(float));
  float* result = (float*)malloc(N * sizeof(float));
  //   float* h_result = (float*)malloc(N * sizeof(float));

  // initialize matrix
  for (int i = 0; i < M * N; i++) {
    matrix[i] = (float)rand() / RAND_MAX;
    // matrix[i] = 1.0f;
  }

  // column reduction on host
//   column_reduce_host(matrix, h_result);

  float *d_matrix, *d_result;
  CHECK(cudaMalloc((void**)&d_matrix, M * N * sizeof(float)));
  CHECK(cudaMalloc((void**)&d_result, N * sizeof(float)));

    int TILE_SIZE = 32;
  dim3 block(256, 1);
  int grid_y = (M + TILE_SIZE - 1) / TILE_SIZE;
  dim3 grid((N + block.x - 1) / block.x * grid_y, 1);


  CHECK(cudaMemcpy(d_matrix, matrix, M * N * sizeof(float),
                   cudaMemcpyHostToDevice));
  cudaMemset(d_result, 0, N * sizeof(float));
  colreduct1_256x32<<<grid, block>>>(d_matrix, d_result, M, N, TILE_SIZE);
  CHECK(cudaMemcpy(result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaDeviceSynchronize());
  //   CHECK(cudaDeviceReset());
//   check_result(h_result, result);



  cudaMemcpy(d_matrix, matrix, M * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_result, 0, N * sizeof(float));
  TILE_SIZE = 32;
  block.x = 512;
  grid_y = (M + TILE_SIZE - 1) / TILE_SIZE;
  grid.x = (N + block.x - 1) / block.x * grid_y;
  colreduct1_512x32<<<grid, block>>>(d_matrix, d_result, M, N, TILE_SIZE);
  CHECK(
      cudaMemcpy(result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaDeviceSynchronize());
//   check_result(h_result, result);



  CHECK(cudaMemcpy(d_matrix, matrix, M * N * sizeof(float), cudaMemcpyHostToDevice));
  cudaMemset(d_result, 0, N * sizeof(float));
  TILE_SIZE = 64;
  block.x = 32;
  block.y = 8;
  grid_y = (M + (64 * 8) - 1) / (64 * 8);
  grid.x = (N + 32 - 1) / 32 * grid_y;
  colreduct2_32x8x64<<<grid, block>>>(d_matrix, d_result, M, N, TILE_SIZE);
  cudaMemcpy(result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);
  CHECK(cudaDeviceSynchronize());
//   check_result(h_result, result);



  CHECK(cudaMemcpy(d_matrix, matrix, M * N * sizeof(float),
                   cudaMemcpyHostToDevice));
  cudaMemset(d_result, 0, N * sizeof(float));
  block.x = 8;
  block.y = 32;
  grid_y = (M + block.y - 1) / block.y;
  grid.x = (N + block.x - 1) / block.x * grid_y;
  colreduct_disc<<<grid, block>>>(d_matrix, d_result, M, N);
  CHECK(cudaMemcpy(result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaDeviceSynchronize());


//   // check result
//   check_result(h_result, result);

  free(matrix);
  free(result);
//   free(h_result);
  cudaFree(d_matrix);
  cudaFree(d_result);

  return 0;
}
