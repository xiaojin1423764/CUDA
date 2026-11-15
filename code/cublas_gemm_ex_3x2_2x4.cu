#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t error__ = (call);                                            \
    if (error__ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,           \
                   __LINE__, cudaGetErrorString(error__));                   \
      return EXIT_FAILURE;                                                    \
    }                                                                        \
  } while (0)

#define CUBLAS_CHECK(call)                                                   \
  do {                                                                       \
    cublasStatus_t status__ = (call);                                        \
    if (status__ != CUBLAS_STATUS_SUCCESS) {                                 \
      std::fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n", __FILE__,   \
                   __LINE__, static_cast<int>(status__));                    \
      return EXIT_FAILURE;                                                    \
    }                                                                        \
  } while (0)

int main() {
  constexpr int m = 3;  // A 的行数，也是 C 的行数
  constexpr int n = 4;  // B 的列数，也是 C 的列数
  constexpr int k = 2;  // A 的列数，也是 B 的行数

  // 输入保持通常的行主序，不预先重排：
  // A = [ 1  2 ]       B = [ 1  2  3  4 ]
  //     [ 3  4 ]           [ 5  6  7  8 ]
  //     [ 5  6 ]
  const float h_a[m * k] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const float h_b[k * n] = {1.0f, 2.0f, 3.0f, 4.0f,
                            5.0f, 6.0f, 7.0f, 8.0f};
  float h_c[m * n] = {};

  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;
  cublasHandle_t handle = nullptr;

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a), sizeof(h_a)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), sizeof(h_b)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c), sizeof(h_c)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a, sizeof(h_a), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, sizeof(h_b), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c, 0, sizeof(h_c)));

  CUBLAS_CHECK(cublasCreate(&handle));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  // 内存中的 A、B、C 都是行主序，但 cuBLAS 按列主序解释内存：
  // A_memory = A^T，B_memory = B^T，C_memory = C^T。
  // 因而 C^T = B^T * A^T，交换 A/B 后即可不拷贝、不转置地计算行主序结果。
  // C = alpha * A * B + beta * C。输入和输出都是 FP32，使用 FP32 累加。
  CUBLAS_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_b, CUDA_R_32F, n,
      d_a, CUDA_R_32F, k, &beta, d_c, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT));

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_c, d_c, sizeof(h_c), cudaMemcpyDeviceToHost));

  std::printf("A (3x2) * B (2x4) = C (3x4):\n");
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      // 输出内存是 C 的行主序，因此元素 (row, col) 的下标为 row * n + col。
      std::printf("%6.1f", h_c[row * n + col]);
    }
    std::putchar('\n');
  }

  // 期望结果为：
  // [ 11  14  17  20 ]
  // [ 23  30  37  44 ]
  // [ 35  46  57  68 ]
  const float expected[m * n] = {11.0f, 14.0f, 17.0f, 20.0f, 23.0f, 30.0f,
                                 37.0f, 44.0f, 35.0f, 46.0f, 57.0f, 68.0f};
  bool passed = true;
  for (int i = 0; i < m * n; ++i) {
    if (std::abs(h_c[i] - expected[i]) > 1.0e-5f) {
      passed = false;
      break;
    }
  }
  std::printf("verification: %s\n", passed ? "PASSED" : "FAILED");

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_a));
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
