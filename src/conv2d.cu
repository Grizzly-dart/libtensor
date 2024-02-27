#include <cuda_runtime.h>

struct {
  uint32_t x;
  uint32_t y;
} Dim2;

typedef enum {
  CONSTANT,
  CIRCULAR,
  REFLECT,
  REPLICATION
} PaddingMode;

template <typename T>
T lookupConstantPadding(T* data, Dim2 size, Dim2 padding, T constant, uint64_t x, uint64_t y) {
  if (x < padding.x || y < padding.y || x >= (size.x + padding.x) || y >= (size.y + padding.y)) {
    return constant;
  } else {
    return data[(y - padding.y) * size.x + (x - padding.x)];
  }
}

template <typename T>
T lookupCircularPadding(T* data, Dim2 size, Dim2 padding, T constant, uint64_t x, uint64_t y) {
  if (x < padding.x) {
    x += size.x;
  } else if (x >= (size.x + padding.x)) {
    x -= size.x;
  }
  if (y < padding.y) {
    y += size.y;
  } else if (y >= (size.y + padding.y)) {
    y -= size.y;
  }
  return data[(y - padding.y) * size.x + (x - padding.x)];
}

template <typename T>
T lookupReflectPadding(T* data, Dim2 size, Dim2 padding, T constant, uint64_t x, uint64_t y) {
  if (x < padding.x) {
    x = padding.x - x;
  } else if (x >= (size.x + padding.x)) {
    x = (size.x + padding.x) - (x - size.x - padding.x) - 1;
  }
  if (y < padding.y) {
    y = padding.y - y;
  } else if (y >= (size.y + padding.y)) {
    y = (size.y + padding.y) - (y - size.y - padding.y) - 1;
  }
  return data[(y - padding.y) * size.x + (x - padding.x)];
}

template <typename T>
T lookupReplicationPadding(T* data, Dim2 size, Dim2 padding, T constant, uint64_t x, uint64_t y) {
  if (x < padding.x) {
    x = padding.x;
  } else if (x >= (size.x + padding.x)) {
    x = size.x + padding.x - 1;
  }
  if (y < padding.y) {
    y = padding.y;
  } else if (y >= (size.y + padding.y)) {
    y = size.y + padding.y - 1;
  }
  return data[(y - padding.y) * size.x + (x - padding.x)];
}

/*
  Dim2 outS;
  outS.x = (inS.x + 2 * padding.x - kernS.x) / stride.x + 1;
  outS.y = (inS.y + 2 * padding.y - kernS.y) / stride.y + 1;
*/

// TODO implement padding modes
// TODO implement dilation
/// https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
template <typename T>
__global__ void conv2d(T* output, const T* input, const T* kernel, Dim2 outS, Dim2 inS, Dim2 kernS, Dim2 padding, Dim2 stride, PaddingMode paddingMode) {
  int output_row = blockIdx.y * blockDim.y + threadIdx.y;
  int output_col = blockIdx.x * blockDim.x + threadIdx.x;

  auto padder = lookupConstantPadding<T>;
  switch (paddingMode) {
    case CONSTANT:
      padder = lookupConstantPadding<T>;
      break;
    case CIRCULAR:
      padder = lookupCircularPadding<T>;
      break;
    case REFLECT:
      padder = lookupReflectPadding<T>;
      break;
    case REPLICATION:
      padder = lookupReplicationPadding<T>;
      break;
  }

  if (output_row < outS.y && output_col < outS.x) {
    T value = 0;
    for (int i = 0; i < kernS.y; i++) {
      for (int j = 0; j < kernS.x; j++) {
        int input_row = output_row * stride.y + i;
        int input_col = output_col * stride.x + j;
        if (input_row >= 0 && input_row < inS.y && input_col >= 0 && input_col < inS.x) {
          value += padder<T>(input, inS, padding, 0, input_col, input_row) * kernel[i * kernS.x + j];
        } else {
          // TODO error
        }
      }
    }
    output[output_row * outS.x + output_col] = value;
  }
}

void conv2d_cuda(float* output, const float* input, const float* kernel,
                 int input_height, int input_width, int kernel_size) {
  int output_height = input_height - kernel_size + 1;
  int output_width = input_width - kernel_size + 1;

  float* d_input;
  float* d_kernel;
  float* d_output;

  /*
  outS.x = (inS.x + 2 * padding.x - kernS.x) / stride.x + 1;
  outS.y = (inS.y + 2 * padding.y - kernS.y) / stride.y + 1;
  */

  cudaMalloc((void**)&d_input, input_height * input_width * sizeof(float));
  cudaMalloc((void**)&d_kernel, kernel_size * kernel_size * sizeof(float));
  cudaMalloc((void**)&d_output, output_height * output_width * sizeof(float));

  cudaMemcpy(d_input, input, input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

  dim3 block_size(16, 16);
  dim3 grid_size((output_width + block_size.x - 1) / block_size.x,
                 (output_height + block_size.y - 1) / block_size.y);

#include <cuda_runtime.h>

  conv2d<<<grid_size, block_size>>>(d_input, d_kernel, d_output,
                                    input_height, input_width, kernel_size);

  cudaMemcpy(output, d_output, output_height * output_width * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_kernel);
  cudaFree(d_output);
}
