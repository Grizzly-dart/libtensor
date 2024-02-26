#include <cuda_runtime.h>

template <typename T>
__global__ void conv2d(T* output, const T* input, const T* kernel, 
             int input_height, int input_width, int kernel_size, int padding) {
  int output_height = input_height + 2 * padding - kernel_size + 1;
  int output_width = input_width + 2 * padding - kernel_size + 1;

  int output_row = blockIdx.y * blockDim.y + threadIdx.y;
  int output_col = blockIdx.x * blockDim.x + threadIdx.x;

  if (output_row < output_height && output_col < output_width) {
    float value = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
      for (int j = 0; j < kernel_size; j++) {
        int input_row = output_row + i - padding;
        int input_col = output_col + j - padding;
        if (input_row >= 0 && input_row < input_height && input_col >= 0 && input_col < input_width) {
          value += input[input_row * input_width + input_col] * kernel[i * kernel_size + j];
        }
      }
    }
    output[output_row * output_width + output_col] = value;
  }
}

template <typename T>
__global__ void conv2d(T* output, const T* input, const T* kernel, 
                       int input_height, int input_width, int kernel_size) {
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;

    int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    int output_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (output_row < output_height && output_col < output_width) {
        float value = 0.0f;
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                int input_row = output_row + i;
                int input_col = output_col + j;
                value += input[input_row * input_width + input_col] * kernel[i * kernel_size + j];
            }
        }
        output[output_row * output_width + output_col] = value;
    }
}

void conv2d_cuda(float* output, const float* input, const float* kernel, 
                  int input_height, int input_width, int kernel_size) {
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;

    float* d_input;
    float* d_kernel;
    float* d_output;

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
