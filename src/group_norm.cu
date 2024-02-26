#include <cuda_runtime.h>

__global__ void groupNorm(float* input, float* output, int numChannels, int height, int width, int groupSize, float epsilon)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = tid / (height * width);
    int spatialIdx = tid % (height * width);

    if (channel < numChannels)
    {
        int group = channel / groupSize;
        int groupStart = group * groupSize;
        int groupEnd = min(groupStart + groupSize, numChannels);

        float mean = 0.0f;
        float variance = 0.0f;

        // Compute mean and variance within the group
        for (int i = groupStart; i < groupEnd; i++)
        {
            int idx = i * height * width + spatialIdx;
            float value = input[idx];
            mean += value;
            variance += value * value;
        }

        mean /= groupSize;
        variance = variance / groupSize - mean * mean;

        // Normalize within the group
        for (int i = groupStart; i < groupEnd; i++)
        {
            int idx = i * height * width + spatialIdx;
            float value = input[idx];
            output[idx] = (value - mean) / sqrtf(variance + epsilon);
        }
    }
}

int main()
{
    // Example usage
    int numChannels = 64;
    int height = 32;
    int width = 32;
    int groupSize = 4;
    float epsilon = 1e-5;

    // Allocate and initialize input data
    float* input;
    cudaMallocManaged(&input, numChannels * height * width * sizeof(float));
    // Initialize input data...

    // Allocate output data
    float* output;
    cudaMallocManaged(&output, numChannels * height * width * sizeof(float));

    // Launch the kernel
    int numThreads = numChannels * height * width;
    int blockSize = 256;
    int numBlocks = (numThreads + blockSize - 1) / blockSize;
    groupNorm<<<numBlocks, blockSize>>>(input, output, numChannels, height, width, groupSize, epsilon);
    cudaDeviceSynchronize();

    // Clean up
    cudaFree(input);
    cudaFree(output);

    return 0;
}
