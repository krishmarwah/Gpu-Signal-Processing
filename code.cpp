#include <iostream>
#include <vector>
#include <fstream>
#include <cuda.h>

__global__ void scaleSignal(float *data, int n, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] *= factor;
}

int main() {
    std::vector<float> signal;
    float value;

    // Load CSV data
    std::ifstream file("signal.csv");
    while (file >> value)
        signal.push_back(value);

    int n = signal.size();
    size_t bytes = n * sizeof(float);

    // Allocate GPU memory
    float *d_data;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, signal.data(), bytes, cudaMemcpyHostToDevice);

    // Kernel launch
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scaleSignal<<<blocks, threads>>>(d_data, n, 2.0f);
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(signal.data(), d_data, bytes, cudaMemcpyDeviceToHost);

    // Save output
    std::ofstream out("output.csv");
    for (float x : signal)
        out << x << "\n";

    cudaFree(d_data);
    std::cout << "GPU Signal Processing Complete\n";
}

