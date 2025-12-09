Project Description

This project demonstrates GPU-accelerated signal processing using CUDA. A large dataset of signal samples stored in a CSV file is processed on the GPU, where each value is scaled by a constant factor using a CUDA kernel. The program uses parallel threads to process millions of signal values simultaneously.

GPU Usage

The CUDA kernel assigns one thread per signal element. Each thread multiplies its assigned value by a constant scale factor, demonstrating data-parallel signal processing.

Dataset

The dataset is a CSV file containing over one million signal values.

Outcome

All signal values are processed using the GPU and stored into an output file.


Generate Dataset
for i in {1..1000000}; do echo $((RANDOM % 100)); done > signal.csv

Compile and Run

nvcc gpu_signal_scale.cu -o signal_gpu
./signal_gpu

