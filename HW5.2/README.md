# Comments on Bandwidth Performance

- For the Non-Tiled Kernel, the bandwidth is about 48 GB/s
- For the Tiled Kernel, the bandwidth is about 128 GB/s

Using tiling to put the transpose operation in shared memory and making all global memory access contiguous, the bandwidth performance of the kernel triples. This shows that tiling offers a significant benefit when compared to doing the calculation in global memory.

- When tiling is used in the kernel, the bandwidth performance of the kernel triples.
    - Tiling allows all reading from and writing to global memory to be accessed contiguously
    - Tiling allows the transpose operation to be done by writing to shared memory in a contiguous manner
- Because of the bandwidth performance increase that tiling provides, it is clear that using tiling provides a significant performance benefit when programming GPU kernels.