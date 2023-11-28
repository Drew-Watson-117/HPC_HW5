# Performance Explanation

## CUDA device_query

- NVIDIA TITAN V
- Warp Size 32
- Max Threads / Multiprocessor : 2048
- Max Threads / Block : 1024
- Max Block Dim : (1024,1024,64)
- Max Grid Dim : (2147483647, 65535, 65535)

## Block Architectures
Three block sizes chosen (Block Size) & (Grid Size):

1. Block/Grid Size 1: (1024,1024,1) & (1,1,1)
2. Block/Grid Size 2: (1024,1,1) & (1024,1,1)
3. Block/Grid Size 3: (32,32,1) & (32,32,1)

### Expectations
1. Block/Grid Size 1 has all the threads in one thread block. The issue with this architecture is that the max threads per block is 1024, so the number of threads per block for the (1024,1024,1) case exceeds the maximum. Because of this, I expect this architecture to perform worse than the other two.

2. Block/Grid Size 2 has all the threads laid out on the x axis of the grid. This architecture does not exceed any maximums, and the threads being used are contiguous, so I expect this architecture to perform best.

3. Block/Grid Size 3 has the threads laid out on a 2-D plane, with 1024 threads laid out in 32x32 size blocks, and 1024 blocks laid out in a 32x32 grid. Because this approach doesn't use contiguous threads in the GPU, I expect this to perform slightly worse than Block/Grid Size 2. However, this architecture is more scalable to more threads as it utilizes 2 dimensions of the GPU instead of just 1.

### Experimental Results
Output for the timed_kernel.cu program, with running each kernel 200 times. 

1. Time for (1024,1024,1) (1,1,1) = 0.359000 s
2. Time for (1024,1,1) (1024,1,1) = 0.002412 s
3. Time for (32,32,1) (32,32,1) = 0.002432 s

As expected, Block/Grid Size 1 performed the worst because of the fact that it exceeded the maximum threads/block. Block/Grid Size 2 performed best, probably because the threads were contiguously allocated. Block/Grid Size 3 performed in the middle, but only slightly worse than Block/Grid Size 2. However, the performance of Block/Grid Size 2 is not always better than the performance of Block/Grid Size 3. Sometimes, Block/Grid Size 3 yields a better performance (smaller execution time).
