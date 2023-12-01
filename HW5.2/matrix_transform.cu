#include <stdlib.h>
#include <stdio.h>

#define TILE_WIDTH 32


// Takes an input rgb image inImage, its width and height, and transposes it then stores it in outImage
__global__ void matrix_transpose(unsigned char* inImage, unsigned char* outImage, int width, int height){
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;

    if (Row*width+Col < width*height) {
        // 1-D coordinate for input image
        int offset = 3*(Row*width + Col);
        // 1-D coordinate for the transposed matrix
        int transposeIndex = 3 * (Col*width + Row);
        outImage[offset] = inImage[transposeIndex];
        outImage[offset+1] = inImage[transposeIndex+1];
        outImage[offset+2] = inImage[transposeIndex+2];
    }
}

//Matrix Transpose Kernel with Tiling
__global__ void tiled_transpose(unsigned char* inImage, unsigned char* outImage, int width, int height) {
    __shared__ unsigned char subTile[3*TILE_WIDTH*TILE_WIDTH];
    int threadX = threadIdx.x; int threadY = threadIdx.y;
    int blockX = blockIdx.x; int blockY = blockIdx.y;
    int Col = threadX + blockX * blockDim.x;
    int Row = threadY + blockY * blockDim.y;

    int inIndex = 3 * (Row * width + Col);

    // Write block to shared memory, transposing it
    int tileIdx = 3 * (threadX*TILE_WIDTH + threadY); // Transposed index in tile
    subTile[tileIdx] = inImage[inIndex];
    subTile[tileIdx + 1] = inImage[inIndex + 1];
    subTile[tileIdx + 2] = inImage[inIndex + 2];
    __syncthreads(); // Make sure all threads have written before reading
    // Determine start index of transposed block
    int transposedBlockStart = (blockDim.x*blockDim.y*blockDim.z)*gridDim.x*blockX +blockY*gridDim.x;
    int outIndex = 3*(transposedBlockStart + threadY*width+threadX);
    // Write contiguously to global memory
    int tileIndexOut = 3 * (threadY * TILE_WIDTH + threadX);
    outImage[outIndex] = subTile[tileIndexOut];
    outImage[outIndex+1] = subTile[tileIndexOut+1];
    outImage[outIndex+2] = subTile[tileIndexOut+2];
    __syncthreads(); // Make sure all threads have written before continuing
}

// Serial computation of the matrix transpose
void cpu_matrix_transpose(unsigned char* inImage, unsigned char* outImage, int width, int height)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Define 1-D indicies for input and transposed images
            int offset = 3*(y * width + x);
            int transposeIndex = 3*(x * width + y);
            // Copy red, green, and blue chars to the transposed pixel
            outImage[offset] = inImage[transposeIndex];
            outImage[offset+1] = inImage[transposeIndex+1];
            outImage[offset+2] = inImage[transposeIndex+2];
        }
    }
}

// Validate the results of a matrix transpose calculation against the serial transpose computation
void validateResult(unsigned char* result, unsigned char* input_buffer, size_t FILE_SIZE) {

    unsigned char cpu_buffer_h[FILE_SIZE]; // Buffer for output as computed by the cpu for comparison
    // Compute serial transpose
    cpu_matrix_transpose(input_buffer,cpu_buffer_h,1024,1024);

    // Flag to determine whether at every index the two outputs are equal
    int areEqual = 0;
    for(int i = 0; i < FILE_SIZE; i++) {
        if (result[i] != cpu_buffer_h[i]) {
            areEqual = 1;
            printf("Parallel and Serial DO NOT match\n");
            printf("result value: %u\n", result[i]);
            printf("serial value: %u\n", cpu_buffer_h[i]);
            printf("Global Index: %d\n", i);
            break;
        }
    }
    if (areEqual == 0) {printf("Parallel and Serial Match!\n");}
}

int main()
{
    size_t FILE_SIZE = 1024 * 1024 * 3;
    unsigned char buffer_h[FILE_SIZE]; // Host buffer
    unsigned char* buffer_d; // Device buffer
    unsigned char* out_buffer_d; // Buffer for output image on device
    unsigned char out_buffer_h[FILE_SIZE]; // Buffer for output image on host
    unsigned char* tiled_buffer_d; // Buffer for output with tiling on device
    unsigned char tiled_buffer_h[FILE_SIZE]; // Buffer for output with tiling on host
    // Allocate space on the cuda device for the device buffer
    cudaMalloc((void **) &buffer_d, FILE_SIZE);
    cudaMalloc((void **) &out_buffer_d, FILE_SIZE);
    cudaMalloc((void **) &tiled_buffer_d, FILE_SIZE);

    FILE *image_raw = fopen("gc_1024x1024.raw","rb");
    size_t r = fread(buffer_h, sizeof(unsigned char), FILE_SIZE, image_raw);
    fclose(image_raw);

    // Copy Host buffer to Device buffer
    cudaMemcpy(buffer_d, buffer_h, FILE_SIZE, cudaMemcpyHostToDevice);

    //Define events for timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Define dimensions of cuda grid
    dim3 DimGrid(32,32,1);
    dim3 DimBlock(32,32,1);

    //Call the kernel to transpose the matrix
    cudaEventRecord(start);
    matrix_transpose<<<DimGrid,DimBlock>>>(buffer_d,out_buffer_d,1024,1024);
    cudaEventRecord(stop);
    
    // Calculate Bandwidth
    cudaEventSynchronize(stop);
    float timeOfNonTiledKernel = 0; // Execution time of kernel in milliseconds
    cudaEventElapsedTime(&timeOfNonTiledKernel, start, stop);
    float bandwidth = (6*sizeof(unsigned char) * 1024 * 1024) / timeOfNonTiledKernel / 1e6;

    // Define events for timing of tiled kernel
    cudaEvent_t tileStart, tileStop;
    cudaEventCreate(&tileStart);
    cudaEventCreate(&tileStop);

    // Define dimensions of grid for tiling
    dim3 TiledGrid(1024/TILE_WIDTH,1024/TILE_WIDTH,1);
    dim3 TiledBlock(TILE_WIDTH,TILE_WIDTH,1);

    //Call kernel for tiled transpose
    cudaEventRecord(tileStart);
    tiled_transpose<<<TiledGrid,TiledBlock>>>(buffer_d,tiled_buffer_d,1024,1024);
    cudaEventRecord(tileStop);

    // Calculate Bandwidth
    cudaEventSynchronize(tileStop);
    float timeOfTiledKernel = 0; // Execution time of kernel with tiling in milliseconds
    cudaEventElapsedTime(&timeOfTiledKernel,tileStart,tileStop);
    float tiledBandwidth = (6*sizeof(unsigned char) * 1024 * 1024) / timeOfTiledKernel / 1e6;

    // Copy device buffer to host
    cudaMemcpy(out_buffer_h, out_buffer_d, FILE_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(tiled_buffer_h, tiled_buffer_d, FILE_SIZE, cudaMemcpyDeviceToHost);


    // Write buffer to file
    FILE *file = fopen("gc_transpose.raw", "wb");
    size_t w = fwrite(out_buffer_h, sizeof(unsigned char), FILE_SIZE, file);
    fclose(file);

    // Write tiled buffer to file
    file = fopen("gc_tiled_transpose.raw", "wb");
    w = fwrite(tiled_buffer_h, sizeof(unsigned char), FILE_SIZE, file);
    fclose(file);

    // Print Bandwidth
    printf("===== Bandwidth Calculation ===== \n");
    printf("Bandwidth of Non-Tiled Kernel (GB/s): %f\n", bandwidth);
    printf("Bandwidth of Tiled Kernel (GB/s): %f\n", tiledBandwidth);

    // Compare to serial implementation
    printf("\n===== Validating Global Memory Transpose =====\n");
    validateResult(out_buffer_h, buffer_h, FILE_SIZE);
    printf("\n===== Validating Shared Memory Transpose =====\n");
    validateResult(tiled_buffer_h, buffer_h, FILE_SIZE);

    cudaFree(buffer_d);
    cudaFree(out_buffer_d);
    cudaFree(tiled_buffer_d);

    return 0;
}