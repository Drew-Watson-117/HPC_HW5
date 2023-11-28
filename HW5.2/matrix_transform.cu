#include <stdlib.h>
#include <stdio.h>

// Takes an input rgb image inImage, its width and height, and transposes it
__global__ void matrix_transpose(unsigned char* inImage, unsigned char* outImage, int width, int height){
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;

    if (Col < width && Row < height) {
        // 1-D coordinate for input image
        int offset = 3*(Row*width + Col);
        // 1-D coordinate for the transposed matrix
        int transposeIndex = 3 * (Col*width + Row);
        outImage[offset] = inImage[transposeIndex];
        outImage[offset+1] = inImage[transposeIndex+1];
        outImage[offset+2] = inImage[transposeIndex+2];
    }
}

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

int main()
{
    size_t FILE_SIZE = 1024 * 1024 * 3;
    char buffer_h[FILE_SIZE]; // Host buffer
    char buffer_d[FILE_SIZE]; // Device buffer
    char out_buffer_d[FILE_SIZE]; // Buffer for output image on device
    char out_buffer_h[FILE_SIZE]; // Buffer for output image on host
    char cpu_buffer_h[FILE_SIZE]; // Buffer for output as computed by the cpu for comparison
    // Allocate space on the cuda device for the device buffer
    cudaMalloc((void **) &buffer_d, FILE_SIZE);
    cudaMalloc((void **) &out_buffer_d, FILE_SIZE);
    cudaDeviceSynchronize();

    FILE *image_raw = fopen("gc_1024x1024.raw","rb");
    size_t r = read(image_raw, buffer_h, FILE_SIZE);
    fclose(image_raw);

    // Copy Host buffer to Device buffer
    cudaMemcpy(buffer_d, buffer_h, FILE_SIZE, cudaMemcpyHostToDevice);

    // Define dimensions of cuda grid
    dim3 DimGrid(1024,1,1);
    dim3 DimBlock(1024,1,1);
    matrix_transpose<<<DimGrid,DimBlock>>>(buffer_d,out_buffer_d,1024,1024);

    // Copy device buffer to host
    cudaMemcpy(out_buffer_h, out_buffer_d, FILE_SIZE, cudaMemcpyDeviceToHost);

    // Write buffer to file
    FILE *file = fopen("gc.raw", "wb");
    size_t w = fwrite(out_buffer_h, sizeof(unsigned char), FILE_SIZE, file);
    fclose(file);

    // Compare to serial implementation
    cpu_matrix_transpose(buffer_h,cpu_buffer_h,1024,1024);

    cudaFree(buffer_d);
    cudaFree(out_buffer_d);
    free(buffer_h);
    free(out_buffer_h);
    free(cpu_buffer_h);

    return 0;
}