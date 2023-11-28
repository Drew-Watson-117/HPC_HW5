#include <stdlib.h>
#include <stdio.h>

// Takes an input rgb image inImage, its width and height, and transposes it
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
    unsigned char buffer_h[FILE_SIZE]; // Host buffer
    unsigned char* buffer_d; // Device buffer
    unsigned char* out_buffer_d; // Buffer for output image on device
    unsigned char out_buffer_h[FILE_SIZE]; // Buffer for output image on host
    unsigned char cpu_buffer_h[FILE_SIZE]; // Buffer for output as computed by the cpu for comparison
    // Allocate space on the cuda device for the device buffer
    cudaMalloc((void **) &buffer_d, FILE_SIZE);
    cudaMalloc((void **) &out_buffer_d, FILE_SIZE);

    FILE *image_raw = fopen("gc_1024x1024.raw","rb");
    size_t r = fread(buffer_h, sizeof(unsigned char), FILE_SIZE, image_raw);
    fclose(image_raw);

    // Copy Host buffer to Device buffer
    cudaMemcpy(buffer_d, buffer_h, FILE_SIZE, cudaMemcpyHostToDevice);

    // Define dimensions of cuda grid
    dim3 DimGrid(32,32,1);
    dim3 DimBlock(32,32,1);

    //Call the kernel to transpose the matrix
    matrix_transpose<<<DimGrid,DimBlock>>>(buffer_d,out_buffer_d,1024,1024);

    // Copy device buffer to host
    cudaMemcpy(out_buffer_h, out_buffer_d, FILE_SIZE, cudaMemcpyDeviceToHost);

    // Write buffer to file
    FILE *file = fopen("gc_transpose.raw", "wb");
    size_t w = fwrite(out_buffer_h, sizeof(unsigned char), FILE_SIZE, file);
    fclose(file);

    // Compare to serial implementation
    cpu_matrix_transpose(buffer_h,cpu_buffer_h,1024,1024);

    FILE *serial = fopen("gc_transpose_serial.raw", "wb");
    w = fwrite(cpu_buffer_h, sizeof(unsigned char), FILE_SIZE, serial);
    fclose(serial);

    int areEqual = 0;
    for(int i = 0; i < FILE_SIZE; i++) {
        if (out_buffer_h[i] != cpu_buffer_h[i]) {
            areEqual = 1;
            printf("Parallel and Serial DO NOT match\n");
            printf("out_buffer_h value: %u\n", out_buffer_h[i]);
            printf("cpu_buffer_h value: %u\n", cpu_buffer_h[i]);
            printf("Global Index: %d\n", i);
            break;
        }
    }
    if (areEqual == 0) {printf("Parallel and Serial Match!\n");}

    cudaFree(buffer_d);
    cudaFree(out_buffer_d);

    return 0;
}