#include <stdlib.h>
#include <stdio.h>

// Takes an array of unsigned chars [0,255] rgbImage and a width and height of image and computes the grayScale image
__global__ void rgb_to_grayscale(unsigned char* grayImage, unsigned char* rgbImage, int width, int height){
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    // 1-D coordinate for grayscale image
    int grayOffset = Row*width + Col;

    // Make sure grayOffset is in image boundary
    if (grayOffset < width*height) {
        // RGB image has 3 * columns of grayscale image
        int rgbOffset = grayOffset * 3;
        unsigned char r = rgbImage[rgbOffset];
        unsigned char g = rgbImage[rgbOffset + 1];
        unsigned char b = rgbImage[rgbOffset + 2];
        // Rescale by multiplying r, g, and b by float constants
        grayImage[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b; 
    }
}

int main()
{
    size_t FILE_SIZE = 1024 * 1024 * 3;
    size_t GRAY_FILE_SIZE = 1024 * 1024;
    unsigned char buffer_h[FILE_SIZE]; // Host buffer for RGB image
    unsigned char* buffer_d; // Device buffer for RGB image
    unsigned char* gray_buffer_d; // Buffer for grayscale image on device
    unsigned char gray_buffer_h[GRAY_FILE_SIZE]; // Buffer for grayscale image on host
    // Allocate space on the cuda device for the device buffer
    cudaMalloc((void **) &buffer_d, FILE_SIZE);
    cudaMalloc((void **) &gray_buffer_d, GRAY_FILE_SIZE);

    FILE *image_raw = fopen("gc_conv_1024x1024.raw","rb");
    size_t r = fread(buffer_h, sizeof(unsigned char), FILE_SIZE, image_raw);
    fclose(image_raw);

    // Copy Host buffer to Device buffer
    cudaMemcpy(buffer_d, buffer_h, FILE_SIZE, cudaMemcpyHostToDevice);

    // Define dimensions of cuda grid
    dim3 DimGrid(1024,1,1);
    dim3 DimBlock(1024,1,1);

    // Run kernel on cuda grid
    rgb_to_grayscale<<<DimGrid,DimBlock>>>(gray_buffer_d,buffer_d,1024,1024);

    // Copy device buffer to host
    cudaMemcpy(gray_buffer_h, gray_buffer_d, GRAY_FILE_SIZE, cudaMemcpyDeviceToHost);

    // Write buffer to file
    FILE *file = fopen("gc.raw", "wb");
    size_t w = fwrite(gray_buffer_h, sizeof(unsigned char), GRAY_FILE_SIZE, file);
    fclose(file);

    // Free device memory
    cudaFree(buffer_d);
    cudaFree(gray_buffer_d);

    return 0;
}