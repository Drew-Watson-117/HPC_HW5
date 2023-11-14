#include <stdlib.h>
#include <stdio.h>

// Takes an array of unsigned chars [0,255] rgbImage and a width and height of image and computes the grayScale image
__global__ void rgb_to_grayscale(unsigned char* grayImage, unsigned char* rgbImage, int width, int height){
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;

    if (Col < width && Row < height) {
        // 1-D coordinate for grayscale image
        int grayOffset = Row*width + Col;
        // RGB image has 3 * columns of grayscale image
        int rgbOffset = grayOffest * 3;
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
    char buffer_h[FILE_SIZE]; // Host buffer
    char buffer_d[FILE_SIZE]; // Device buffer
    char gray_buffer_d[GRAY_FILE_SIZE]; // Buffer for grayscale image on device
    char gray_buffer_h[GRAY_FILE_SIZE]; // Buffer for grayscale image on host
    // Allocate space on the cuda device for the device buffer
    cudaMalloc((void **) &buffer_d, FILE_SIZE);
    cudaMalloc((void **) &gray_buffer_d, GRAY_FILE_SIZE);
    cudaDeviceSynchronize();

    FILE *image_raw = fopen("gc_conv_1024x1024.raw","rb");
    size_t r = read(image_raw, buffer_h, FILE_SIZE);
    fclose(image_raw);

    // Copy Host buffer to Device buffer
    cudaMemcpy(buffer_d, buffer_h, FILE_SIZE, cudaMemcpyHostToDevice);

    // Define dimensions of cuda grid
    // TODO: MAKE SURE THIS DIMENSION ENSURES Row BECOMES 1024 AND Col BECOMES 1024
    dim3 DimGrid(ceil(GRAY_FILE_SIZE/1024,1,1));
    dim3 DimBlock(1024,1,1);
    rgb_to_grayscale<<<DimGrid,DimBlock>>>(gray_buffer_d,buffer_d,1024,1024);

    // Copy device buffer to host
    cudaMemcpy(gray_buffer_h, gray_buffer_d, GRAY_FILE_SIZE, cudaMemcpyDeviceToHost);

    // Write buffer to file
    FILE *file = fopen("gc.raw", "w");
    fprintf(file,gray_buffer_h);
    fclose(file);

    cudaFree(buffer_d);
    cudaFree(gray_buffer_d);
    free(buffer_h);
    free(gray_buffer_h);

    return 0;
}