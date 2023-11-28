#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>

// Timer struct/functions from support.cu in vector addition example

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

// End of Timer struct/functions

// Takes an array of unsigned chars [0,255] rgbImage and a width and height of image and computes the grayScale image
__global__ void rgb_to_grayscale(unsigned char* grayImage, unsigned char* rgbImage, int width, int height){
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    // 1-D coordinate for grayscale image
    int grayOffset = Row*width + Col;

    if (grayOffset < width * height) {
        
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
    Timer timer;
    size_t FILE_SIZE = 1024 * 1024 * 3;
    size_t GRAY_FILE_SIZE = 1024 * 1024;
    unsigned char buffer_h[FILE_SIZE]; // Host buffer
    unsigned char* buffer_d; // Device buffer
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

    int TIMES_TO_RUN_KERNEL = 200;
    // Define dimensions of cuda grid
    dim3 DimGrid(1024,1024,1);
    dim3 DimBlock(1,1,1);

    // TIMING (1024,1024,1) (1,1,1) ARCHITECTURE
    // Time kernel ran TIMES_TO_RUN_KERNEL times
    startTime(&timer);
    for (int i = 0; i < TIMES_TO_RUN_KERNEL; i++) {
        rgb_to_grayscale<<<DimGrid,DimBlock>>>(gray_buffer_d,buffer_d,1024,1024);
        cudaDeviceSynchronize();
    }
    stopTime(&timer);
    printf("Time for (1024,1024,1) (1,1,1) = %f s\n", elapsedTime(timer));

    // TIMING (1024,1,1) (1024,1,1) ARCHITECTURE
    DimGrid = dim3(1024,1,1);
    DimBlock = dim3(1024,1,1);

    // Time kernel ran TIMES_TO_RUN_KERNEL times
    startTime(&timer);
    for (int i = 0; i < TIMES_TO_RUN_KERNEL; i++) {
        rgb_to_grayscale<<<DimGrid,DimBlock>>>(gray_buffer_d,buffer_d,1024,1024);
        cudaDeviceSynchronize();
    }
    stopTime(&timer);
    printf("Time for (1024,1,1) (1024,1,1) = %f s\n", elapsedTime(timer));

    // TIMING (32,32,1) (32,32,1) ARCHITECTURE
    DimGrid = dim3(32,32,1);
    DimBlock = dim3(32,32,1);

    // Time kernel ran TIMES_TO_RUN_KERNEL times
    startTime(&timer);
    for (int i = 0; i < TIMES_TO_RUN_KERNEL; i++) {
        rgb_to_grayscale<<<DimGrid,DimBlock>>>(gray_buffer_d,buffer_d,1024,1024);
        cudaDeviceSynchronize();
    }
    stopTime(&timer);
    printf("Time for (32,32,1) (32,32,1) = %f s\n", elapsedTime(timer));

    // Copy device buffer to host
    cudaMemcpy(gray_buffer_h, gray_buffer_d, GRAY_FILE_SIZE, cudaMemcpyDeviceToHost);

    // Write buffer to file
    FILE *file = fopen("gc.raw", "wb");
    size_t w = fwrite(gray_buffer_h, sizeof(unsigned char), GRAY_FILE_SIZE, file);
    fclose(file);

    cudaFree(buffer_d);
    cudaFree(gray_buffer_d);

    return 0;
}