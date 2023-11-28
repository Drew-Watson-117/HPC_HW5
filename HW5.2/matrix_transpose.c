#include <stdlib.h>
#include <stdio.h>


void cpu_matrix_transpose(unsigned char* inImage, unsigned char* outImage, int width, int height)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int offset = 3*(y * width + x);
            int transposeIndex = 3*(x * width + y);
            outImage[offset] = inImage[transposeIndex];
            outImage[offset+1] = inImage[transposeIndex+1];
            outImage[offset+2] = inImage[transposeIndex+2];
        }
    }
}
int main()
{
    #define FILE_SIZE  1024 * 1024 * 3
    char buffer_h[FILE_SIZE]; // Host buffer
    char cpu_buffer_h[FILE_SIZE]; // Buffer for output as computed by the cpu for comparison

    FILE *image_raw = fopen("gc_1024x1024.raw","rb");
    size_t r = fread(buffer_h, sizeof(unsigned char), FILE_SIZE, image_raw);
    fclose(image_raw);

    // Apply transpose
    cpu_matrix_transpose(buffer_h,cpu_buffer_h,1024,1024);

    // Write buffer to file
    FILE *file = fopen("gc.raw", "wb");
    size_t w = fwrite(cpu_buffer_h, sizeof(unsigned char), FILE_SIZE, file);
    fclose(file);


    // free(buffer_h);
    // free(cpu_buffer_h);

    return 0;
}