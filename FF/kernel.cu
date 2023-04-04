#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void blackAndWhite(unsigned char* image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int index = (y * width + x) * 3;
        unsigned char r = image[index];
        unsigned char g = image[index + 1];
        unsigned char b = image[index + 2];
        unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        image[index] = gray;
        image[index + 1] = gray;
        image[index + 2] = gray;
    }
}

int main() {
    int width, height, channels;
    unsigned char* image = stbi_load("image.jpg", &width, &height, &channels, 0);

    unsigned char* d_image;
    cudaMalloc(&d_image, width * height * channels);
    cudaMemcpy(d_image, image, width * height * channels, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    blackAndWhite << <numBlocks, threadsPerBlock >> > (d_image, width, height);

    cudaMemcpy(image, d_image, width * height * channels, cudaMemcpyDeviceToHost);
    stbi_write_jpg("output.jpg", width, height, channels, image, 100);

    stbi_image_free(image);
    cudaFree(d_image);
    return 0;
}