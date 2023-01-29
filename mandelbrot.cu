#include <stdio.h>
#include <cuda_runtime.h>
#include "dbldbl.h"

extern "C" {
__global__ void mandelbrot_kernel(unsigned char *output, int width, int height, float xMin, float xMax, float yMin, float yMax) {
    //printf("BBCCDD griddim:%d %d %d, blockdim: %d %d %d\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
    //printf("BBCCDE blockIdx:%d %d %d, threadIdx: %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
   
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    //int y = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.y;

    float xCoord = xMin + (xMax - xMin) * x / width;
    float yCoord = yMin + (yMax - yMin) * y / height;

    float real = xCoord;
    float imag = yCoord;

    float real2 = real * real;
    float imag2 = imag * imag;

    int iterations = 0;
    //printf("Hello%d\n", iterations);
    while (real2 + imag2 < 4.0f && iterations < 255) {
        imag = 2 * real * imag + yCoord;
        real = real2 - imag2 + xCoord;
        real2 = real * real;
        imag2 = imag * imag;
        iterations++;
    }

    int offset = y * width + x;
    //printf("%d\n", iterations);
    output[offset] = iterations;
}
void mandelbrot(unsigned char *output, int width, int height, float xMin, float xMax, float yMin, float yMax) {
    //printf("AAAAA\n");
    //dim3 gridDim(1, 1, 1);
    //dim3 blockDim(width, height, 1);
    dim3 gridDim(4, 4096, 1);
    dim3 blockDim(1024, 1, 1);

    unsigned char * dev_data;

    cudaMalloc((void**)&dev_data, width*height);
    //cudaMemcpy(host_data, dev_data, width*height, cudaMemcpyHostToDevice);

    mandelbrot_kernel<<<gridDim, blockDim>>>(dev_data, width, height, xMin, xMax, yMin, yMax);
    cudaDeviceSynchronize();

    cudaMemcpy(output, dev_data, width*height, cudaMemcpyDeviceToHost);

}
__global__ void mandelbrot2_kernel(int *output, int width, int height, double xMin, double xMax, double yMin, double yMax, int maxiter) {
    //printf("BBCCDD griddim:%d %d %d, blockdim: %d %d %d\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
    //printf("BBCCDE blockIdx:%d %d %d, threadIdx: %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
   
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    //int y = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.y;
    if (x>=width)
        return ;

    double xCoord = xMin + (xMax - xMin) * x / width;
    double yCoord = yMin + (yMax - yMin) * y / height;

    double real = xCoord;
    double imag = yCoord;

    double real2 = real * real;
    double imag2 = imag * imag;

    int iterations = 0;
    //printf("Hello%d\n", iterations);
    while (real2 + imag2 < double(4.0) && iterations < maxiter) {
        imag = 2 * real * imag + yCoord;
        real = real2 - imag2 + xCoord;
        real2 = real * real;
        imag2 = imag * imag;
        iterations++;
    }

    int offset = y * width + x;
    //printf("%d\n", iterations);
    output[offset] = iterations;
}
void mandelbrot2(int *output, int width, int height, double xMin, double xMax, double yMin, double yMax, int maxiter) {
    //printf("AAAAA\n");
    //dim3 gridDim(1, 1, 1);
    //dim3 blockDim(width, height, 1);
    int x = int((width + 511) / 512);
    dim3 gridDim(x, height, 1);
    dim3 blockDim(512, 1, 1);

    static int * dev_data = 0;
    if (dev_data == 0)
        cudaMalloc((void**)&dev_data, width*height*sizeof(int));
    //cudaMemcpy(host_data, dev_data, width*height, cudaMemcpyHostToDevice);

    mandelbrot2_kernel<<<gridDim, blockDim>>>(dev_data, width, height, xMin, xMax, yMin, yMax, maxiter);
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("Error: %s\n", cudaGetErrorString(error));
        //return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(output, dev_data, width*height*sizeof(int), cudaMemcpyDeviceToHost);

}
__global__ void mandelbrot3_kernel(int *output, int width, int height, double cx_h, double cx_t, double cy_h, double cy_t, double scale, int maxiter) {
    //printf("BBCCDD griddim:%d %d %d, blockdim: %d %d %d\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
    //printf("BBCCDE blockIdx:%d %d %d, threadIdx: %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
   
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    //int y = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.y;
    if (x>=width)
        return ;
    dbldbl cx = add_double_to_dbldbl(cx_h, cx_t);
    dbldbl cy = add_double_to_dbldbl(cy_h, cy_t);
    dbldbl _width = make_dbldbl(width, 0);
    dbldbl _height = make_dbldbl(height, 0);
    dbldbl _scale = make_dbldbl(scale, 0);
    dbldbl whr = div_dbldbl(_width, _height); 
    dbldbl xMin = sub_dbldbl(cx, mul_dbldbl(_scale, whr));
    dbldbl xMax = add_dbldbl(cx, mul_dbldbl(_scale, whr));;
    dbldbl yMin = add_dbldbl(cy, _scale);
    dbldbl yMax = sub_dbldbl(cy, _scale);

    dbldbl _x = make_dbldbl(x, 0);
    dbldbl _y = make_dbldbl(y, 0);
    dbldbl _two = make_dbldbl(2, 0);

    //double xCoord = xMin + (xMax - xMin) * x / width;
    //double yCoord = yMin + (yMax - yMin) * y / height;
    dbldbl xCoord = add_dbldbl(xMin, div_dbldbl(mul_dbldbl(sub_dbldbl(xMax, xMin), _x), _width));
    dbldbl yCoord = add_dbldbl(yMin, div_dbldbl(mul_dbldbl(sub_dbldbl(yMax, yMin), _y), _height));

    dbldbl real = xCoord;
    dbldbl imag = yCoord;

    dbldbl real2 = mul_dbldbl(real, real);
    dbldbl imag2 = mul_dbldbl(imag, imag);

    int iterations = 0;
    //printf("Hello%d\n", iterations);
    while (get_dbldbl_head(real2) + get_dbldbl_head(imag2) < double(4.0) && iterations < maxiter) {
        //imag = 2 * real * imag + yCoord;
        //real = real2 - imag2 + xCoord;
        //real2 = real * real;
        //imag2 = imag * imag;
        imag = add_dbldbl(mul_dbldbl(mul_dbldbl(_two, real), imag), yCoord);
        real = add_dbldbl(sub_dbldbl(real2, imag2), xCoord);
        real2 = mul_dbldbl(real, real);
        imag2 = mul_dbldbl(imag, imag);
        iterations++;
    }

    int offset = y * width + x;
    //printf("%d\n", iterations);
    output[offset] = iterations;
}
void mandelbrot3(int *output, int width, int height, double cx_h, double cx_t, double cy_h, double cy_t, double scale, int maxiter) {
    //printf("AAAAA\n");
    //dim3 gridDim(1, 1, 1);
    //dim3 blockDim(width, height, 1);
    int x = int((width + 511) / 512);
    dim3 gridDim(x, height, 1);
    dim3 blockDim(512, 1, 1);

    static int * dev_data = 0;
    if (dev_data == 0)
        cudaMalloc((void**)&dev_data, width*height*sizeof(int));
    //cudaMemcpy(host_data, dev_data, width*height, cudaMemcpyHostToDevice);
    

    mandelbrot3_kernel<<<gridDim, blockDim>>>(dev_data, width, height, cx_h, cx_t, cy_h, cy_t, scale, maxiter);
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("Error: %s\n", cudaGetErrorString(error));
        //return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(output, dev_data, width*height*sizeof(int), cudaMemcpyDeviceToHost);

}

//int main() {
//    printf("BAAAA\n");
//    dim3 gridDim(4, 4096, 1);
//    dim3 blockDim(1024, 1, 1);
//    unsigned char * output;
//    int size = 1024*800;
//    cudaMalloc((void**)&output, size);
//    int width = 1024;
//    int height = 800;
//    float xMin = -2;
//    float xMax = 1;
//    float yMin = -1.5;
//    float yMax = 1.5;
//
//    //mandelbrot_kernel<<<400*400/1024, 1024>>> (output, width, height, xMin, xMax, yMin, yMax);
//    mandelbrot_kernel<<<gridDim, blockDim>>>(output, width, height, xMin, xMax, yMin, yMax);
//    cudaError_t error = cudaGetLastError();
//    if(error != cudaSuccess){
//        printf("Error: %s\n", cudaGetErrorString(error));
//        return 1;
//    }
//    cudaDeviceSynchronize();
//    //output[0,0] = 100;
//    return 0;
//}
}
