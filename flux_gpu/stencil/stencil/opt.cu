//
//  stencil main program
//  using CUDA to do parallel computing of stencil
//  do t repeated run
//  do it for n = 500, 1000, 20000 and  t = 10
//  Created by Ethan Zhang on 11/8/18.
//  Copyright © 2018 Ethan Zhang. All rights reserved.
//


#include <iostream>
#include <algorithm>
#include "stdio.h"
#include "cmath"
#include <float.h>
#include "cuda.h"
#include <ctime>

#define THREADS_PER_DIM 25
// #define TASKS_PER_THREADS 50
// #define BLOCKS 32
// #define N 1000*1000
// #define RADIUS  1001
// #define TASKS 
using namespace std;

__device__
double get2ndMin(double *candidates){
    double first, second;
    first = second = DBL_MAX;
    for(int k =0; k<4; k++){
        if(candidates[k]<=first){
            second = first;
            first = candidates[k];
        }
        else if (candidates[k] <= second && candidates[k] >= first){
            second = candidates[k];}
    }
    return second;
}


__global__ 
void calc(int n, double *dA, double *prev_dA){

    int j = threadIdx.y + blockIdx.y * blockDim.y; 
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i ==0 || i ==n-1 || j ==0 || j ==n-1){
        dA[i*n+j] = prev_dA[i*n+j];
    }else{
        // tmp[lindex_x-1][lindex_y-1] = A[i-1][j-1]
        double candidates[] = {prev_dA[(i+1)*n+(j+1)], prev_dA[(i+1)*n+(j-1)],prev_dA[(i-1)*n+(j-1)],prev_dA[(i-1)*n+(j+1)]};
        dA[i*n+j] = prev_dA[i*n+j] + get2ndMin(candidates);
    }
    __syncthreads();
    // printf("exec. in block%d, threads%d, i%d, j%d, \n", blockIdx.x, threadIdx.x, i, j);
}

//parent node
// __global__ void stencil(double *dA,int n){

//     calc<<<BLOCKS, THREADS_PER_DIM>>>(n, dA); 
//     __syncthreads();
//     printf("exec. in parent node\n");
// }

__global__ void reduce(double *g_idata, double *g_odata) {
    extern __shared__ double sdata[];
    // each thread loads one element from global to shared mem
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x *blockDim.x +threadIdx.y;

    unsigned int i = (blockIdx.x *blockDim.x + blockIdx.y)*(blockDim.x*2) + tid; // global index, threads in previous blocks and 
    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s=1;s<blockDim.x *blockDim.y; s++) {
        if (tid < s) {
            sdata[tid] += sdata[tid + 1];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x *blockDim.x + blockIdx.y] = sdata[0];
}




__global__
void verification(double *A, int n){
    double v1, v2,v3;
    v1 = 0.0;
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            v1 += A[i*n+j];
        }
    }

    int fl = floor((double)n/2);
    v2 = A[fl*n+fl];
    v3 = A[37*n+47];
    // __syncthreads();
    A[0] = v1;
    A[1] = v2;
    A[2] = v3; 
}

double verisum_all(int n, double *A){
    double sum=0.0;
    for(int i = 0; i<n; i++){
        for(int j=0; j<n; j++){
            sum += A[i*n+j];
        }
    }
    return sum;
}

double value_half(int n, double *A){
    int fl = floor((double)n/2);
    double result  = A[fl*n+ fl];
    return result;
}

double value_37_47(int n, double *A){
    double result =A[37*n+47];
    return result;
}

int main(int argc, char** argv) {
    // initialize below
    int n = atoi(argv[1]);

    int N  = n*n;
    printf("size N%d\n",N);

    // initialize below
    // 1d stencil

    double *array;
    double *sum;
    int step = n/THREADS_PER_DIM; 
    int size = (N) * sizeof(double);

    int g_size = (step*step) * sizeof(double); 
    array =(double *)malloc(size);
    sum = (double *)malloc(g_size);

    for(int i =0; i<n;i++){
        for(int j =0; j<n; j++){
            array[i*n+j] = pow(1+cos(2*i)+sin(j),2);
        }
    }

    for(int i=0; i<step; i++){
        for(int j=0; j<step; j++){
            sum[i*step+j] =0.0;
        }
    }


    //verify initialization results
    double verisum_1 = verisum_all(n, array);
    double half_value_1 = value_half(n, array);
    double spec_1  = value_37_47(n, array);
    
    //print result
    printf("init verisum all %f\n", verisum_1);
    printf("init verification n/2 %f\n", half_value_1);
    printf("init verification A[37][47] %f\n", spec_1);

    double *dA;
    double *prev_dA;
    double *g_out;
    
    // allocate memory on device
    cudaMalloc((void **)&dA, size);
    cudaMalloc((void **)&prev_dA, size);
    cudaMalloc((void **)&g_out, g_size);

    // Copy inputs to device
    cudaMemcpy(dA, array, size, cudaMemcpyHostToDevice);
    cudaMemcpy(prev_dA, array, size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_out, sum, g_size, cudaMemcpyHostToDevice);

    //launch kernal on device
    int t  = 10;
    dim3 dimBlock(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 dimGrid(n/THREADS_PER_DIM, n/ THREADS_PER_DIM);
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // double v1 =0.0; 
    
    cudaEventRecord(start, 0);

    for(int episode =0; episode<t; episode++){
        // printf("loop %d\n", episode );
        calc<<<dimGrid, dimBlock>>>(n, dA, prev_dA);
        cudaDeviceSynchronize();

        double *tem_a = dA;
        dA = prev_dA;
        prev_dA = tem_a;  
    }
    

    // verification<<<1,1>>>(prev_dA,n); //  para1 verification 
    
    cudaMemcpy(array,prev_dA, size, cudaMemcpyDeviceToHost);
    
    reduce<<<dimGrid,dimBlock, dimBlock.x *dimBlock.y *sizeof(double)>>>(prev_dA,g_out); //better verification
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&time, start, stop);

    cudaMemcpy(sum,g_out, g_size, cudaMemcpyDeviceToHost);
    double verisum=0;
    for(int i=0; i<step*step; i++){
        verisum += sum[i];
    }
        // print result
    printf ("Time for the kernel: %f ms\n", time);
    printf("verisum all %f\n", verisum);
    printf("verification n/2 %f\n", array[1]);
    printf("verification A[37][47] %f\n", array[2]);





    //free memory
    free(array);
    // free(sum);
    cudaFree(dA);
    cudaFree(prev_dA);
    // cudaFree(g_out);

    return 0;
}
