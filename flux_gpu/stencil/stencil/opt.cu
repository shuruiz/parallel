//
//  Copyright © 2018 Ethan Zhang. All rights reserved.

/////// READ BELOW FIRST///// 
/// Ethan Zhang, shuruiz@umich.edu
//====================================================================
/////////////////////////////////////////// Aditional information below!!!!!!!//////////////
/////  I did some modification on the number of threads per block dimension to do better reduce.
/////  In my report, I use <25, 25> threads per block
/////  In this script, I use a <32, 32> to do reduce to make the code more concise.
/////  The performance is the same as the result in my report. 
///// And in this modified version,  I also print out the performance directly. 

// use cuda/9.1 and this command the compile on flux: nvcc -arch=sm_35 -o stencil opt.cu -rdc=true -lcudadevrt
// =================================================




#include <iostream>
#include <algorithm>
#include "stdio.h"
#include "cmath"
#include <float.h>
#include "cuda.h"
#include <ctime>

#define THREADS_PER_DIM 32
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
        double candidates[] = {prev_dA[(i+1)*n+(j+1)], prev_dA[(i+1)*n+(j-1)],prev_dA[(i-1)*n+(j-1)],prev_dA[(i-1)*n+(j+1)]};
        dA[i*n+j] = prev_dA[i*n+j] + get2ndMin(candidates);
    }
    __syncthreads();
    
}

//parallel_2 algorithm verification
__global__ 
void reduce(double *g_idata, int step, int n, double *g_odata) {
    extern __shared__ double sdata[];
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x *blockDim.x +threadIdx.y;

    unsigned int i = (blockIdx.x * step + blockIdx.y)*(blockDim.x*blockDim.y) + tid; // global index, threads in previous blocks and 
    if(i<n*n){
        sdata[tid] = g_idata[i];
    }
    else{
        sdata[tid] = 0.0;
    }
    __syncthreads();
    // do reduction in shared mem, sum all threads in block
    for (unsigned int s=1;s<blockDim.x *blockDim.y; s*=2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x * step + blockIdx.y] = sdata[0];
}



//parallel_1 algorithm verification
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


// serial verification below
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

    int n = atoi(argv[1]);

    int N  = n*n;
    printf("size N:%d\n",N);

    double *array;
    double *sum;
    int step = ceil((double)n/THREADS_PER_DIM); //blocks
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
    dim3 dimGrid(ceil((double)n/dimBlock.x), ceil((double)n/ dimBlock.y));
    cudaEvent_t start, stop, stop1;
    float time, time1, time2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&stop1);
    

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
    cudaEventRecord(stop1, 0);

    //parallel_2 algorithm verification
    reduce<<<dimGrid,dimBlock, dimBlock.x *dimBlock.y *sizeof(double)>>>(prev_dA,step, n, g_out); 
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    
    cudaEvent_t start2, stop2;
    cudaEventCreate(&stop2);
    cudaEventCreate(&start2);
    cudaEventRecord(start2, 0);
    verification<<<1,1>>>(prev_dA,n); //  parallel_1 algorithm verification 
    cudaEventRecord(stop2, 0);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventElapsedTime(&time1, start, stop1);
    cudaEventElapsedTime(&time2, start2, stop2);


    cudaMemcpy(array,prev_dA, size, cudaMemcpyDeviceToHost);

    cudaMemcpy(sum,g_out, g_size, cudaMemcpyDeviceToHost);
    double verisum=0;
    for(int i=0; i<step*step; i++){
        verisum += sum[i];
    }
        // print result
    printf("=====================VERIFICATION========================\n");
    printf ("Time for the parallel_1 algorithm: %f ms\n", time1+time2);
    printf ("Time for the parallel_2 algorithm: %f ms\n", time);
    printf("para1 verisum %f\n", array[0]);
    printf("para2 verisum  %f\n", verisum);
    printf("verification n/2 %f\n", array[1]);
    printf("verification A[37][47] %f\n", array[2]);
    printf("=======================END OF VERIFICATION=================\n");



    //free memory
    free(array);
    free(sum);
    cudaFree(dA);
    cudaFree(prev_dA);
    cudaFree(g_out);

    return 0;
}
