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

#define THREADS_PER_DIM 20
// #define TASKS_PER_THREADS 50
// #define BLOCKS 32
// #define N 1000*1000
// #define RADIUS  1001
// #define TASKS 
using namespace std;

__device__
double get2ndMin(double *tmp){
    double first, second;
    first = second =DBL_MAX;
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
//child node
__global__ 
void calc(int n, double **dA, double **prev_dA){
    // __shared__ double tmp[THREADS_PER_DIM  + 2 ][THREADS_PER_DIM  + 2 ]; //SM, above cells
    int gindex_y = threadIdx.y + blockIdx.y * blockDim.y; 
    int gindex_x = threadIdx.x + blockIdx.x * blockDim.x;

    int lindex_x = threadIdx.x +1;
    int lindex_y = threadIdx.y +1; 
    // tmp[lindex_x][lindex_y] = A[gindex_x][gindex_y];
    if(gindex_x ==0 || gindex_x ==n-1 || gindex_y ==0 || gindex_y ==n-1){
        // do nothing
    }else{
        // tmp[lindex_x-1][lindex_y-1] = A[gindex_x-1][gindex_y-1]
        double candidates[] = {prev_dA[gindex_x+1][gindex_y+1], prev_dA[gindex_x+1][gindex_y-1],prev_dA[gindex_x-1][gindex_y-1],prev_dA[gindex_x-1][gindex_y+1]};
        dA[gindex_x][gindex_y] += get2ndMin(candidates);
    }
    __syncthreads();
    printf("exec. in block%d, threads%d, i%d, j%d, \n", blockIdx.x, threadIdx.x, i, j);
    
}

//parent node
// __global__ void stencil(double *dA,int n){

//     calc<<<BLOCKS, THREADS_PER_DIM>>>(n, dA); 
//     __syncthreads();
//     printf("exec. in parent node\n");
// }

double verisum_all(int n, double **A){
    double sum=0.0;
    for(int i = 0; i<n; i++){
        for(int j=0; j<n; j++){
            sum += A[i][j];
        }
    }
    return sum;
}

double value_half(int n, double **A){
    int fl = floor((double)n/2);
    double result  = A[fl][fl];
    return result;
}

double value_37_47(int n, double **A){
    double result =A[37][47];
    return result;
}


int main(int argc, char** argv) {
    // initialize below
    int n = *argv[1];

    int N  = n*n;
    printf("size N%d\n",N);
//2d stencil, represented by 1d stencil
    // initialize below
    double **array;
    int size = (N) * sizeof(double);
    array =(double *)malloc(size);

    for(int i =0; i<n;i++){
        for(int j =0; j<n; j++){
            array[i][j] = pow(1+cos(2*i)+sin(j),2);
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

    double **dA;
    double **prev_dA
    // allocate memory on device
    cudaMalloc((void **)&dA, size);
    cudaMalloc((void **)&prev_dA, size);
    // Copy inputs to device
    cudaMemcpy(dA, array, size, cudaMemcpyHostToDevice);
    //launch kernal on device
    int t  = 10;
    dim3 dimBlock(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 dimGrid(n/THREADS_PER_DIM, n/ THREADS_PER_DIM);
    
    for(int episode =0; episode<t; episode++){
        printf("loop %d\n", episode );
        calc<<<dimGrid, dimBlock>>>(n, dA, prev_dA);
        cudaDeviceSynchronize();
        prev_dA = dA;   
    }
    cudaMemcpy(array,dA, size, cudaMemcpyDeviceToHost);
    // Copy result back to host
    //verify results
    double verisum = verisum_all(n, array);
    double half_value = value_half(n, array);
    double spec  = value_37_47(n, array);
    
    //print result
    printf("verisum all %f\n", verisum);
    printf("verification n/2 %f\n", half_value);
    printf("verification A[37][47] %f\n", spec);
    
    //free memory
    free(array);
    cudaFree(dA);
    cudaFree(prev_dA);
    return 0;
}
