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
double get2ndMin(double *candidates){
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

// __global__ 
// void compute_verification(double *A, int n){
    
//     double v1,v2,v3;
//     v1 = 0.0;
//     for(int i = 0 ; i < n*n ; i++){
//         v1 += A[i];
//     }
//     v2 = A[ (int) ( n * floor(n/2.0) + floor(n/2.0) )  ];
//     v3 = A[n * 37 + 47];
//     A[0] = v1;
//     A[1] = v2;
//     A[2] = v3;
// }


__global__ 
void calc(int n, double **dA, double **prev_dA){

    int gindex_y = threadIdx.y + blockIdx.y * blockDim.y; 
    int gindex_x = threadIdx.x + blockIdx.x * blockDim.x;

    if(gindex_x ==0 || gindex_x ==n-1 || gindex_y ==0 || gindex_y ==n-1){
        dA[gindex_x][gindex_y] = prev_dA[gindex_x][gindex_y];
    }else{
        // tmp[lindex_x-1][lindex_y-1] = A[gindex_x-1][gindex_y-1]
        double candidates[] = {prev_dA[gindex_x+1][gindex_y+1], prev_dA[gindex_x+1][gindex_y-1],prev_dA[gindex_x-1][gindex_y-1],prev_dA[gindex_x-1][gindex_y+1]};
        dA[gindex_x][gindex_y] = prev_dA[gindex_x][gindex_y] + get2ndMin(candidates);
    }
    __syncthreads();
    printf("exec. in block%d, threads%d, i%d, j%d, \n", blockIdx.x, threadIdx.x, gindex_x, gindex_y);
}

//parent node
// __global__ void stencil(double *dA,int n){

//     calc<<<BLOCKS, THREADS_PER_DIM>>>(n, dA); 
//     __syncthreads();
//     printf("exec. in parent node\n");
// }

__global__
void verification(double **A, double *result, int n){
    double v1,v2,v3;
    v1 = 0.0;
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            v1 += A[i][j];
        }
    }

    int fl = floor((double)n/2);
    v2 = A[fl][fl];
    v3 = A[37][47];
    result[0] = v1; 
    result[1] = v2;
    result[2] = v3; 

}

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
    double *result; 
    int size = (N) * sizeof(double);
    array =(double **)malloc(size);

    int size_result = 3* sizeof(double);
    result = (double *)malloc(size_result);

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
    double **prev_dA;
    double *d_result;
    // allocate memory on device
    cudaMalloc((void **)&dA, size);
    cudaMalloc((void **)&prev_dA, size);
    cudaMalloc((void **)&d_result, size_result);

    // Copy inputs to device
    cudaMemcpy(dA, array, size, cudaMemcpyHostToDevice);
    cudaMemcpy(prev_dA, array, size, cudaMemcpyHostToDevice);


    //launch kernal on device
    int t  = 10;
    dim3 dimBlock(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 dimGrid(n/THREADS_PER_DIM, n/ THREADS_PER_DIM);
    
    for(int episode =0; episode<t; episode++){
        printf("loop %d\n", episode );
        calc<<<dimGrid, dimBlock>>>(n, dA, prev_dA);
        cudaDeviceSynchronize();

        double **tem_a = dA;
        dA = prev_dA;
        prev_dA = tem_a;  
    }

    verification<<<1,1>>>(prev_dA, d_result, n);

    cudaMemcpy(result,d_result, size_result, cudaMemcpyDeviceToHost);

    //print result
    printf("verisum all %f\n", result[0]);
    printf("verification n/2 %f\n", result[1]);
    printf("verification A[37][47] %f\n", result[2]);
    
    //free memory
    free(array);
    free(result);
    cudaFree(dA);
    cudaFree(prev_dA);
    cudaFree(d_result);
    return 0;
}
