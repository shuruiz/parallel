//
//  main.cpp
//  using CUDA to do parallel computing of stencil
//  do t repeated run
//  do it for n = 500, 1000, 20000 and  t = 10
//  Created by Ethan Zhang on 11/8/18.
//  Copyright © 2018 Ethan Zhang. All rights reserved.
//


#include <iostream>
#include <algorithm>
#include "stdio.h"
#include <vector>
#include "cmath"
#include <float.h>
#include "cuda.h"

#define THREADS_PER_BLOCK 256
#define PARENT_THREADS 128
#define N_BLOCKS 16

using namespace std;


//child node
__global__
void calc(int n, double *A){
    
    __shared__ double tmp[blockDim.x+2*n]; //radius =n
    int gindex = threadId.x + blockIdx.x *blockDim.x;
    int lindex = threadId.x + n ;
    //read input elements into shared memory
    tmp[lindx] = A[gindex];
    if(threadId.x < n){
        tmp[lindex-n] = A[gindex -n];
        //block size = threads per block
        tmp[lindex + THREADS_PER_BLOCK ] = A [gindex+ THREADS_PER_BLOCK];
    }
    __syncthreads();
    
    //update A below
    double first, second;
    first = second = DBL_MAX;
    int i = floor(gindex / n);
    int j = gindex % n;
    if(i ==0 || i ==n-1 || j ==0 || j ==n-1){ // unchanged, do nothing
        A[i*n+j] = prev_A[i*n+j];
    }
    //find secondMin below
    else{
        double candidates[] = {tmp[(i+1)*n+ (j+1)], tmp[(i+1)*n+(j-1)],tmp[(i-1)*n +(j+1)],tmp[(i-1)*n + (j-1)]};
        for(int k =0; k<4; k++){
            if(candidates[k]<first){
                second = first;
                first = candidates[k];
            }
            else if (candidates[k] < second && candidates[k] != first){
                second = candidates[k];}
        }
        A[i*n+j] += second;
    }
}


//parent node
__global__
void stencil(double *dA,int n, int t){
    for(int episode = 0; episode <t; episode++){
        int N = n*n;
        calc<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(n, dA);
        __syncthreads();
    }
}

double verisum_all(int n, double *A){
    double sum=0.0;
    for(int i = 0; i<n; i++){
        for(int j=0; j<n; j++){
            sum += A[i*n+ j];
        }
    }
    return sum;
}

double value_half(int n, double *A){
    int fl = floor(n/2);
    return A[fl * n + fl];
}

double value_37_47(int n, double *A){
    return A[37*n + 47];
}


int main(int argc, char** argv) {
    // initialize below
    int n = *argv[1];
    int N  = n*n;
    
//2d stencil, represented by 1d stencil
    // initialize below
    double *array;
    int size = N * sizeof(double);
    array =(double *)malloc(N);
    for(int i =0; i<n;i++){
        for(int j =0; j<n; j++){
            array[i*n+j] = pow(1+cos(2*i)+sin(j),2);
        }
    }
    double *dA;
    // allocate memory on device
    cudaMalloc((void **)&dA, size);

    // Copy inputs to device
    cudaMemcpy(dA, array, size, cudaMemcpyHostToDevice);
    //launch kernal on device
    int t  = 10;
    
    stencil<<<1, PARENT_THREADS>>>(dA, n, t);
    cudaDeviceSynchronize();
    
    
    // Copy result back to host
    cudaMemcpy(array, dA, size, cudaMemcpyDeviceToHost);
    
    //verify results
    double verisum = verisum_all(n, array);
    double half_value = value_half(n, array);
    double spec  = value_37_47(n, array);
    
    //print result
    printf("verisum all %f/n", verisum);
    printf("verisum n/2 %f/n", half_value);
    printf("verisum A[37][47] %f/n", spec);
    
    
    //free memory
    free(array);
    cudaFree(dA); cudaFree(prev_dA);
  
    return 0;
}
