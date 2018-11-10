//
//  main.cpp
//  using CUDA to do parallel computing of stencil
//
//  Created by Ethan Zhang on 11/8/18.
//  Copyright © 2018 Ethan Zhang. All rights reserved.
//


#include <iostream>
#include <algorithm>
#include "stdio.h"
#include <vector>
#include "cmath"
#include <float.h>

#define THREADS_PER_BLOCK 512

using namespace std;
void calc(int i, int j, int n, double **arr){
    if(i ==0 || i ==n-1 || j ==0 || j ==n-1){ // unchanged, do nothing
    }
    else{
        //assign secondMin to A[i][j]
        double first,second;
        first = second = DBL_MAX;
    }
    
}

__global__ void stencil_2d(double **arr, int n) {
    int index_i = floor(threadIdx.x + blockIdx.x * THREADS_PER_BLOCK / (n*1.0));
    int index_j = floor(threadIdx.x + blockIdx.x * THREADS_PER_BLOCK % n);
    calc(index_i, index_j, n, arr);
    
    
}



int main(int argc, char** argv) {
    // initialize below
    int n = *argv[1];
    std::vector<double> v1(n);
    std::vector<std::vector<double> > array(n,v1);
    for(int i =0; i<array.size();i++){
        for(int j =0; j<array[i].size(); j++){
            array[i][j] = pow(1+cos(2*i)+sin(j),2);
        }
    }
    // variables
    double *a, *b, *c;
    double *d_a, *d_b, *d_c;
    int N  = n*n;
    int size = N * sizeof(double);
    
    // allocate memory on device
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    
    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    //launch kernal on device
    stencil_2d<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    
    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  
    return 0;
}
