//
//  main.cpp
//  using CUDA to do parallel computing of stencil
//
//  Created by Ethan Zhang on 11/8/18.
//  Copyright © 2018 Ethan Zhang. All rights reserved.
//


#include <iostream>
#include <algorithm>
#include <vector>
#include "cmath"
using namespace std;
#define N (1024*1024)
#define THREADS_PER_BLOCK 512


__global__ void stencil_1d(double *a, double *b, double *c, int n) {
    
}


int main(int argc, char** argv) {
    std::vector<std::vector<double> > array;
    // initialize below
    int n = *argv[1];
    double array(n,n);
    
    for(int i =0; i<n;i++){
        for(int j =0; j<n; j++){
            array[i][j] = pow(1+cos(2*i)+sin(j),2);
        }
    }
    
    double *a, *b, *c;
    double *d_a, *d_b, *d_c;
    int size = N * sizeof(int);
    
    // allocate memory on device
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_a, size);
    
    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    stencil_1d<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    
    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  
    return 0;
}
