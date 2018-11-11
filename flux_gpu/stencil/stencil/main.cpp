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

using namespace std;

__global__
void calc(int i, int j, int n, double *A, double *prev_A){
    double first,second;
    first = second = DBL_MAX;
    if(i ==0 || i ==n-1 || j ==0 || j ==n-1){ // unchanged, do nothing
        A[i*n+j] = prev_A[i*n+j];
    }
    else{
        //assign secondMin to A[i][j]
        double tmp[4] = {arr[i+1][j+1], arr[i+1][j-1],arr[i-1][j+1],arr[i-1][j-1]};
        for(int k =0; k<4; k++){
            if(tmp[k]<first){
                second = first;
                first = tmp[k];
            }
            else if (tmp[k] < second && tmp[k] != first){
                second = tmp[k];}
        }
    }
    arr[i][j] += second;
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

__global__
void stencil(double *A, double *prev_dA, int n) {
    int index = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
    int i = floor(index / n);
    int j = index % n;
    
    double first, second;
    first = second = DBL_MAX;
    if(i ==0 || i ==n-1 || j ==0 || j ==n-1){ // unchanged, do nothing
        A[i*n+j] = prev_A[i*n+j];
    }
    else{
        double tmp[] = {A[(i+1)*n+ (j+1)], A[(i+1)*n+(j-1)],A[(i-1)*n +(j+1)],A[(i-1)*n + (j-1)]};
        for(int k =0; k<4; k++){
            if(tmp[k]<first){
                second = first;
                first = tmp[k];
            }
            else if (tmp[k] < second && tmp[k] != first){
                second = tmp[k];}
        }
        A[i*n+j] += second;
    }
    prev_dA = A
}

void compute(double *dA, double *prev_dA, int n, int t){
    for(int episode = 0; episode <t; episode++){
        int N = n*n;
        stencil<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(dA, prev_dA, n);
    }
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
    
    
    // variables
//    double *a, *b, *c, *d;
//    double *d_a, *d_b, *d_c, *d_d;
    
    
    double *dA;
    double *prev_dA;  // time t-1 matrix
    // allocate memory on device
    cudaMalloc((void **)&dA, size);
    cudaMalloc((void **)&prev_dA, size);
    
    // Copy inputs to device
    cudaMemcpy(dA, array, size, cudaMemcpyHostToDevice);
//    cudaMemcpy(prev_dA, array, size, cudaMemcpyHostToDevice);
    
    //launch kernal on device
//    stencil<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a,d_b, d_c);
    int t  = 10;
    compute(dA, prev_dA, n, t)
    
    // Copy result back to host
    cudaMemcpy(array, dA, size, cudaMemcpyDeviceToHost);
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
