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

#define THREADS_PER_BLOCK 32
#define BLOCKS 16
#define PARENT_THREADS 32
#define RADIUS  1000

using namespace std;


//child node
__global__ void calc(int n, double *A){
    // const int RADIUS = n; 
    __shared__ double tmp[THREADS_PER_BLOCK + 2 * RADIUS]; //radius =n
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + n ; //local index + radius

    //read input elements into shared memory
    tmp[lindex] = A[gindex];
    if(threadIdx.x < n && gindex >n && gindex < n*n){ // the first row doesn't cal
        tmp[lindex-n] = A[gindex -n];
        //block size = threads per block
        tmp[lindex + THREADS_PER_BLOCK ] = A [gindex+ THREADS_PER_BLOCK];
    }
    __syncthreads();

    //update A below
    double first, second;
    first = second = DBL_MAX;
    int i = floor((double)gindex / n);
    int j = gindex % n;
    if(i ==0 || i ==n-1 || j ==0 || j ==n-1){ // unchanged, do nothing
    }
    //find secondMin below
    else{
        double candidates[] = {tmp[(i+1)*n+ (j+1)], tmp[(i+1)*n+(j-1)],tmp[(i-1)*n +(j+1)],tmp[(i-1)*n + (j-1)]};
        for(int k =0; k<4; k++){
            if(candidates[k]<=first){
                second = first;
                first = candidates[k];
            }
            else if (candidates[k] <= second && candidates[k] != first){
                second = candidates[k];}
        }
        A[i*n+j] += second;
    }

    printf("exec. in child node, block%d, threads%d\n", blockIdx.x, threadIdx.x);
}

//parent node
__global__ void stencil(double *dA,int n){

    calc<<<BLOCKS, THREADS_PER_BLOCK>>>(n, dA); 
    __syncthreads();
    printf("exec. in parent node\n");
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
    int fl = floor((double)n/2);
    double result  = A[fl * n + fl];
    return result;
}

double value_37_47(int n, double *A){
    double result =A[37*n + 47];
    return result;
}


int main(int argc, char** argv) {
    // initialize below
    int n = *argv[1];
    int N  = n*n;
    
//2d stencil, represented by 1d stencil
    // initialize below
    double *array;
    int size = (N) * sizeof(double);
    array =(double *)malloc(size);

    for(int i =0; i<n;i++){
        for(int j =0; j<n; j++){
            array[i*n+j] = pow(1+cos(2*i)+sin(j),2);
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
    // allocate memory on device
    cudaMalloc((void **)&dA, size);

    // Copy inputs to device
    cudaMemcpy(dA, array, size, cudaMemcpyHostToDevice);
    //launch kernal on device
    int t  = 10;
    
    for(int episode =0; episode<t; episode++){
        printf("loop %d\n", episode );
        stencil<<<1, PARENT_THREADS>>>(dA, n);
        cudaDeviceSynchronize();
    }
    
    
    // Copy result back to host
    cudaMemcpy(array,dA, size, cudaMemcpyDeviceToHost);

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
    return 0;
}
