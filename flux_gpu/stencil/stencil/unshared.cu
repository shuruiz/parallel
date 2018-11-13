//
//  main.cpp
//  GPU_CUDA
//
//  Created by xietiany on 07/11/2018.
//  Copyright © 2018 xietiany. All rights reserved.
//

#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <vector>
#include <cuda.h>
#include <ctime>
using namespace std;

typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

// Set a matrix element

__device__ void SetElement(Matrix A, int row, int col, float value)
{
    A.elements[row * A.stride + col] = value;
}

__device__ float findsecMin(float A, float B, float C, float D) {
    float min = A;
    float smin = B;
    if (smin < min) {
        float temp = min;
        min = smin;
        smin = temp;
    }
    if (C <= min) {
        smin = min;
        min = C;
    }
    else if(C > min && C < smin) {
        smin = C;
    }
    
    if (D <= min) {
        smin = min;
        min = D;
    }
    else if(D > min && D < smin) {
        smin = D;
    }
    
    return smin;
}

Matrix init_matrix(int n);

#define BLOCK_SIZE 10
#define matrix_size 1000

__global__ void MatrixUpdate(Matrix d_M1, Matrix d_store);

int main(int argc, const char * argv[]) {
    // insert code here...
    cout << "------------------------Program start------------------------" << endl;
    int junk = atof(argv[1]);
    
    //init matrix;
    
    
    Matrix init = init_matrix(matrix_size);
    
    
    Matrix d_M1;
    d_M1.width = d_M1.stride = matrix_size;
    d_M1.height = matrix_size;
    d_M1.elements = (float*) malloc(matrix_size * matrix_size * sizeof(float));
    
    Matrix d_store;
    d_store.width = d_store.stride = matrix_size;
    d_store.height = matrix_size;
    d_store.elements = (float*) malloc(matrix_size * matrix_size * sizeof(float));
    size_t size = matrix_size * matrix_size * sizeof(float);
    
    cout << "here1" << endl;
    
    cudaMalloc(&d_M1.elements, size);
    cudaMalloc(&d_store.elements, size);
    cudaMemcpy(d_M1.elements, init.elements, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_store.elements, init.elements, size, cudaMemcpyHostToDevice);
    
    cout << "here2" << endl;
    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(matrix_size / BLOCK_SIZE, matrix_size / BLOCK_SIZE);

    cout << "here3" << endl;
    clock_t t; 
    t = clock();
    for (int i = 0; i < 10; i++) {
        MatrixUpdate<<<dimGrid, dimBlock>>>(d_M1, d_store);
    }
    t = clock()-t;
    printf("total time %f\n", ((float)t)/CLOCKS_PER_SEC);
    cout << "here4" << endl;
    
    Matrix res;
    res.width = res.stride = matrix_size;
    res.height = matrix_size;
    res.elements = (float*) malloc(matrix_size * matrix_size * sizeof(float));
    cudaMemcpy(res.elements, d_M1.elements, size, cudaMemcpyDeviceToHost);
    
    //free cuda memory
    cudaFree(d_M1.elements);
    cudaFree(d_store.elements);
    
    float sum = 0;
    
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            sum += res.elements[res.width * i + j];
        }
    }
    
    int pos = matrix_size / 2;
    
    cout << "the sum is " << sum << endl;
    cout << "the special value is " << res.elements[res.width * 37 + 47] << endl;
    cout << "n/2 value is " << res.elements[res.width * pos + pos] << endl;
    
    return 0;
}

__global__ void MatrixUpdate(Matrix d_M1, Matrix d_store) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    //float center = d_store.elements[d_store.width * row + col];
    float plus = 0;
    
    if (row > 0 && row < matrix_size - 1 && col > 0 && col < matrix_size - 1) {
        float up = d_store.elements[d_store.width * (row - 1) + col];
        float down = d_store.elements[d_store.width * (row + 1) + col];
        float left = d_store.elements[d_store.width * row + col - 1];
        float right = d_store.elements[d_store.width * row + col + 1];
        
        plus = findsecMin(up, down, left, right);
        
    }
    
    //center += plus;
    //SetElement(d_M1, row, col, center);
    d_M1.elements[d_M1.width * row + col] += plus;
    
    __syncthreads();
    
    float value = d_M1.elements[d_M1.width * row + col];
    SetElement(d_store, row, col, value);
    
    __syncthreads();
}

Matrix init_matrix(int n) {
    Matrix res;
    res.width = res.stride = res.height = n;
    res.elements = (float*) malloc(n * n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            //cout << i << " " << j << " " << i * res.width + j << endl;
            float value = pow((1 + cos(2 * i) + sin(j)), 2);
            //cout << "here1" << endl;
            res.elements[i * res.width + j] = value;
        }
    }
    
    return res;
}


