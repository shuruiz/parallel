// dynamic reduction on weights mapping 

// calculate neural weights in real time. 
// serial in 50 mins, pytorch 5 mins 
// parallel target, solve in less than 10 ms - real time 

#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <ctime>
#include <cuda.h>
#include <float.h>

#define THREADS_PER_DIM 32
using namespace std;


__global__
void dynamic(double *d_A, int *d_B, double *d_C){

}

int main(int argc, char** argv){
	int m= atoi(argv[1]);
	// int m =1000000;
	//n = atoi(argv[2]);
	double *A,*C;
	int *B; // index
	int size_a = m*sizeof(double);
	int size_b = m*sizeof(int);

	A = (double *)malloc(size_a);
	B = (int *)malloc(size_b)


	// init below 
	for(int i =0; i<m; i++){
		A[i] = rand()%10000;
		B[i] = rand()%450000;
	}

	int len_c = *std::max_element(B,B+m);
	int size_c =  len_c *sizeof(double);
	C = (double *)malloc(size_c);

	// clock_t startTime = clock();
	for(int j=0; j<len_c;j++){
		C[j]=0; // init C
	}
	double *dA, *dC;
	int *dB; 

	// allocate memory on device
    cudaMalloc((void **)&dA, size_a);
    cudaMalloc((void **)&dB, size_b);
    cudaMalloc((void **)&dC, size_c);
    // Copy inputs to device
    cudaMemcpy(dA, A, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size_b, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, size_c, cudaMemcpyHostToDevice);

    int n_ele = m;

    dim3 dimBlock(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 dimGrid(ceil((double)n_ele/dimBlock.x), ceil((double)n_ele/ dimBlock.y));

    //timer 
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);    
    cudaEventRecord(start, 0);

    // launch kernal on GPU
    dynamic<<<dimGrid,dimBlock>>>(dA,dB,dC); 


    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(C,dC, size_c, cudaMemcpyDeviceToHost);
    printf("runtime for parallel algorithm:%f ms\n", time);
	free(A);
	free(B);
	free(C);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
	return 0;


}