#include <iostream>
#include <ctime>
#include <stdlib.h>
#include <cmath>
#include <float.h>


using namespace std;
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


int main(int argc, char** argv){
	int n = atoi(argv[1]);
	int N = n*n;

	int size  = N*sizeof(double);

	double *A;
	double *prev_A;
	A = (double *) malloc(size);
	prev_A = (double *) malloc(size);
    for(int i =0; i<n;i++){
        for(int j =0; j<n; j++){
            A[i*n+j] = pow(1+cos(2*i)+sin(j),2);
            prev_A[i*n+j] = pow(1+cos(2*i)+sin(j),2);
        }
    }

    double starttime = time(0);

    int t  = 10; 
    
    for(int e = 0; e<t; e++){ // 10 iteration
    	for(int i = 0; i<n; i++){
    		for(int j=0; j<n; j++){
    			if(i==0 || i==n-1 || j==0 || j==n-1){
    				A[i*n+j] = prev_A[i*n+ j];
    			}
    			double candidates[] ={A[(i+1)*n+ (j+1)], A[(i+1)*n +(j-1)], A[(i-1)*n + j-1], A[(i-1)*n + j+1]};

    			A[i*n+j] = prev_A[i*n+j] + get2ndMin(candidates);
    		}
    		// printf("i %d\n", i);

    	}

    	
    	double *xx = A;
    	A = prev_A;
    	prev_A = xx;
    }
    // verification(prev_A, n);
    double v1, v2,v3;
    v1 = 0.0;
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            v1 += prev_A[i*n+j];
        }
    }

    int fl = floor((double)n/2);
    v2 = prev_A[fl*n+fl];
    v3 = prev_A[37*n+47];
    // __syncthreads();
    prev_A[0] = v1;
    prev_A[1] = v2;
    prev_A[2] = v3; 


    double endtime = time(0);

    printf ("Time for the kernel: %f s\n", (starttime - endtime));
    printf("verisum all %f\n", prev_A[0]);
    printf("verification n/2 %f\n", prev_A[1]);
    printf("verification A[37][47] %f\n", prev_A[2]);

}