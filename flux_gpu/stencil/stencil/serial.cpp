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


double* verification(double **A, int n){
    double v1, v2,v3;
    double *result;
    v1 = 0.0;
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            v1 += A[i][j];
        }
    }

    int fl = floor((double)n/2);
    v2 = A[fl][fl];
    v3 = A[37][47];
    // __syncthreads();
    result[0] = v1;
    result[1] = v2;
    result[2] = v3; 

    return result;
}


int main(int argc, char** argv){
	int n = atoi(argv[1]);
	static double** array;
	// double ** array = new double*[n]; 
	// for(int i=0; i<n; i++){array[i] = new double[n]; }
    for(int i =0; i<n;i++){
        for(int j =0; j<n; j++){
            array[i][j] = pow(1+cos(2*i)+sin(j),2);
        }
    }

    double starttime = time(0);

    int t  = 10; 
    
    static double **tmp = array;

    for(int e = 0; e<t; e++){ // 10 iteration
    	for(int i = 0; i<n; i++){
    		for(int j=0; j<n; j++){

    			if(i==0 || i==n-1 || j==0 || j==n-1){
    				array[i][j] = tmp[i][j];
    			}
    			double candidates[] ={array[i+1][j+1], array[i+1][j-1], array[i-1][j-1], array[i-1][j+1]};

    			array[i][j] = tmp[i][j] + get2ndMin(candidates);
    		}
    	}

    	double **xx = array;
    	tmp = array;
    	array = xx;
    }

    double *result;
    result =(double *)malloc(3*sizeof(double));
    result = verification(array, n);
    double endtime = time(0);


    printf ("Time for the kernel: %f s\n", (starttime - endtime));
    printf("verisum all %f\n", result[0]);
    printf("verification n/2 %f\n", result[1]);
    printf("verification A[37][47] %f\n", result[2]);

    free(result);
}