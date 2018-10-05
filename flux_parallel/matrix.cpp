#include <stdlib.h>
#include <mpi.h>
#include <math.h>

using namespace std;


m=2000;
n=500;
p =1; // # of proc. 
int main(int argc, char** argv){
	
	MPI_BARRIER(0)
	double starttime, endtime;
	starttime = MPI_Wtime();
	MPT_INITE()
	double A[m][n];
	for(i=0;i<=m;i++){
		for(j=0;j<=n;j++){
			// initialize below 
			A[i][j] = i* sin(i) +j * cos(j) + sqrt(i+j);			
		}

	// iteration 
	}

/*
	double A_0[m][n];

	for(i=0; i<=m; i++){
		for(j=0;j<=n;j++){
			A_0 = A;
			if(i==0 || i ==m-1 || j==0||j==n-1){
				A[i][j] = A_0 [i][j];  //unchanged along boarder
			}

			else{
				z = (f(A_0[i-1][j])+f(A_0[i+1][j])+f(A_0[i][j-1]) + f(A_0[i][j+1]) + f(A_0[i][j])) / 5;
				A[i][j] = max(-100, min(100, z)); 
			}
		}
	}

	*/ 

	endtime = MPI_Wtime();

}

double f(double x){
	double y;
	int i;

	y = x; 
	for(i=1; i<=10; 1++){
		y = y+sin(x*i)/(2.0^i);
	}

	return y;

}