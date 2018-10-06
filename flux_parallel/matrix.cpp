#include <stdlib.h>
#include <mpi.h>
#include <math.h>

using namespace std;


m=2000;
n=500;
p =1; // # of proc. 
int main(int argc, char** argv){
	
	double starttime;
	double endtime;
	int rank;
	int size;


	

	double A[m][n];
	for(i=0;i<=m;i++){
		for(j=0;j<=n;j++){
			// initialize below 
			A[i][j] = i* sin(i) +j * cos(j) + sqrt(i+j);			
		}
	}

	MPI_Init(&argc, &argv);

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// do 10 iteration below
	starttime = MPI_Wtime();
	double A_0[m][n];
	for(t=0; t<10; i++ ){
		A_0 = A;

		block = ceil(n*1.0/p);

		tail = min(block*(rank+1),(n%p));
		for(i=0;i<m;i++){
			for(j = rank*block; j< tail; j++){
				if(i==0 || i ==m-1 || j==0||j==n-1){
				A[i][j] = A_0 [i][j];  //unchanged along boarder
				}

				else{
					z = (f(A_0[i-1][j])+f(A_0[i+1][j])+f(A_0[i][j-1]) + f(A_0[i][j+1]) + f(A_0[i][j])) / 5;
					A[i][j] = max(-100, min(100, z)); 
				}
			}
		}
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