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


	if(rank==p-1){
		b = n-floor(n*1.0/p)*rank;
	}else{
		b = floor(n*1.0/p);
	}

	n_row = b+2; // append one row above and one row below the target matrix, ghost cells
	double A[m][n_row];  

	for(i=0;i<m;i++){
		for(j=1;j<n_row-1;j++){
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
		loop = ceil(n*1.0/p);
		if(rank = p-1){
			end = n;
		}
		else{
			end = rank*(loop+1);
		}

		// calcalate the value 
		for(i=0;i<m;i++){
			for(j = rank*loop; j<end; j++){
				if(i==0 || i ==m-1 || j==0||j==n-1){
				A[i][j] = A_0 [i][j];  //unchanged along boarder
				}

				else{
					z = (f(A_0[i-1][j])+f(A_0[i+1][j])+f(A_0[i][j-1]) + f(A_0[i][j+1]) + f(A_0[i][j])) / 5;
					A[i][j] = max(-100, min(100, z)); 
				}
			}
		}

		if(rank ==0){
			//communication for proc 1
		}else if(rank ==p-1){
			//communication for proc p
		}else{
			// comminication for intermediate processors

		}

	}

	endtime = MPI_Wtime();

	printf("time: %s\n", (endtime-starttime));

}

//function  f
double f(double x){
	double y;
	int i;

	y = x; 
	for(i=1; i<=10; 1++){
		y = y+sin(x*i)/(2.0^i);
	}

	return y;

}