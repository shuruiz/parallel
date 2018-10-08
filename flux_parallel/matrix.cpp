#include <stdlib.h>
#include <mpi.h>
#include <stdio.h>
#include <math.h>
// #include <vector>
#include <algorithm>

using namespace std;

// template<typename T>
// std::vector<T> slice(std::vector<T> &const v, int s, int r){

// 	auto first =v.cbegin()+s+1;
// 	auto last =v.cbegin+r+2;
// 	std::vector<T> vec(first, last);
// 	return vec;
// }

int m=2000;
int n=500;
int p =2; // # of proc. 

//function  f
double f(double x){
	double y;
	int i;

	y = x; 
	for(int i=1; i<=10; i++){
		y = y+sin(x*i)/(pow(2.0,i));
	}
	return y;
}


int main(int argc, char** argv){
	
	double starttime;
	double endtime;
	int rank;
	int size;
	int b;

	MPI_Init(&argc, &argv);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if(rank==p-1){
		b = n-floor(n*1.0/p)*rank;
	}else{
		b = floor(n*1.0/p);
	}


	int n_row = b+2; // append one row above and one row below the target matrix, ghost cells
	// std::vector<double> v[m][n_row];


	double A[m][n_row]; 
	// initialize below 
	for(int i=0;i<m;i++){
		for(int j=1;j<n_row-1;j++){
			
			A[i][j] = i* sin(i) +j * cos(j) + sqrt(i+j);			
		}
	}


	// do 10 iteration below
	if(rank==0){
		starttime = MPI_Wtime();
	}
	
	
	for(int t=0; t<10; t++ ){ // 10 iteration below
		// std::vector<double>  prev = slice(v, 0, m);
		// std::vector<double>  prev = slice(v, n_row, m);

		// send self_prev, self_tail 
		double self_prev[m], self_tail[m];
		for(int num =0; num<m; num++){
			self_prev[num] = A[1][num]; 
			self_tail[num] = A[n_row-2][num]; 
		}  // two ghost cells

		double prev[m], tail[m];
		if(rank ==0){
			//communication for proc 1
			MPI_Send(&self_tail, m, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
			MPI_Recv(&tail, m, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);

		}else if(rank ==p-1){
			//communication for proc p
			MPI_Send(&self_prev, m, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD);
			MPI_Recv(&prev, m, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);

		}else{
			// comminication for intermediate processors
			MPI_Send(&self_prev, m, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD);
			MPI_Send(&self_tail, m, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
			MPI_Recv(&prev, m, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			MPI_Recv(&tail, m, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		}

		// if prev || tail not null, then append them to A matrix
		if(prev[0]){		
			for(int num =0; num<m; num++){
				A[0][num]=prev[num]; 
			}  // two ghost cell
		}
		if(tail[0]){
			for(int num =0; num<m; num++){
				A[n_row-1][num]=tail[num]; 
			}  // ghost cell
		}

		double A_0[m][n_row];
		for(int i = 0; i<m; i++){ //copy A to A_0
			for(int j=0; j<n_row; j++){
				A_0[i][j] = A[i][j];
			}
		}

		// do the calculation on core matrix (ghost cells do not included) below
		for(int i=0; i<m; i++){
			for(int j=1;j<n_row-1;j++){
				if((rank==0 && j==1)|| (rank ==p-1 && j== n_row-2) || i==0 || i==(m-1)){ // boarder 
					A[i][j] = A_0[i][j];
				}else{
					double z = (f(A_0[i-1][j])+f(A_0[i+1][j])+f(A_0[i][j-1]) \
						+ f(A_0[i][j+1]) + f(A_0[i][j])) / 5.0;

					double _min = fmin(100, z);
					A[i][j] = fmax(-100, _min); 
				}
			}
		}

	}

	// sum below
	double sum[2]={0.0,0.0};



	for(int i=0; i<m; i++){
		for(int j=1;j<n_row-1;j++){
			sum[0] += A[i][j];
			sum[1] += pow(A[i][j],2);
		}
	}

	printf("rank: %d \n", rank);
	printf("sum %f\n", sum[0]);
	printf("square_sum: %f\n", sum[1]);

	double rev_sum[2];

	if(rank==0){

		MPI_Recv(&rev_sum, 2, MPI_DOUBLE, rank+1, 2, MPI_COMM_WORLD,MPI_STATUS_IGNORE);		
		sum[0] += rev_sum[0];
		sum[1] += rev_sum[1];
	

	}else if(rank==p-1){
		MPI_Send(&sum, 2, MPI_DOUBLE, rank-1, 2, MPI_COMM_WORLD);

	}else{
		MPI_Recv(&rev_sum, 2, MPI_DOUBLE, rank+1, 2, MPI_COMM_WORLD,MPI_STATUS_IGNORE);	
		sum[0] += rev_sum[0];
		sum[1] += rev_sum[1];
		MPI_Send(&sum, 2, MPI_DOUBLE, rank-1, 2, MPI_COMM_WORLD);
	}
	if(rank==0){
		endtime = MPI_Wtime();
		printf("time: %f\n", (endtime-starttime));
	}

	if(rank==0){
		printf("SUM: %f\n", sum[0]);
		printf("square_sum: %f\n", sum[1]);
	}
	

	MPI_Finalize();

}

