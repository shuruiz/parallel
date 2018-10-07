#include <stdlib.h>
#include <mpi.h>
#include <math.h>
// #include <vector>

using namespace std;

// template<typename T>
// std::vector<T> slice(std::vector<T> &const v, int s, int r){

// 	auto first =v.cbegin()+s+1;
// 	auto last =v.cbegin+r+2;
// 	std::vector<T> vec(first, last);
// 	return vec;
// }

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
	// std::vector<double> v[m][n_row];


	double A[m][n_row] 
	// initialize below 
	for(i=0;i<m;i++){
		for(j=1;j<n_row-1;j++){
			
			A[i][j] = i* sin(i) +j * cos(j) + sqrt(i+j);			
		}
	}

	MPI_Init(&argc, &argv);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	// do 10 iteration below
	if(rank==0){
		starttime = MPI_Wtime();
	}
	
	double A_0[m][n_row];
	for(t=0; t<10; i++ ){
		// std::vector<double>  prev = slice(v, 0, m);
		// std::vector<double>  prev = slice(v, n_row, m);

		// send self_prev, self_tail 
		double self_prev, self_tail;
		for(num ==0; num<m; num++){
			self_prev[num] = A[1][num]; 
			self_tail[num] = A[n_row-1][num]; 
		}  // two ghost cells

		double prev[m], tail[m];
		if(rank ==0){
			//communication for proc 1
			MPI_Send(&self_tail, 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
			MPI_Recv(&tail, 1, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD);

		}else if(rank ==p-1){
			//communication for proc p
			MPI_Send(&self_prev, 1, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD);
			MPI_Recv(&prev, 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD);

		}else{
			// comminication for intermediate processors
			MPI_Send(&self_prev, 1, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD);
			MPI_Send(&self_tail, 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
			MPI_Recv(&prev, 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD);
			MPI_Recv(&tail, 1, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD);
		}

		// if prev || tail not null, then append them to A matrix
		if(prev){		
			for(num =0; num<m; num++){
				A[0][num]=prev[num]; 
			}  // two ghost cell
		}
		if(tail){
			for(num =0; num<m; num++){
				A[n_row-1][num]=tail[num]; 
			}  // ghost cell
		}

		double A_0 = A;

		// do the calculation below
		for(i=0; i<m; i++){
			for(j=1;j<n_row-1;j++){
				if((rank==0 && j==1)|| (rank ==p-1 && j== n_row-2) || i==0 || i==(m-1)){ // boarder 
					A[i][j] = A_0[i][j];
				}else{
					z = (f(A_0[i-1][j])+f(A_0[i+1][j])+f(A_0[i][j-1]) \
						+ f(A_0[i][j+1]) + f(A_0[i][j])) / 5;
					A[i][j] = max(-100, min(100, z)); 
				}
			}
		}

		// calcalate the value 
		// for(i=0;i<m;i++){
		// 	for(j =0; j<n_row; j++){
		// 		if(i==0 || i ==m-1 || j==0||j==n-1){
		// 		A[i][j] = A_0 [i][j];  //unchanged along boarder
		// 		}

		// 		else{
		// 			z = (f(A_0[i-1][j])+f(A_0[i+1][j])+f(A_0[i][j-1]) + f(A_0[i][j+1]) + f(A_0[i][j])) / 5;
		// 			A[i][j] = max(-100, min(100, z)); 
		// 		}
		// 	}
		// }



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