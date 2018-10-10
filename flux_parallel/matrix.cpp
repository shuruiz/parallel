#include <stdlib.h>
#include <mpi.h>
#include <stdio.h>
#include <math.h>
// #include <vector>
#include <algorithm>
#include <iostream>
#include <ctime>

using namespace std;

// template<typename T>
// std::vector<T> slice(std::vector<T> &const v, int s, int r){

// 	auto first =v.cbegin()+s+1;
// 	auto last =v.cbegin+r+2;
// 	std::vector<T> vec(first, last);
// 	return vec;
// }

// int m=2000;
// int n=500;
// int p =2; // # of proc. 

//function  f
double f(double x){
	double y;
	int i;

	y = x; 
	for(int i=1; i<=10; i++){
		y = y+sin(x*i)/pow(2.0,i);
	}
	return y;
}

double* serial(int m, int n){


	double A[m][n], A_0[m][n];
	for(int i =0; i<m; i++){
		for(int j=0; j<n; j++){
			A[i][j] = i*sin(i) +j*cos(j) +sqrt(i+j);
		}
	}

	for(int ite = 0; ite <10; ite++){
		for(int i = 0; i<m; i++){ //copy A to A_0
			for(int j=0; j<n; j++){
				A_0[i][j] = A[i][j];
			}
		}

		for(int i = 0; i<m; i++){ //copy A to A_0
			for(int j=0; j<n; j++){
				if(i==0 || j==0 || j==n-1 || i==m-1){
					A[i][j] = A_0[i][j];
				}else{
					double z = (f(A_0[i-1][j])+f(A_0[i+1][j])+f(A_0[i][j-1])+ f(A_0[i][j]))/5.0;
					double _min = fmin(100, z);
					A[i][j] = fmax(-100, _min);
				}
			}
		}
	}

	double *sum=new double[2];
	sum[0]=0.0;
	sum[1]=0.0;

	for(int i=0; i<m; i++){
		for(int j=0;j<n;j++){
			sum[0] += A[i][j];
			sum[1] += pow(A[i][j],2);
		}
	}

	return sum;
}

int main(int argc, char** argv){
	
	double starttime;
	double endtime;
	int rank;
	int size;
	int b;
	int m, n; 

	// cout<<"arg:"<<argv[2]<<endl;
	m =atoi(argv[1]);
	
	n =atoi(argv[2]);


	// MPI_Abort(MPI_COMM_WORLD,11);

	MPI_Init(&argc, &argv);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int p =size; // num of procs. 


	if(rank==0){
		starttime = MPI_Wtime();
	}

	if(rank==p-1){
		b = n-floor(n*1.0/p)*rank;
		// printf("b:%d\n", b);
	}else{
		b = floor(n*1.0/p);
		// printf("b:%d\n", b);
	}


	if(p==1){
		// if only 1 proc, serial 

		double *sum = serial(m,n);

		printf("sum: %f\n", sum[0]);
		printf("square_sum: %f\n", sum[1]);

		endtime = MPI_Wtime();
		printf("time: %f\n", (endtime-starttime));
		MPI_Finalize();
		return 0;
	}




	int n_row = b+2; // append one row above and one row below the target matrix, ghost cells



	// printf("rows:%d\n", n_row);
	// std::vector<double> v[m][n_row];
	double A[m][n_row]; 
	// printf("m %d", m );

	// initialize below 
	for(int i=0;i<m;i++){
		int start_j = (rank*b);
		for(int j=1;j<n_row-1;j++){
			int j_ =start_j+j-1; // mapping local j to global j. 

			// A[i][j] = i* cos(i) +(j_) * sin(j_) + sqrt(i+j_);		
			//change cos to sin and sin to cos when using changing the order of m and n, 
			// when dubugging using m =500, n = 2000, change use the above line. 	

			A[i][j] = i* sin(i) +(j_) * cos(j_) + sqrt(i+j_);	
		}
	}
	// do 10 iteration below

	for(int t=0; t<10; t++ ){ // 10 iteration below
		// std::vector<double>  prev = slice(v, 0, m);
		// std::vector<double>  prev = slice(v, n_row, m);

		// send self_prev, self_tail 
		// printf("rank: %d \n", rank);
		// printf("iteration: %d \n", t);
		double self_prev[m];
		double self_tail[m];

		for(int num =0; num<m; num++){

			self_prev[num] = A[num][1]; 
			self_tail[num] = A[num][n_row-2]; 
			// cout<< "num"<<num<<endl;
			// printf("times %d", num );
		}  // two ghost cells

		// printf("get prev tail");
		// cout<<"Get"<<endl;
		// cout<<"A"<<A[1][1];
		// cout<<"prev:"<<self_prev[1]<<endl;

		double prev[m]; 
		double tail[m];
		int flag_prev = 0; // check if prev has value or not 
		int flag_tail = 0; 


		// cout<<"rank"<<rank<<endl;
		// cout<<"m"<<m<<endl;
		// int count = sizeof(prev);
		// cout<<"count"<<count<<endl;
		if(rank ==0){
			// cout<<"sending "<<rank<<endl;
			//communication for proc 1
			MPI_Isend(self_tail, m, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
			// cout<<"send"<<rank<<endl;

			MPI_Recv(tail, m, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			flag_tail =1; 
			// printf("ihere, %d", rank);
			// cout<<"rev"<<rank<<endl;


		}else if(rank ==p-1){
			// cout<<"sending"<<rank<<endl;

			//communication for proc p
			MPI_Isend(self_prev, m, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD);
			// cout<<"send"<<rank<<endl;
			MPI_Recv(prev, m, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			flag_prev = 1; 
			// cout<<"zzz"<<endl;
			// printf("ihere, %d", rank);

		}else{
			// comminication for intermediate processors
			MPI_Isend(self_prev, m, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD);
			MPI_Isend(self_tail, m, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
			// cout<<"send"<<rank<<endl;
			MPI_Recv(prev, m, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			flag_prev=1;
			MPI_Recv(tail, m, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			flag_tail=1;

			// cout<<"comm"<<endl;
			// printf("ihere, %d", rank);
		}
		// cout<<"here"<<endl;
		// cout<<"iteration:"<<t<<endl;
		// if prev || tail not null, then append them to A matrix
		if(flag_prev==1){		// has value 
			for(int num =0; num<m; num++){
				A[num][0]=prev[num]; 
			}  // two ghost cell
		}
		if(flag_tail==1){ // has value 
			for(int num =0; num<m; num++){
				A[num][n_row-1]=tail[num]; 
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

		// printf("get Z\n");

	}

	// sum below
	double sum[2]={0.0,0.0};



	for(int i=0; i<m; i++){
		for(int j=1;j<n_row-1;j++){
			sum[0] += A[i][j];
			sum[1] += pow(A[i][j],2);
		}
	}

	// printf("rank: %d \n", rank);
	// printf("sum %f\n", sum[0]);
	// printf("square_sum: %f\n", sum[1]);

	double rev_sum[2];

	if(rank==0){

		MPI_Recv(rev_sum, 2, MPI_DOUBLE, rank+1, 2, MPI_COMM_WORLD,MPI_STATUS_IGNORE);		
		sum[0] += rev_sum[0];
		sum[1] += rev_sum[1];
	

	}else if(rank==p-1){
		MPI_Isend(sum, 2, MPI_DOUBLE, rank-1, 2, MPI_COMM_WORLD);

	}else{
		MPI_Recv(rev_sum, 2, MPI_DOUBLE, rank+1, 2, MPI_COMM_WORLD,MPI_STATUS_IGNORE);	
		sum[0] += rev_sum[0];
		sum[1] += rev_sum[1];
		MPI_Isend(sum, 2, MPI_DOUBLE, rank-1, 2, MPI_COMM_WORLD);
	}
	if(rank==0){
		endtime = MPI_Wtime();
		printf("time: %f\n", (endtime-starttime));
	}

	if(rank==0){
		printf("sum: %f\n", sum[0]);
		printf("square_sum: %f\n", sum[1]);
	}
	

	MPI_Finalize();

}

