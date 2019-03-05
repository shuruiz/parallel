#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <ctime>
#include <string>
// #include <cuda.h>
#include <float.h>
#include <algorithm>
#include <stdio.h>
#include <vector>
#include <fstream>

using namespace std; 

int main(int argc, char** argv){
	ifstream mA, mB;
	mA.open("/Users/ethan/Documents/repo/parallel/flux_itsc/A.txt"); 
	mB.open("/Users/ethan/Documents/repo/parallel/flux_itsc/B.txt"); 
	std::vector<double>numbers;
	double number;
	while(mA >> number){numbers.push_back(number);}
    if(numbers.empty()){cout<<"empty\n";}
    cout<<"A size"<<numbers.size()<<'\n';

    std::vector<int>indexes;
    int index;
    while(mB >> index){indexes.push_back(index);}
    cout<<"B size"<<indexes.size()<<'\n';


	int m= numbers.size(); 
	// int m =1000000;
	//n = atoi(argv[2]);
	double *A,*C;
	int *B; // index
	int size_a = m*sizeof(double);
	int size_b = m*sizeof(int);

	A = (double *)malloc(size_a);
	B = (int *)malloc(size_b);


	// init below 
	for(std::vector<int>::size_type i = 0; i <numbers.size(); i++) {
		A[i] = numbers[i];
		B[i] = indexes[i];
	}

	int len_c = *std::max_element(B,B+m);
	int size_c =  len_c *sizeof(double);
	C = (double *)malloc(size_c);


	clock_t startTime = clock();
	// for(int j=0; j<len_c;j++){
 //        C[j]=0;}
    
 //    for(int k=0; k<m; k++){
 //        int index =B[k];
 //        C[index] += A[k];
 //        if(k%1000==0){
 //            cout<<k<<endl;
            
 //        }
 //    }

    // or two for loop below
	for(int j=0; j<len_c;j++){
		C[j]=0;
		for(int k=0; k<m; k++){
			int index =B[k];
			if(index ==j){
				C[j] += A[index];
			}
		}
	}
	clock_t endTime = clock();
	clock_t clockTicksTaken = endTime - startTime;
	double timeInSeconds = clockTicksTaken / (double) CLOCKS_PER_SEC;
	printf(" time in second %f\n", timeInSeconds );
	free(A);free(B);
	free(C);
	return 0;
}