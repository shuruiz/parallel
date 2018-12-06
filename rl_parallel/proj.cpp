//massive asynchronous reinforcement learning sarsa, q-learning and A3C. 
// problems to solve:
// is asynchronous good? all sychornized agents work better in massive parallelism status. 

#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <ctime>
#include <cuda.h>
#include <float.h>

#define THREADS_PER_DIM 32
using namespace std;

__device__
void one_step_q(double *d_Theta, double *d_prevTheta, int max_T){
	int step =0;
	double *target_Theta = d_Theta; 
	// initialize gradient 
	//code below



	// initialize state
	//code below


	while(step<max_T){
		
	}

}

int main(int argc, char** argv){

	return 0;
}