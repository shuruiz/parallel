//
//  main.cpp
//  serial_proj
//
//  Created by Ethan Zhang on 12/5/18.
//  Copyright © 2018 Ethan Zhang. All rights reserved.
//

#include <iostream>
#include <stdlib.h>
#include <cmath>
//#include <random>
#include <ctime>
#include <float.h>

using namespace std;

int uniform_distribution(int rangeLow, int rangeHigh)
{
    int myRand = (int)rand();
    int range = rangeHigh - rangeLow ; //+1 makes it [rangeLow, rangeHigh], inclusive.
    int myRand_scaled = (myRand % range) + rangeLow;
    return myRand_scaled;
}

int main(int argc, char** argv){
    // int m= atoi(argv[1]);
    int m =10000000;
    int n =10000;
    //n = atoi(argv[2]);
    double *A,*C;
    int *B; // index
    int size_a = m*sizeof(double);
    int size_b = m*sizeof(int);
    
    A = (double *)malloc(size_a);
    B = (int *)malloc(size_b);
    
    
    // init below
//    std::random_device rd;  //Will be used to obtain a seed for the random number engine
//    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
//    std::uniform_int_distribution<> dis(0, n);
    
    for(int i =0; i<m; i++){
        A[i] = rand()%100000;
        B[i] = uniform_distribution(0,n);
    }
    
    int len_c = *std::max_element(B,B+m);
    int size_c =  len_c *sizeof(double);
    C = (double *)malloc(size_c);
    
    
    clock_t startTime = clock();
    for(int j=0; j<len_c;j++){
        C[j]=0;}
    
    for(int k=0; k<m; k++){
        int index =B[k];
        C[index] += A[k];
//        if(k%1000==0){
//            cout<<k<<endl;
            
//        }
    }

    
    clock_t endTime = clock();
    clock_t clockTicksTaken = endTime - startTime;
    double timeInSeconds = clockTicksTaken / (double) CLOCKS_PER_SEC;
    printf("time in seconds %f \n",timeInSeconds);
    
    free(A);free(B);
    free(C);
    return 0;
}
