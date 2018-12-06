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
#include <ctime>
#include <float.h>

using namespace std;

int main(int argc, char** argv){
    // int m= atoi(argv[1]);
    int m =4000000;
    //n = atoi(argv[2]);
    double *A,*C;
    int *B; // index
    int size_a = m*sizeof(double);
    int size_b = m*sizeof(int);
    
    A = (double *)malloc(size_a);
    B = (int *)malloc(size_b);
    
    
    // init below
    for(int i =0; i<m; i++){
        A[i] = rand()%100000;
        B[i] = rand()%450000;
    }
    
    int len_c = *std::max_element(B,B+m);
    int size_c =  len_c *sizeof(double);
    C = (double *)malloc(size_c);
    
    
    clock_t startTime = clock();
    for(int j=0; j<len_c;j++){
        C[j]=0;
        for(int k=0; k<m; k++){
            int index =B[k];
            if(index ==j){
                C[j] += A[index];
            }
        }
        if(j%1000==0){
            cout<<j<<endl;}
    }
    clock_t endTime = clock();
    clock_t clockTicksTaken = endTime - startTime;
    double timeInSeconds = clockTicksTaken / (double) CLOCKS_PER_SEC;
    printf("time in seconds %f \n",timeInSeconds);
    
    free(A);free(B);
    free(C);
    return 0;
}
