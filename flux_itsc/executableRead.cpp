
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
	int m= atoi(argv[1]);
	string fileA, fileB; 
	fileA = atoi(argv[2]);
	fileB = atoi(argv[3]);

	ifstream mA, mB;
	mA.open(fileA); 
	mB.open(fileB); 

	std::vector<double>numbers;
	double number;
	while(mA >> number){
    numbers.push_back(number);}

    std::vector<int>indexes;
    int index;
    while(mB >> index){
    	indexes.push_back(index);
    }

    printf("top3A %f, %f, %f \n", numbers[0], numbers[1], numbers[2]);
    cout<<numbers.size()<<'\n';
    cout<<endl;

    printf("top3B %d, %d, %d  \n", indexes[0], indexes[1], indexes[2]);
    cout<<indexes.size()<<'\n';

}