
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
	// int m= atoi(argv[1]);
	cout<<"starting"<<endl;
	// string fileA, fileB; 
	// fileA = atoi(argv[1]);
	// fileB = atoi(argv[2]);
	// printf("here");
	// cout<<fileA<<endl;
	// cout<<fileB<<endl;
	// string fileA("A.txt");
	// string fileB("B.txt");
	ifstream mA;

	printf("%s\n", "opening file");
	// cout<<fileA<<endl;
	// cout<<fileB<<endl;

	mA.open("/Users/ethan/Documents/repo/parallel/flux_itsc/A.txt"); 
	if (!mA) {
    cout << "Unable to open file A.txt\n";}
  // call system to stop
    cout<<mA<<endl;

	ifstream mB;
	mB.open("/Users/ethan/Documents/repo/parallel/flux_itsc/B.txt"); 
	if (!mB) {
    cout << "Unable to open file B.txt\n";}
    cout<<mB<<endl;
	printf("%s\n", "file read");
	
	std::vector<double>numbers;
	double number;
	while(mA >> number){numbers.push_back(number);}
    printf("%s\n", "get A");
    if(numbers.empty()){cout<<"empty\n";}
    // cout<<numbers[0]<<numbers[1]<<numbers[2]<<endl;
    cout<<numbers.size()<<'\n';
    int m = numbers.size(); 
    printf("length %d \n", m);
    cout<<endl;

    std::vector<int>indexes;
    int index;
    while(mB >> index){
    	indexes.push_back(index);
    }
    printf("%s\n", "get B");
    // printf("top3B %d, %d, %d  \n", indexes[0], indexes[1], indexes[2]);
    cout<<indexes.size()<<'\n';
    for(std::vector<int>::size_type i = 0; i <30; i++) {
    cout<<indexes[i]<<endl;
}
// 	for(std::vector<T>::iterator it = v.begin(); it != v.end(); ++it) {
//     /* std::cout << *it; ... */
// }

//     for ( auto i = numbers.begin(); i != numbers.begin() +30; i++ ) {
//     std::cout << *i << std::endl;
// }



    return 0;

}