//
//  shuruiz
//

#include<iostream>
#include<float.h>
#include<cmath>
#include<stack>
#include<utility>
#include<omp.h>
using namespace std;

stack < pair<double, double> > _stack;
double epsilon=pow(10, -6);
int s=12;
double fx(double x);
double M= -DBL_MAX; // initialize M to the -max_double, infinite small
int activeThread=0;

omp_lock_t maxLock;
omp_lock_t stackLock;

int main(int argc, char** argv)
{
    
    _stack.push(make_pair(1, 100));
    double starttime=omp_get_wtime();
    omp_init_lock(&maxLock);
    omp_init_lock(&stackLock);
    
#pragma omp parallel
    {
        while(true)
        {
            omp_set_lock(&stackLock);
            // elimiate cout command to shorten the running time.
            // cout<<"on thread: "<<omp_get_thread_num()<<endl;
            if(_stack.empty())
            {
                if(activeThread==0)
                {
                    //master thread
                    omp_unset_lock(&stackLock);
                    break;
                }
                else
                    omp_unset_lock(&stackLock);
            }
            else
            {
                activeThread++;
                double c=_stack.top().first;
                double d=_stack.top().second;
      
                _stack.pop();
                omp_unset_lock(&stackLock);
                
                double f_c=fx(c);
                double f_d=fx(d);
                
                omp_set_lock(&maxLock);
                
                if(f_c>M || f_d>M)
                {
                    //set M to max{M, fc, fd}
                    M=max(f_c, f_d);
                }
                
                if(((f_c+f_d+s*(d-c))/2)>=M+epsilon) // computing the max of interval [c,d]
                {
                    omp_set_lock(&stackLock);
                    // breadth-first generating intervals
                    _stack.push(make_pair(c, (c+d)/2));
                    _stack.push(make_pair((c+d)/2, d));
                    activeThread--;
                    omp_unset_lock(&stackLock);
                }
                
                else
                //get rid of this interval, no need to compute because max interval [a,b] < M+epsilon
                {
                    omp_set_lock(&stackLock);
                    activeThread--;
                    omp_unset_lock(&stackLock);
                }
                omp_unset_lock(&maxLock);
            }
        }
    }
    double endtime=omp_get_wtime();
    
    double total_time=endtime-starttime;
    cout<<"The runtime is: "<<total_time<<endl;
    cout<<"The final result is: "<<M<<endl;
    return 0;
}

//f(x) below, double precision
double fx(double x)
{
    double result=0;
    for(int i=100; i>=1; i--)
    {
        double tmp=0;
        for(int j=i; j>=1; j--)
        {
            tmp+=pow(x+0.5*j, -3.3);
        }
        result+=(sin(x+tmp)/pow(1.3, i));
    }
    return result;
}

