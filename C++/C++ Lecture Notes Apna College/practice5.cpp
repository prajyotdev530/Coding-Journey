#include<iostream>
using namespace std;
void swap(int &a,int &b){
    int temp=a;
    a=b;;
   b=temp;
   
}

int main(){

int arr[]={1,2,3,4,5,6};
int a=6;
int b=7;
swap(a, b);



    
  return 0;  
}

