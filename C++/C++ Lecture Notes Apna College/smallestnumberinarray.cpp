#include<iostream>
using namespace std;


int main(){
    int marks[5]={5,2,3,4,5};
    
int i=0;
int smallest;
smallest=marks[0];
    while(i<=5){
       if(marks[i]<smallest){
        smallest=marks[i];
       } 
       i++;
    }
cout<<smallest;

return 0;
}