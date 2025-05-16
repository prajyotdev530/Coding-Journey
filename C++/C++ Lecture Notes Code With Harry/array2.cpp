#include<iostream>
using namespace std;

int main(){
//pointers and array
int marks[4]={88,56,43,54};
int* p=marks;
cout<<p<<endl;
//if we write like this the p will store the address 
//of first block of array,and *p will give the value of
//first block of array
//for finding the address of next elements of array
//do (p+i)
cout<<*p<<endl;
cout<<*(p+1)<<endl;
cout<<(p+1)<<endl; //this gives the address of 2nd block
int i;
do{
    cout<<"the value of"<<i<<"th block is"<<*(p+i)<<endl;
    i++;
}
while(i<=3);
cout<<*(p++)<<endl;//here it givves the value of first block 
//but it gets incremented for by one for next cout
//if we do ++p then firstly it gets increment and then print
cout<<*(p);


return 0;
}