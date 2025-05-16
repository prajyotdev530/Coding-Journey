#include<iostream>
using namespace std;

int main(){
 //union is like structures but it is memory efficient 
    //suppose in struct if we made 3 vaaariables of 4,3,2  bytes 
    //then the memory used is 9 bytes as in struct we can use all the 
    //variables at same time but in union in the same case the memory alloted will be equal to 
    //that of highest one as in union we can use only one variable at  a time

cout<<"hii"<<endl;
union currencytypes{
    int ricebags;
    float wheatweight;

};
currencytypes m1;
m1.ricebags=4;
m1.wheatweight=4.5;
cout<<m1.wheatweight;
cout<<m1.ricebags;


//here we will get value of only one variable



return 0;
}