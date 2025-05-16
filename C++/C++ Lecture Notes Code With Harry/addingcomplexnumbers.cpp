#include<iostream>
using namespace std;
class complex{
int real;
int imaginary;
public:
void setdata(int a ,int b){
    real=a;
    imaginary=b;
}
void sumcomplex(complex o1,complex o2){
real=o1.real+o2.real;
imaginary=o1.imaginary+o2.imaginary;
}
void displaynumber(){
    cout<<"the sum of the complex numbers are"<<real<<"+i"<<imaginary;
}



};

int main(){
complex o1;
complex o2;
complex o3;

 
 o1.setdata(1,2);
 o2.setdata(2,3);
o3.sumcomplex(o1,o2);
o3.displaynumber();


return 0;
}