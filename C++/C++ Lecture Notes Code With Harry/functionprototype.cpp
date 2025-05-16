#include<iostream>
using namespace std;



 /*here in functions if we write the sum after main it will not run
    as for main to be executed sum is required 
    we can solve this problem using function prototyping 
    we tell our compiler that you will get the needed function,keep finding
    format of function prototyping is
    type function name(argument)*/
    int sum(int a,int b);//here we can also only state as int sum (int,int)
     
int main(){
    int num1,num2;
    cin>>num1;
    cin>>num2;
    cout<<"the sum is"<<sum(num1,num2)<<endl;

//here the values which are in function are called as formal parameters
//here the formal parameters  are a and b
//and the actual values which pass in function are called as actual parameters 
//here the actual parameters are num1 and num2
//a and b will be taking values from actual parameters
 
return 0;
}
int sum(int a ,int b){
    int c=a+b;

return c;

}