#include<iostream>
using namespace std;
//when a function is run firstly it finds main function and then 
//others,here the int a and int b are called as the arguments if sum function
//the  value which we return here it is c goes to the sum(num1,num2){function name of c}
int sum(int a,int b){
    int c=a+b;
return c;


}

int main(){

    int num1,num2;
    cout<<"Enter the value of num1"<<endl;
    cin>>num1;
    cout<<"Enter the value of num2"<<endl;
    cin>>num2;
cout<<"The sum is"<<sum(num1,num2);

return 0;
}