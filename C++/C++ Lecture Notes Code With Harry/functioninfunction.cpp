#include<iostream>
using namespace std;
//here g() function  is run in main function

int loop(int a,int b);
void g();

int main(){
    int num1,num2;
cin>>num1;
cin>>num2;
cout<<loop(num1,num2);
g();

return 0;
}
int loop(int a,int b){
int p =a*b;
 return p;

}
//void means nothing
void g()
{

    cout<<"good night";
}

