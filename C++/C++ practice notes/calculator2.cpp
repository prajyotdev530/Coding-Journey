#include<iostream>
using namespace std;

int main(){
int a;
int b;
char operation;
cin>>a;
cin>>b;
cin>>operation;
switch(operation){


    case '+':
    cout<<a+b;
    break;
    case '-':
    cout<<a-b;
    break;
    case '%':
    cout<<a*b;
    break;
    case '*':
    cout<<a*b;
    default:
    cout<<"no operation";
}

return 0;
}