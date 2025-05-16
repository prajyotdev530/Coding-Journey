#include<iostream>
using namespace std;
int factorial(int a){
    int multiply=1;
for(int i=1;i<=a;i++){

multiply=multiply*i;
}
return multiply;

    
}
int main(){
    int b;
    cin>>b;
    cout<<factorial(b);
   
return 0;
}