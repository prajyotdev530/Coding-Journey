#include<iostream>
using namespace std;
int sum(int a){
    
    int sum=0;
    for(int b=1;b<a;b++){

       sum =sum+b;
    }
    return sum;



}

int main(){
    int i;
    cin>>i;

    cout<<sum(i);








return 0;
}