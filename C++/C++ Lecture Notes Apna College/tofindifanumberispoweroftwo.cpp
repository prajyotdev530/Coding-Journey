#include<iostream>
using namespace std;
int decitobin(int a){
    int power=1;
    int rem=0;
    int binary=0;
    while(a>0){
        rem=a%2;
        a=a/2;
        binary=binary+(rem*power);
        power=power*10;
    
    }
    if(binary%1==0){
        cout<<"the number is of twos power";
    }
    else{
        cout<<"the number is not of twos power";
    }
    return binary;



}

int main(){
    int a;
    cin>>a;
    cout<<decitobin(a);
   






return 0;
}