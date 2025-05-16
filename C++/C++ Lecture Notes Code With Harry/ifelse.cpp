#include <iostream>
using namespace std;
int main(){
    int a;
    cin>>a;
    if((a<18)&&(a>0)){


        cout<<"you are not eligible for the party";

    }
    else if(a==18){
        cout<<"you are a kid you can get a kid pass";
    }
    else if(a<0){
        cout<<"you are not yet born";
    }
    else{
        cout<<"you are eligible";
    }
    cout<<"hello world";






return 0;    
}