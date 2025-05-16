#include<iostream>
using namespace std;
int swapRefrenceVar(int &a,int &b){
    int temporary=a;
    a=b;
    b=temporary;
    return 0;
}
//as by using refrence variable the both variables 
//changes if one changes
int main(){


    int x=4,y=3;
    cout<<x<<y;
    swapRefrenceVar(x,y);
    cout<<x<<y;
    return 0;

}

