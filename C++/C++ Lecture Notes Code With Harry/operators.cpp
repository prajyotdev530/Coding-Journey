
#include <iostream>
using namespace std;
int c=45;
int main(){
    int a,b,c;

    cin>>a;
    cin>>b;
    c=a+b;
    cout<<"the value of a+b is"<<c;
    cout<<"the value of global c is"<<::c; 
    


    


    return 0;
}