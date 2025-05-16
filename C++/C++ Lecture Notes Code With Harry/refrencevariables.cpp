#include<iostream>
using namespace std;
int main(){
    //refrence variable is used to point to a same thing
    //here y and x point the same thing if we change any var then another will also change
    
    float x=780 ;
float &y=x;
cout<<x<<endl;
cout<<y;
return 0;
     




}