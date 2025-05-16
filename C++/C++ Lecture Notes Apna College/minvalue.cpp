#include<iostream>
using namespace std;





int minvalue(int a,int b)// a and b are paramerters
{
    
if(a<b){return a;}
else{return b;}
    
    }
    int main(){

cout<<"min number is"<<minvalue(3,5);//3 and 5 are arguments

    

return 0;
}