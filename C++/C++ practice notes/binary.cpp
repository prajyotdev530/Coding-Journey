#include<iostream>
using namespace std;






int main(){

int a;
cin>>a;
int rem;
int power =1;
int binary=0;
while(a>0){
   rem=a%2;
   binary=binary+(rem*power);
  
   power=power*10;
    a=a/2;     
}
cout<<binary;


return 0;
}