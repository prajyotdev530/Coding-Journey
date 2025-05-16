#include<iostream>
using namespace std;
/*int sum(int a){
int c;
int d;
int e;
int f;
int g;
c=a%10;
d=a%100;
e=d/10;
f=a/100;

return (c+e+f);

}*/
/*int sum(int a){

int b;
int c;
int d;
b=a%10;
a=a/10;
c=a%10;
a=a/10;
d=a%10;
return (b+c+d);
while(a&10>0){

}*/
int sum(int a){
    int sum=0;
while(a%10>0){
    sum=sum+(a%10);
    a=a/10;
    
}
return sum;


}





int main(){

int a;
cin>>a;
cout<<sum(a);
char hi="hii";
cout<<hi;


return 0;
}