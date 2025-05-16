#include<iostream>
using namespace std;

int bino(int a,int b){

int afacto=1;
for(int i=1;i<=a;i++){
    afacto=afacto*i;
}
int bfacto=1;
for(int i=1;i<=b;i++){
    bfacto=bfacto*i;
}
int bafacto=1;
for(int i=1;i<=(a-b);i++){
    bafacto=bafacto*i;
}
int c=bfacto*bafacto;
return afacto/c;

}

int main(){

int a;
int b;
cin>>a;
cin>>b;
if (a>=b)
{
    cout<<"The Binomial coeeficent of the given numbers is : "<<bino(a,b);

}
else{
    cout <<"Please apni gand jake kahi aur maraye!!!";
}




return 0;
}