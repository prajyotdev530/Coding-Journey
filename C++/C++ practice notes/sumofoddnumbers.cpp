#include<iostream>
using namespace std;

int main(){


/*int i;
cin>>i;
int sum=0;
for(int a=0;a<=i;a++){

    if(!(a%2==0)){

        sum=sum+a;
    }

}
cout<<sum;*/
int i;
cin>>i;
int a=0;
int sum=0;
while(a<i){

    if(!(a%2==0)){

     sum=sum+a; 
     a++;
    }
}
cout<<sum;



return 0;}
