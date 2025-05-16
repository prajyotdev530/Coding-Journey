#include<iostream>
using namespace std;


int bitodeci(int a){
    int rem;
    int binary=0;
    int power=1;

while(a>0){
rem=a%10;
binary=binary+(rem*power);
power=power*2;
a=a/10; 


}
return binary;

}

int main(){


int a;
cin>>a;
    cout<<bitodeci(a)<<endl;
    cout<<353%1000;






return 0;
}