#include<iostream>
using namespace std;

int main(){
int n;
cin>>n;
int cof[n+1];
for(int i=0;i<=n;i++){
   cin>>cof[i];
};

cout<<"The Polynomial is:P(x)";
cout<<cof[0]<<"+";
cout<<cof[1]<<"x";
for(int i=2;i<=n;i++){
   
cout<<cof[i];
if(cof[i]!=0){
    cout<<"+";
    cout<<"x^"<<i;
}

}





return 0;
}