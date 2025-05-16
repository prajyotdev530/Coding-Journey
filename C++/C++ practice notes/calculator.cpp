#include<iostream>
using namespace std;

int main(){
int a;
int b;
int c;
cout<<"enter number 1"<<endl;
cin>>a;
cout<<"enter number 2"<<endl;
cin>>b;


cout<<"which mathematical operation you want to perform "<<endl<<"if you need add then type 1"<<endl<<"if div 2"<<endl<<"if multi then 4"<<endl<<"if sub 3"<<endl;
cin>>c;
if(c==1){
    cout<<a+b;

}
else if(c==2){
    cout<<a%b;
}
else if(c==4){
    cout<<a*b;

}
else if(c==3){
    cout<<a-b;
};
return 0;
}