#include<iostream>
using namespace std;

int main(){

char a;
cin>>a;
int b=int(a);
if(b>=65&&b<=92){
    cout<<"the number is capital";
}
else{
    cout<<"the number is small";
}
char ch='A';
cout<<(ch==65);

return 0;
}