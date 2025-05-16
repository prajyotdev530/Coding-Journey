#include<iostream>
using namespace std;
int a(int);
int hii(int);
int main(){


int a=3;
hii(a);
return 0;

};
int hii(int x){
    
cout<<x;
a(x);

return 0;


};
int a(int y){
    
int marks[4]={5,6,4,7};
cout<<marks[y];
for(int x=0;x<3;x++){
    cout<<marks[x];
}
return 0;
}