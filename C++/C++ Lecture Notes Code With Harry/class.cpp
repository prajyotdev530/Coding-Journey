#include<iostream>
using namespace std;
class employee{
    private:
    int a,b,c;
  public:
  int d,e;
    int address(int a1,int b1,int c1);
    int printadress(){
     cout<<a;
     cout<<b;
     cout<<c;
     cout<<d;
     cout<<e;
    }
};
int employee::address(int a1,int b1,int c1){
    a=a1;
    b=b1;
    c=c1;
}

int main(){
    employee prajyot;
    prajyot.address(1,2,3);
    prajyot.d=3;
    prajyot.e=4;
    prajyot.printadress();



return 0;
}