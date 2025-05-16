#include<iostream>
using namespace std;
void changearray(int marks[100],int size){

    for(int i=0;i<size;i++){

        marks[i]=2*marks[i];
    }
}

 


int main(){

int size=3;
int marks[200]={1,2,3};
cout<<marks;
changearray(marks,3);
for(int i=0;i<size;i++){

    cout<<marks[i];
  
}
  cout<<marks;
return 0;
}
//here marks basically store the value of the address of first variable,ex if 100 is the starting address and it is integer array then by adding 4 we can get the address of the next block