#include<iostream>
using namespace std;

int add1;
int add2;

 void swap(int array[]){

int smallest=INT_MAX;
int largest=INT_MIN;


for(int i=0;i<5;i++){
      if(array[i]<smallest){
        smallest=array[i];
        add2=i;
      };
      if(array[i]>largest){
           add1=i;
        largest=array[i];
      }

}

for(int i=0;i<5;i++){
    cout<<array[i];}


cout<<smallest<<largest;


int a=array[add1];
array[add1]=array[add2];
array[add2]=a;

}



int main(){
int array[]={2,3,4,5,6};
swap(array);


for(int i=0;i<5;i++){
    cout<<array[i];
}



return 0;
}