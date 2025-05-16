#include<iostream>
using namespace std;

int main(){
int marks[5]={4,5,6,7,8};
int smallest=INT_MAX;
int largest=INT_MIN;

for(int i=0;i<5;i++){
    smallest=min(marks[i],smallest);
    largest=max(marks[i],largest);
    if(marks[i]>smallest){
        cout<<i;
    }
    if(marks[i]<largest){
        cout<<i;
    }

}
cout<<smallest;
cout<<largest;





return 0;
}