#include<stdio.h>
int main(){

int count=0;

    int arr[40]={1,2,3,4,5,6,6,3,2,1};
    for(int i=0;i<10;i++){
        count=count+arr[i];
    }
   float mean=count/10;
    printf("%f",mean);
}