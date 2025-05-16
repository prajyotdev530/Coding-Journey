#include<stdio.h>

void swap(int *a,int *b){
    int temp=*a;
    *a=*b;
    *b=temp;
}
int partition(int arr[],int low,int high){
    int pivot=arr[high];
    int i=low-1;
    for(int j=low;j<high;j++){
        if(arr[j]<pivot){
            i++;
            swap(&arr[i],&arr[j]);
        }
    };
    swap(&arr[i + 1], &arr[high]); 
    return i + 1;  
}


void quicksort(int arr[],int low,int high){
    if (low<high){
        int pi=partition(arr,low,high);
        quicksort(arr,low,pi-1);
        quicksort(arr,pi+1,10);
    }
}

int main(){
int arr[10]={5,4,6,7,8,3,21,9,10,2};
quicksort(arr,0,10);
for (int i = 0; i < 10; i++){
printf("%d ", arr[i]);}




}


