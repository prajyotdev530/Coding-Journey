#include <stdio.h>
void swap(int *a,int *b){
int t=*a;
*a=*b;
*b=t;
};
int partition(int arr[],int low,int high){
 int i=low-1;
 int pivot=arr[high];
for(int j=low;j<high;j++){
   if(arr[j]<pivot){
        i++;
        swap(&arr[j],&arr[i]);
   };
};
swap(&arr[i+1],&arr[high]);
return i+1;
}


void quicksort(int arr[],int low,int high){
if(low<high){
    int pivot_index=partition(arr,low,high);
    int pivot=arr[high];
    quicksort(arr,low,pivot_index-1);
    quicksort(arr,pivot_index+1,high);
}
} 


int main(){
int arr[]={1,2,7,4,2,5,9,32,5,7};
quicksort(arr,0,10);
for(int i=0;i<10;i++){
    printf("%d ",arr[i]);
}


}