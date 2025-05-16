#include<stdio.h>
int findminindex(int arr[],int left,int right);
int findmaxindex(int arr[],int left,int right);
void swap(int arr[],int i,int j);
void sortarray(int arr[],int n){
    int left=0;
    int right=n-1;
    int minindex,maxindex;
    while(left<right){
minindex=findminindex(arr,left,right);
swap(arr,arr[left],arr[minindex]);
maxindex=findmaxindex(arr,left,right);
swap(arr,arr[right],arr[maxindex]);
left++;
right--;
}
 }
                    int findminindex(int arr[],int left,int right){
                        int min=arr[left];
                        int mini=0;
                             for(int i=left;i<=right;i++){
                                if(arr[i]<min){
                                         min=arr[i];
                                         mini=i;
                                    if(i==right){
                                    return mini;
                                       }
                                     }
                                    }
                                 }
int findmaxindex(int arr[],int left,int right){
    int max=arr[left];
    int maxi=0;
    for(int i=left;i<=right;i++){
        if(arr[i]>max){
            max=arr[i];
            if(i==left){return maxi;}
        }
    }
}
void swap(int arr[],int i,int j){
    int temp=arr[i];
    arr[i]=arr[j];
    arr[j]=temp;
}








int main(){
int arr[5]={5,4,3,2,1};
sortarray(arr,5);
for(int i=0;i<5;i++){
    printf("%d",arr[i]);}


}