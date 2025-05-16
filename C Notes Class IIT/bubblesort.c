#include <stdio.h>

void bubblesort(int x[],int size){
for (int i=0;i<size;i++){
for (int j=0;j<size-i-1;j++){
    if (x[j]>x[j+1]){
        int temp=x[j];
        x[j]=x[j+1];
        x[j+1]=temp;
}
}

}}

int main(){
int x[]={1,2,3,5,4,6,7,8,9,10};
int size=10;
bubblesort(x,size);
for (int i=0;i<size;i++){
    printf("%d",x[i]);}


}