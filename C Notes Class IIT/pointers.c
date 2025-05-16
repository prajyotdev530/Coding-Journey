#include<stdio.h>
int main(){
int a=10;
int *p=&a;

int arr[]={1,2,3,4,5,6,7,8,9};
int *ptr0=arr;
int *ptr1=&arr[1];
int *ptr2=&arr[2];
int *ptr3=&arr[3];
printf("%p\n",ptr0);
printf("%p\n",ptr1);
printf("%p\n",ptr2);
printf("%p\n",ptr3);
printf("%ld\n", (long)ptr1 - (long)ptr0);
printf("%ld\n", (long)ptr2 - (long)ptr1);
printf("%ld\n", (long)ptr3 - (long)ptr2);
printf("the sizeof arr is %lu",sizeof(arr));


}