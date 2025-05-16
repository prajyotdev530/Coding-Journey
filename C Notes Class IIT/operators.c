#include<stdio.h>
int main(){

int a=3;

printf("%d",a);
++a;
printf("%d",a);
int b= 50+ a++;
printf("%d",b);
printf("%d",a);
if(10>9){
    printf("hello");
}
return 0;
}