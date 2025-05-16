#include<stdio.h>
#include<math.h>
int main(){
int num1;
int num2;
int num3;
float num4;
float root;
float threshhold=10;

scanf("%d%d%d%f",&num1,&num2,&num3,&num4);

root=sqrt((num1+num2+num3+num4)/4);
if(root<threshhold){printf("Hello!");}
else{printf("no luck");}

}