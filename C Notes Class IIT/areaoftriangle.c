#include<stdio.h>
#include<math.h>
int main(){
    float x1,y1,x2,y2,x3,y3,a;
    printf("Enter the first Co ordinate:");
    scanf("%f%f",&x1,&y1);
    printf("Enter the second Co ordinate:");
    scanf("%f%f",&x2,&y2);
    printf("Enter the third Co ordinate:");
    scanf("%f%f",&x3,&y3);
    a=0.5*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2));
    printf("%f",a);

    return 0;
}