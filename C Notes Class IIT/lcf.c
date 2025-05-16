#include<stdio.h>

int lcm(int a,int b){
    static int multiple = 2;
if(a%multiple==0&&b%multiple==0){
    return multiple;
}
else{
    multiple+=1;
    return lcm(a,b);}

}




int main(){

int a,b;
scanf("%d %d",&a,&b);
printf("%d",lcm(a,b));
return 0;
}




