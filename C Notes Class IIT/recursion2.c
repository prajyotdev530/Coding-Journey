#include<stdio.h>
int a=0;
int sum(int n);
int main(){
int n;
scanf("%d",&n);
int b=sum(n);
printf("%d",b);

}
int sum(int n){ 
    if(n==0){
        return 0;
    } 
return n+sum(n-1);


}