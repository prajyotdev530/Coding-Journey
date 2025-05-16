#include<stdio.h>
int main(){

int a;
int b;
int c;
scanf("%d%d%d",&a,&b,&c);
if(a==b+c){
    printf("a=b+c");

}
else if(b==c+a){
    printf("b=c+a");
}
else if(c==a+b){
    printf("c=a+b");
}

}