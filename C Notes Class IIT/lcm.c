#include<stdio.h>
int multiple=0;
int lcm(int a,int b){
multiple+=1;
if(multiple%a==0&&multiple%b==0){
    return multiple;
}
else{return lcm(a,b);}

}


int main(){
    int a,b;
scanf("%d %d",&a,&b);
printf("%d",lcm(a,b));
return 0;
}