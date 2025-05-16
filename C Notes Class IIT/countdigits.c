#include<stdio.h>

int countdigit(long int n,int count){
if(n!=0){
count++;
return countdigit(n/10,count);
}
else{return count;}


}
int main(){
int a=2343;
int count=0;
printf("%d",countdigit(7894678,0));



}