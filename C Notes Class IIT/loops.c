/*Read in an integer N. Then read in integers, and find the sum of the first N positive integers read. Ignore any
negative integers or 0 read (so you may actually read in more than N integers, just find the sum with only the
positive integers and stop when N such positive integers are read)*/


#include<stdio.h>
int main(){
int N;
int sum=0;
scanf("%d",&N);
int a;
int count=0;
do{
scanf("%d",&a);
if(a>0){
sum=sum+a;
count++;
}




}
while(count<N);
printf("%d",sum);


}