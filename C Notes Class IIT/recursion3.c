#include<stdio.h>
int findarr(int i);
int main(){

int i=0;
printf("%d",findarr(i));
}
int findarr(int i){
    char arr[]={'a','b','c'};
   if(i<3){
if(arr[i]=='c'){
    return i;
}
else{return findarr(i+1);};
   }
  
}