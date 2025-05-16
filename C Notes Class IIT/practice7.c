/*Read in characters until the ‘\n’ character is typed. Count and print the number of lowercase letters, the
number of uppercase letters, and the number of digits entered.*/

#include<stdio.h>
int main(){
char a;
int lcount=0;
int ucount=0;
while(a!="\n"){
      scanf("%c",&a);
      int c=a;
      if(65<c<90){
        ucount++;
         }
       elif(97<c<122){
  lcount++;
}
else
}




    return 0;
}