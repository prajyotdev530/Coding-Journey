#include<stdio.h>
int main(){
    
        int num;
    
        printf("Enter a number between 0 and 999: ");
        
        while (scanf("%d", &num) != 1 || num < 0 || num > 999) {
            printf("Invalid input! Please enter a number between 0 and 999: ");
            while (getchar() != '\n'); // Clear input buffer
        }
    
        printf("Valid input: %d\n", num);
        
    
    
int a=num%10;
num=num/10;
int b=num%10;
num=num/10;
int c=num;
int sum=a+b+c;
if(num%sum==0){
    printf("Niven Number");
}
else{
    printf("Not a Niven Number");

}
return 0;}