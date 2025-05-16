#include<stdio.h>
#include<stdlib.h>
typedef struct{
    char element;
    int top;

}stack;
stack init ( )
{
stack S;

S.top = -1;
return S;

}


int isempty(stack S)
{
    if(S.top==-1)
        return 1;
    else
        return 0;
}
int main(){
    stack S;
   
    S.top=1;
    if(isempty(S))
        printf("Stack is empty");
    else
        printf("Stack is not empty");
    return 0;
}