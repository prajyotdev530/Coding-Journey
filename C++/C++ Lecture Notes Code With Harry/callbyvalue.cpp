#include<iostream>
using namespace std;
int sum(int a,int b){
int c;
    c=a+b;
    return c;
}
/*the below code cannot be used to swap numbers
void swap(int a,int b){

    int temp=a;
    a = b;
    b=temp;
}

int main(){
    int x=4,y=6;
    cout<<x<<endl<<y;
 swap(x,y);
 cout<<x<<endl<<y;
 //here even after running swap the numbers are not changed as 
 //as the value goes in swap as x and y and ni changes occur in main

return 0;
}*/

void swapPointer(int* a,int* b){
    int temp= *a;
    *a= *b;
    *b = temp;
}
int main(){
int x=6,y=2;
    cout<<x<<endl<<y<<endl;
    swapPointer(&x,&y);
    cout<<x<<endl<<y<<endl;
    return 0;
}
