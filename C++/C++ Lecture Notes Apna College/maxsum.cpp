#include<iostream>
using namespace std;









int main(){
    int currentsum=0;
    int maxsum=0;
    int size=6;
int arr[6]={1,2,3,100,-6,-10000000};
for(int st=0;st<size;st++){
for(int end=st;end<size;end++){
currentsum=currentsum+arr[end];
maxsum=max(currentsum,maxsum);
}

currentsum=0;
}




cout<<maxsum;
return 0;
}