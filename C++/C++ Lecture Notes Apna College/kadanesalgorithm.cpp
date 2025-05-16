#include<iostream>
using namespace std;

int main(){
int a=5;
int maxsum=INT_MIN;
int currsum=0;
int arr[5]={1,2,3,45,5};
for(int i=0;i<a;i++){
    currsum=0;
    for(int j=i;j<a;j++){
        currsum=currsum+arr[j];
        maxsum=max(currsum,maxsum);
       if(currsum<0){
        currsum=0;
       }
    }
}
cout<<maxsum;


return 0;
}