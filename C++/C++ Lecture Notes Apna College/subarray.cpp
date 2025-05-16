#include<iostream>
using namespace std;
int subarray(int arr[]){
int size=4;
int sum=0;
int largest=INT_MIN;
for(int start=0;start<size;start++){
    for(int end=start;end<size;end++){

        for(int i=start;i<=end;i++){
            sum=sum+arr[i];
        
        }
        cout<<" ";
        if(sum>largest){
            largest=sum;   
        }
        sum=0;
       
        
        
    }
    cout<<endl;
    
  

}
cout<<largest;
return 0;
}

int main(){
 int arr[4]={-1,100,3,4};
  subarray(arr);
return 0;
}