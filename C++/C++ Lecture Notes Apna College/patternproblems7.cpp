#include<iostream>
using namespace std;

int main(){
    int n;
    cin>>n;
for(int i=0;i<n;i++){
    int num=1;
    for(int j=1;j<n-i;j++){
             cout<<" ";
    }
    
    for(int p=1;p<=i+1;p++){
        cout<<num;
        num++;
       
        }
        
         for(int c=i;c>0;c--)
           {
            
            cout<<c;
           }
       cout<<endl;  
    }
   










return 0;
}