#include<iostream>
using namespace std;

int main(){


 /* for(int j=1;j<=9;j++){
cout<<j;
if(j%3==0){
    cout<<endl;
}


  }






return 0;*/
int n;
cin>>n;
int num=1;
for (int  i = 0; i <n; i++)
{for(int j=0;j<n;j++){

    cout<<num;
    num=num+1;
}
cout<<endl;
}
cout<<num;
    
}
