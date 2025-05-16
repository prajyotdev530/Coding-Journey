#include<iostream>
using namespace std;
int print(int array[]){
int size=7;
for(int i=0;i<size;i++){
      for(int j=i+1;j<size;j++){
        if(array[j]==array[i]){
            cout<<array[i];
        }
      }

    
}

}


int main(){
    int array[]={1,2,3,4,2,3,1};
    int size=7;
    print(array);
    
    
    
    
    
    









return 0;
}