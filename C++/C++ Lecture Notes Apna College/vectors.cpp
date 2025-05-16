#include<iostream>
using namespace std;
int freq=0;
int sort(vector<int> vec,int n){
     for(int a:vec){
    
             freq=0;
             for(int b:vec){

                if(a==b){
                    freq++;
                }
             }
             if(freq>n/2){
                return a;
             }
            }



     }










int main(){
    vector<int> vec={1,1,1,2,2,2,2};
cout<<sort(vec,6);




    return 0;
}
