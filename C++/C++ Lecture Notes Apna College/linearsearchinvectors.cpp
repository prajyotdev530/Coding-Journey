#include<iostream>
using namespace std;





int main(){
    vector<int>nums={1,2,3,4,5};
    int target=6;
   for(int i=0;i<nums.size();i++){

    if(nums[i]==target){
        cout<<"you found it";
        return 0;
    }
   }
   cout<<"bhad me ja";



return 0;
}
