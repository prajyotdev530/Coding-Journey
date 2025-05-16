#include<iostream>

using namespace std;
vector<int> pairsum(vector<int> &nums){
vector<int> ans;
int target=9;
int i=0;
int j=nums.size()-1;
while(i<j){

int pairsum=nums[i]+nums[j];
if(pairsum<target){
    i++;
}
else if(target<pairsum){
    j++;
}
else{
    ans.push_back(i);
    ans.push_back(j);
    
}



}
return ans;

}

int main(){
vector<int>nums={8,3,4,5};
int target=9;
pairsum(nums);





return 0;
}