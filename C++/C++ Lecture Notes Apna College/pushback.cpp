#include<iostream>
using namespace std;

int main(){

vector<int> marks;
marks.push_back(25);
cout<<marks.size();
marks.push_back(35);
cout<<marks.size();
for(int i:marks){
    cout<<i<<endl;
}
cout<<marks[0];/*====*/cout<<marks.at(0);





return 0;
}