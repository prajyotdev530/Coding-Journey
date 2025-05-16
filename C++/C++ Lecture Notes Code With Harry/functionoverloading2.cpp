#include<iostream>
using namespace std;
//volume of cuboid
float volume(float height,float length,float width){


    return (height*length*width);
}
//volume of cube
float volume(int a){

    return a*a*a;
}


int main(){

float x,y,z;
cin>>x>>y>>z;
cout<<volume(x,y,z);
//if we give three arguments it automatically detects to give it to which function


return 0;
}