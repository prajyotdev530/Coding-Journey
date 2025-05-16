 #include<iostream>
 using namespace std;
 
 int main(){
 
//structure is a user defined data type availaible in c++ used to combine diffrent diffrent type of
//types of items and for sme type of items we used arrays
struct  employee
{
    int favnumber;
    float salary;
    char favchar;

};
struct employee prajyot;
prajyot.favnumber=1;
prajyot.salary=1200000000;
    


cout<<"the value  is"<<prajyot.favnumber<<endl;
cout<<"the value  is"<<prajyot.salary<<endl;
cout<<"the value  is"<<prajyot.favchar<<endl;
struct player{
bool single;
    int height;
    float weight;
};
struct player rahul;
rahul.height=156;
rahul.weight=57;
rahul.single=0;
//---------->>>>>>>>>>>------>>>>>>>
cout<<rahul.weight;
cout<<rahul.single<<endl;
typedef struct member{

    int age;
}hii;
//this means that we can
// replace struct member by hii
hii vishal;
vishal.age=23;

cout<<vishal.age;


 return 0;
 }