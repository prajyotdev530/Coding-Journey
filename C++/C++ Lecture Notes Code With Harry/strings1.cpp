#include<iostream>
using namespace std;
//anything that we write in "" is a string ex "prajyot more" is a string
int main(){
    int hii[]={1,2,3};//=integer arrays
char str[]={'a' ,'b','\0'};//=character array//for converting this to string add /0 as end character of array
//here in character arrys the compiler itself add /o as the end memnber of array whose size is 1 byte
//we can use char array to store strings if we add /0 as the end member of the array
//
cout<<hii;//here we get an memory address 
cout<<endl;
cout<<str;//here we get all the element combined and printed
//unless we add /0 at end of char array it doesnt became a valid string
cout<<strlen(str);//in strlen the null chaaracter iss ignored
//for assigning direct a string without individual characters we can do it by
char string[]= "hello";//this is called a string literals//literal means whose value doesnt change//when character array is used to store a string here it is stored like an array only but at last column we has a null character
cout<<string[2];


return 0;
}