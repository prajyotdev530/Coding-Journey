#include <iostream>
using namespace std;
class farmer
{
    int a=4;
    int b=5;

public:
    void setnum();
    friend void add(farmer sarang);
};
void farmer::setnum()
{
    a = 4;
    b = 5;
}
void add(farmer sarang)
{
    int c=a+b;
}

int main()
{
    farmer sarang,prajyot;
    sarang.setnum();
    add(sarang);
    return 0;
}