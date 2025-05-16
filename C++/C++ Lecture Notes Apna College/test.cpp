#include <iostream>
using namespace std;

int reverseInteger(int num) {
    int reversed = 0;
    while (num != 0) {
        int digit = num % 10;   // Extract the last digit
        reversed = reversed * 10 + digit;  // Add it to the reversed number
        num /= 10;  // Remove the last digit
    }
    return reversed;
}

int main() {
    int number;
    cout<< "Enter an integer: ";
    cin>> number;

    int reversedNumber = reverseInteger(number);
    cout << "Reversed integer: " << reversedNumber << endl;

    return 0;
}