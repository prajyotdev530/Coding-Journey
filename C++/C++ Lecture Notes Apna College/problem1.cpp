#include <iostream>

using namespace std;

void f(int r, int l, int n, int arr[]) {
    int sum = 0;

    for (int i = 1; i < n; i++) {  // Loop should run from 1 to n-1
        sum += (n % i);
    }

    arr[n - l] = sum;  // Store sum at correct index
}

int main() {
    int r, l;
    cin >> r >> l;

    if (r < l) {  // Ensure r is greater than l
        swap(r, l);
    }

    int size = r - l + 1;
    int arr[size];

    for (int i = 0; i < size; i++) {
        f(r, l, l + i, arr);
    }

    for (int i = 0; i < size - 1; i++) {
        if (arr[i] == arr[i + 1]) {
            cout << (i + l + 1) << " ";
        }
    }

    return 0;
}