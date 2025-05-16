#include <iostream>
#include <chrono>

using namespace std;
using namespace chrono;

#define START_TIMER auto start = high_resolution_clock::now();
#define END_TIMER auto end = high_resolution_clock::now(); \
                  cout << "Execution Time: " << duration_cast<milliseconds>(end - start).count() << " ms\n";


void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1, n2 = right - mid;
    int L[n1], R[n2];

    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int i = 0; i < n2; i++) R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

int main() {
    START_TIMER
    int arr[] ={71, 31, 37, 35, 31, 14, 64, 69, 59, 91,  
        19, 76, 62, 28, 89, 78, 78, 25, 53, 73,  
        25, 16, 77, 37, 29, 59, 80, 65, 74, 88,  
        73, 46, 89, 73, 46, 37, 92, 37, 90, 96,  
        16, 88, 36, 25, 91, 86, 48, 80, 12, 59,  
        41, 13, 49, 86, 87, 16, 75, 45, 46, 98,  
        89, 46, 55, 20, 51, 42, 63, 10, 21, 55,  
        90, 63, 24, 57, 49, 49, 36, 12, 63, 68,  
        21, 34, 40, 11, 55, 65, 44, 91, 35, 68,  
        13, 94, 96, 58, 15, 24, 21, 20, 30, 84}, n = 100;
    mergeSort(arr, 0, n - 1);
    for (int i=0;i<100;i++) cout <<arr[i] << " ";
    END_TIMER
    return 0;
}