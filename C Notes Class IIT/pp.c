#include <stdio.h>

// Function to sort the array in increasing order
void sortArray(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                // Swap arr[j] and arr[j+1]
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

int main() {
    int arr[] = {5, 3, 8, 1, 2}; // Example array
    int n = sizeof(arr) / sizeof(arr[0]); // Calculate the size of the array

    // Sort the array
    sortArray(arr, n);

    // Print the sorted array
    printf("Array in increasing order:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }

    return 0;
}