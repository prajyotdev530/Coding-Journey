6
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <termios.h>

// Function to simulate getch() in macOS/Linux
int getch(void) {
    struct termios oldt, newt;
    int ch;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    ch = getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return ch;
}

// Function to clear the screen
void clrscr() {
    system("clear");
}

int width = 20, height = 20;
int gameOver, score;
int x, y, fruitX, fruitY, flag;
int tailX[100], tailY[100];
int nTail = 0;

void setup() {
    gameOver = 0;
    x = width / 2;
    y = height / 2;
    
    srand(time(0)); // Seed for random number
    fruitX = rand() % width;
    fruitY = rand() % height;
}

int main() {
    setup();
    printf("Game setup done! Press any key to continue...\n");
    getch(); // Wait for user input
    clrscr(); // Clear screen
    
    printf("Fruit Position: (%d, %d)\n", fruitX, fruitY);
    return 0;
}