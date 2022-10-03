#include <stdio.h>
#include <stdlib.h>

int main() {
    int _, num1, num2;

    scanf("%d", &num1);
    scanf("%d", &num2);
    fprintf(stdout, "----\n");

    if (num1 <= num2) {
        for (_=num1; _<=num2; _++)
            if (_ % 11 == 5) fprintf(stdout, "%d\n", _);
    } else { // num1 > num2
        for (_=num2; _<=num1; _++)
            if (_ % 11 == 5) fprintf(stdout, "%d\n", _);
    }

    return 0;
}

