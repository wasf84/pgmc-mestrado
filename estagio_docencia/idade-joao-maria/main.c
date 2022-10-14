#include <stdio.h>
#include <stdlib.h>

/*
Considere o enunciado:
- João tem o dobro da idade de Maria. Subtraindo 4 anos da idade de Maria, obtemos 1/3 de sua idade.

Crie uma função principal (main) que resolva este problema de maneira iterativa e então imprima as idades de Maria e João.
*/

int main(void) {
    int maria, joao, c;

    c = 0;
    joao = 0;
    maria = 0;
    while (!c) {
        if ((2 * maria) - 12 == 0) {
            joao = 2 * maria;
            c = 1;
        } else {
            maria++;
        }
    }

    fprintf(stdout, "Idade de Maria...: %d\n", maria);
    fprintf(stdout, "Idade de Joao...: %d\n", joao);

    return 0;
}
