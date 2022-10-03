#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX 100

int randint() {
    srand(time(NULL));
    return rand() % (MAX + 1); // Gera um numero aleatorio de 0 ate MAX
}

int main() {
    int opt, resp, cont = 1;

    resp = randint(MAX);

    fprintf(stdout, ">>> %s <<<\n", "Jogo da adivinhação");
    fprintf(stdout, "%s %d:\n", "Digite um numero entre 0 e", MAX);
    scanf("%d", &opt);

    while (cont != 0) { // repete enquanto nao acertar o numero oculto
        if (opt == resp) {
            fprintf(stdout, "%s\n", "ACERTOU!!!");
            fprintf(stdout, "%s\n", "Bye-bye.");
            cont = 0;
        } else {
            if (opt > resp) {
                fprintf(stdout, "%s\n", "Menos.");
                scanf("%d", &opt);
            } else {
                fprintf(stdout, "%s\n", "Mais.");
                scanf("%d", &opt);
            }
        }
    }

    return 0;
}
