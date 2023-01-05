#include <iostream>

using namespace std;

/*
    João fez uma pesquisa em seu site de busca predileto, e encontrou a resposta que estava procurando no terceiro link listado.
    Além disso, ele viu, pelo site, que 't' pessoas já haviam clicado neste link antes.
    João havia lido anteriormente, também na Internet, que o número de pessoas que clicam no segundo link listado é
        o dobro de número de pessoas que clicam no terceiro link listado.
    Nessa leitura, ele também descobriu que o número de pessoas que clicam no segundo link é a metade do número de pessoas que clicam no primeiro link.

    João está intrigado para saber quantas pessoas clicaram no primeiro link da busca, e, como você é amigo dele, quer sua ajuda nesta tarefa.

Entrada:
    A entrada possui apenas um número 't' que representa o número de pessoas que clicaram no terceiro link da busca.

Saída:
    Imprima apenas uma linha, contendo apenas um inteiro, indicando quantas pessoas clicaram no primeiro link, nessa busca.
*/

int main() {
    int t;

    cin >> t;

    // L2 = 2.L3
    // L2 = 1/2.L1
    // 1/2.L1 = 2.L3
    // L1 = 2.2.L3 = 4.L3 (t)
    cout << 4*t << endl;

    return 0;
}
