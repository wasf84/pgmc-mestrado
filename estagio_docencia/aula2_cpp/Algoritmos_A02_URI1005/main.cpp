#include <iostream>

using namespace std;

/*
Desenvolva um algoritmo que leia 2 valores de ponto flutuante de dupla precisão (double) A e B, que correspondem a 2 notas de um aluno.
A seguir, calcule a média do aluno, sabendo que a nota A tem peso 3.5 e a nota B tem peso 7.5 (A soma dos pesos, portanto, é 11).
Assuma que cada nota pode ir de 0 até 10.0, sempre com uma casa decimal.

    Entrada

A entrada contém contém 2 valores com uma casa decimal cada um.

    Saída

Imprima a mensagem "MEDIA" e a média do aluno conforme exemplo abaixo, com 5 dígitos após o ponto decimal e com um espaço em branco antes e depois da igualdade.
*/

#define P1 3.5
#define P2 7.5

double mediaPonderada(double a, double b) {
    return ((a * P1) + (b * P2)) / (P1 + P2);
}

int main() {
    double _a, _b;

    cin.precision(2);
    cin >> _a;
    cin >> _b;

    cout.precision(5); cout << fixed;
    cout << "MEDIA = " << mediaPonderada(_a, _b);

    return 0;
}
