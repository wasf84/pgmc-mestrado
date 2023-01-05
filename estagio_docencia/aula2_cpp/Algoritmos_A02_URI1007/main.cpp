#include <iostream>

using namespace std;

/*
Desenvolva um algoritmo que leia quatro valores inteiros A, B, C e D.
A seguir, calcule e mostre a diferença do produto de A e B pelo produto de C e D segundo a fórmula: >>> DIFERENCA = (A * B - C * D). <<<

    Entrada:

A entrada contém contém 4 valores inteiros.

    Saída:

A saída deve seguir o formato dos exemplos abaixo, com a palavra "DIFERENCA" com todas as letras maiúsculas, e um espaço em branco antes e depois da igualdade.
*/

int calcularDiferenca(int a, int b, int c, int d) {
    return ((a * b) - (c * d));
}

int main() {
    int _a, _b, _c, _d;

    cin >> _a;
    cin >> _b;
    cin >> _c;
    cin >> _d;

    cout << "DIFERENCA = " << calcularDiferenca(_a, _b, _c, _d) << endl;

    return 0;
}
