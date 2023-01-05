#include <iostream>
#include <math.h>

using namespace std;

/*
    Faça um programa que calcule e mostre o volume de uma esfera sendo fornecido o valor de seu raio (R).
    A fórmula para calcular o volume é: (4/3) * pi * R^3. Considere (atribua) para pi o valor 3.14159.

    Dica: Ao utilizar a fórmula, procure usar (4/3.0) ou (4.0/3), pois algumas linguagens (dentre elas o C),
    assumem que o resultado da divisão entre dois inteiros é outro inteiro.

Entrada:

    A entrada contém um valor de ponto flutuante (dupla precisão, ou seja, double), correspondente ao raio da esfera.

Saída:

    A saída deverá ser uma mensagem "VOLUME = " conforme o exemplo fornecido abaixo, com um espaço antes e um espaço depois da igualdade.
    O valor deverá ser apresentado com 3 casas após o ponto.
*/

#define PI 3.14159

void calcularVolumeEsfera(double r) {
    cout.precision(3); cout << fixed;

    cout << "VOLUME = " << ((4/3.0) * PI * pow(r, 3)) << endl;

    return;
}

int main() {
    double _r;

    cout << "Valor do raio R...: ";
    cin >> _r;
    cout << endl;

    calcularVolumeEsfera(_r);

    return 0;
}