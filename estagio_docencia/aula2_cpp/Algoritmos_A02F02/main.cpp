#include <iostream>

using namespace std;

/*
    Faça um programa que leia dois valores reais representando duas notas de um aluno e dois valores inteiros representando os pesos das notas.
    Calcule e imprima a média ponderada das notas, considerando os pesos.
*/

void mediaPonderada(float n1, float n2, float p1, float p2) {
    cout.precision(1); cout << fixed;

    cout << "Media: " << ((n1*p1) + (n2*p2)) / (p1 + p2) << endl;

    return;
}

int main() {
    float _n1, _n2;
    int _p1, _p2;

    cin >> _n1;
    cin >> _n2;
    cin >> _p1;
    cin >> _p2;

    mediaPonderada(_n1, _n2, _p1, _p2);

    return 0;
}
