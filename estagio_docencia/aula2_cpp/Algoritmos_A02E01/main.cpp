#include <iostream>

using namespace std;

/*
Faça um programa que leia a altura e o diâmetro de um copo e imprima na tela o seu volume. Considere para este exercício que pi = 3.141592.

Dica: O volume de um cilindro é dado por:

V = pi * r^2 * h

onde 'r' é o raio e 'h' é a altura do cilindro.
*/

#define PI 3.141592

double calcularVolume(double r, double h) {
    return (PI * (r*r) * h);
}

int main() {
    double _h, _d;

    cin >> _h;
    cin >> _d;

    cout << "O volume do copo é: " << calcularVolume(_d/2.0, _h) << " cm3." << endl;

    return 0;
}
