#include <math.h>
#include <iostream>

using namespace std;

/**
    Faça um programa que leia um valor inteiro do teclado e imprima na tela o seu dobro, triplo e quadrado.
*/

int calcular(int n) {
    cout << "Valor: " << n << endl;

    cout << "Dobro: "   << 2*n  << ", " <<
            "Triplo: "  << 3*n  << ", " <<
            "Quadrado: " << pow(n, 2)  <<
            endl;

    return 0;
}

int main() {
    int _n;

    cin >> _n;
    calcular(_n);

    return 0;
}
