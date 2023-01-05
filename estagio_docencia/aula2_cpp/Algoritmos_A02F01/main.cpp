#include <iostream>

using namespace std;

/*
    Faça um programa que leia dois valores reais do teclado e imprima a soma dos valores.
    Dica: Utilize cout.precision(3) para imprimir com duas casas decimais.
*/

int main() {
    float _a, _b;

    cin >> _a;
    cin >> _b;

    cout.precision(2); cout << fixed;
    cout << "Soma de " << _a << " e " << _b << ": " << _a + _b << endl;

    return 0;
}
