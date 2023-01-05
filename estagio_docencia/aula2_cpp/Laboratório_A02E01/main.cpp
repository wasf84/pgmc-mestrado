#include <iostream>

using namespace std;

/*
    Escreva um programa que leia dois valores inteiros do teclado, efetue as seguintes operações matemáticas:
        adição, subtração, multiplicação, divisão e módulo (resto da divisão) e imprima os resultados.
*/

int main() {
    int a, b;

    cin >> a;
    cin >> b;

    cout << a << " + " << b << " = " << a+b << endl;
    cout << a << " - " << b << " = " << a-b << endl;
    cout << a << " * " << b << " = " << a*b << endl;
    cout << a << " / " << b << " = " << a/b << endl;
    cout << a << " % " << b << " = " << a%b << endl;

    return 0;
}
