#include <iostream>
#include <cmath>

using namespace std;

/*
    Faça um programa que leia do teclado o peso (em quilos) e a altura (em metros) e calcule o índice de massa corporal (IMC) de uma pessoa.
    Seu programa deve imprimir o resultado conforme mostrado no exemplo a seguir.
    O IMC é determinado pela divisão do peso da pessoa pelo quadrado de sua altura.
*/

void calcularIMC(float p, float a) {
    cout.precision(2); cout << fixed;
    cout << "Peso: " << p << " kg, Altura: " << a << " m -> IMC: " << p / pow(a, 2) << endl;

    return;
}

int main() {
    float _p, _a;

    cin >> _p;
    cin >> _a;

    calcularIMC(_p, _a);

    return 0;
}
