#include <iostream>

using namespace std;

/**
    Faça um programa que leia um valor de salário mínimo e o valor do salário de um funcionário.
    Seu programa deve calcular e imprimir quantos salários mínimos este funcionário ganha.
    Utilize 2 casas decimais em suas impressões.
*/

void calcularProporcao(float sm, float sf) {
    cout.precision(2); cout << fixed;
    cout << "O salário do funcionário equivale a " << sf/sm << " salários mínimos." << endl;

    return;
}

int main() {
    float _sm, _sf;

    cin >> _sm;
    cin >> _sf;

    calcularProporcao(_sm, _sf);

    return 0;
}
