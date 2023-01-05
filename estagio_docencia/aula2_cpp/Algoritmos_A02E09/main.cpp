#include <iostream>

using namespace std;

/*
    Faça um programa que leia o preço de um produto e imprima na tela o valor deste com 25% de desconto.
    Imprima o resultado em moeda real (R$) utilizando 2 casas decimais.
*/

void calcularDesconto(float p) {
    cout.precision(2); cout << fixed;
    cout << "O valor do produto com 25% desconto é de R$" << p * 0.75 << "." << endl;

    return;
}

int main() {
    float _p;

    cin >> _p;
    calcularDesconto(_p);

    return 0;
}
