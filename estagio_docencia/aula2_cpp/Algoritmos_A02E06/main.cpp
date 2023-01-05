#include <iostream>

using namespace std;

/**
    Fa�a um programa que leia um valor de sal�rio m�nimo e o valor do sal�rio de um funcion�rio.
    Seu programa deve calcular e imprimir quantos sal�rios m�nimos este funcion�rio ganha.
    Utilize 2 casas decimais em suas impress�es.
*/

void calcularProporcao(float sm, float sf) {
    cout.precision(2); cout << fixed;
    cout << "O sal�rio do funcion�rio equivale a " << sf/sm << " sal�rios m�nimos." << endl;

    return;
}

int main() {
    float _sm, _sf;

    cin >> _sm;
    cin >> _sf;

    calcularProporcao(_sm, _sf);

    return 0;
}
