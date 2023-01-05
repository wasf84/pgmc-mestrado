#include <iostream>
#include <assert.h>

using namespace std;

/*
    A seguir, calcule a média do aluno, sabendo que a nota A tem peso 2, a nota B tem peso 3 e a nota C tem peso 5.
    Desenvolva um algoritmo que leia 3 valores, no caso, variáveis A, B e C, que são as três notas de um aluno.
    Considere que cada nota pode ir de 0 até 10.0, sempre com uma casa decimal.

Entrada:

    A entrada contém 3 valores com uma casa decimal, de dupla precisão (double).

Saída:

    Imprima a mensagem "MEDIA = " e a média do aluno conforme exemplo abaixo, com 1 dígito após o ponto decimal e com um espaço em branco antes e depois da igualdade.
*/

void calcularMediaPonderada(double a, double b, double c) {
    cout.precision(1); cout << fixed;

    cout << "MEDIA = " << (a*2.0 + b*3.0 + c*5.0) / (2.0 + 3.0 + 5.0) << endl;

    return;

}

int main() {
    double _a, _b, _c;

    cin >> _a;
    cin >> _b;
    cin >> _c;

    // As notas precisam estar entre 0 e 10
    assert(_a >= 0 && _a <= 10);
    assert(_b >= 0 && _b <= 10);
    assert(_c >= 0 && _c <= 10);

    calcularMediaPonderada(_a, _b, _c);

    return 0;
}
