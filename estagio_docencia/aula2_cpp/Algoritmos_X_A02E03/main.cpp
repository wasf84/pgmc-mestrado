#include <iostream>
#include <cmath>

using namespace std;

/*
    Leia os quatro valores correspondentes às coordenadas 'x' e 'y' de dois pontos quaisquer no plano, p1 = (x1,y1) e p2 = (x2,y2).
    Calcule a distância entre eles, mostrando 4 casas decimais após a vírgula, segundo a fórmula:

    distancia = sqrt( (x2 - x1)^2 + (y2 - y1)^2 )

Entrada:

    A entrada contém duas linhas de dados. A primeira linha contém dois valores de ponto flutuante (float): x1 y1 e a segunda linha contém dois valores de ponto flutuante x2 y2.

Saída:

    A saída deve conter o texto "Distancia:" seguido por um espaço e pela distância calculada com 4 casas após o ponto decimal.
*/

void calcularDistancia(float x1, float x2, float y1, float y2) {
    cout.precision(4); cout << fixed;

    cout << "Distancia: " << sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2)) << endl;

    return;
}

int main() {
    float _x1, _x2, _y1, _y2;

    cin >> _x1 >> _y1;
    cin >> _x2 >> _y2;

    calcularDistancia(_x1, _x2, _y1, _y2);

    return 0;
}
