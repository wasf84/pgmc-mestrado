#include <iostream>

using namespace std;

/**
    Faça um programa que leia um total de segundos e imprima na tela o equivalente em dias, horas, minutos e segundos.
*/

void converter(int n)
{
    int d = n / (24 * 3600);
    n %= (24 * 3600); // vai pegando o resto pra calcular as demais grandezas

    int h = n / 3600;
    n %= 3600;

    int m = n / 60 ;
    n %= 60;

    int s = n;

    cout << "Conversão: " << d << " " << "dias, " <<
            h << " " << "horas, " <<
            m << " " << "minutos e " <<
            s << " " << "segundos." <<
            endl;
}

int main() {
    int _n;

    cin >> _n;
    converter(_n);

    return 0;
}
