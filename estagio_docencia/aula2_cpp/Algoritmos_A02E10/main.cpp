#include <iostream>

using namespace std;

/*
    Faça um programa que leia do teclado um intervalo de tempo em segundos e imprima na tela sua conversão em horas, minutos e segundos.
*/

void converter(int n) {
    int h = n / 3600;
    n %= 3600;

    int m = n / 60 ;
    n %= 60;

    int s = n;

    cout << "Conversão: " << h << " " << "horas, " <<
                             m << " " << "minutos e " <<
                             s << " " << "segundos. " <<
            endl;

    return;
}

int main() {
    int _n;

    cin >> _n;

    cout << "Total: " << _n << " segundos." << endl;
    converter(_n);

    return 0;
}
