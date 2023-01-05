#include <iostream>

using namespace std;

/**
    Faça um programa que leia uma temperatura em graus Celsius e apresente-a convertida em graus Fahrenheit. A fórmula de conversão é: F = (9.0*C + 160.0)/5.0.
*/

void converterTemperatura(double t) {
    cout.precision(2); cout << fixed;
    cout << "Fahrenheit: " << (9.0*t + 160.0)/5.0 << endl;
    return;
}


int main() {
    double _t;

    cin >> _t;

    cout.precision(2); cout << fixed;
    cout << "Celsius: " << _t << endl;
    converterTemperatura(_t);

    return 0;
}
