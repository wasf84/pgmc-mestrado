#include <iostream>

using namespace std;

/**
Faça uma função que receba o código de um produto e imprima a sua origem de acordo com os critérios a seguir:
    Código do produto entre 01 e 20: "Europa";
    Código do produto entre 21 e 40: "Ásia";
    Código do produto entre 41 e 60: "América";
    Código do produto entre 61 e 80: "África";
    Código do produto maior que 80: "Paraguai";
    Código fora das faixas de valores acima: "Código Inválido".

Faça uma função principal (main) que leia o código de um produto e chame a função anterior.
*/

void origem(int o) {
    switch (o) {
        case 01 ... 20: cout << "Europa"  << endl; break;
        case 21 ... 40: cout << "Ásia"    << endl; break;
        case 41 ... 60: cout << "América" << endl; break;
        case 61 ... 80: cout << "África"  << endl; break;

        default: cout << "ERRO" << endl; break;
    }

}

int main() {

    return 0;
}
