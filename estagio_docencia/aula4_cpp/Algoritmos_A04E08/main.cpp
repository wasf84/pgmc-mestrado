#include <iostream>

using namespace std;

/**
Fa�a uma fun��o que receba o c�digo de um produto e imprima a sua origem de acordo com os crit�rios a seguir:
    C�digo do produto entre 01 e 20: "Europa";
    C�digo do produto entre 21 e 40: "�sia";
    C�digo do produto entre 41 e 60: "Am�rica";
    C�digo do produto entre 61 e 80: "�frica";
    C�digo do produto maior que 80: "Paraguai";
    C�digo fora das faixas de valores acima: "C�digo Inv�lido".

Fa�a uma fun��o principal (main) que leia o c�digo de um produto e chame a fun��o anterior.
*/

void origem(int o) {
    switch (o) {
        case 01 ... 20: cout << "Europa"  << endl; break;
        case 21 ... 40: cout << "�sia"    << endl; break;
        case 41 ... 60: cout << "Am�rica" << endl; break;
        case 61 ... 80: cout << "�frica"  << endl; break;

        default: cout << "ERRO" << endl; break;
    }

}

int main() {

    return 0;
}
