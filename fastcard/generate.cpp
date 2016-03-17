// Generate some data to test fastcard with

#include <stdio.h>
#include <stdint.h>

int n = 30000;

int main() {
    FILE* f = fopen("test.dat", "wb");

    uint8_t a = 0, b = 0;
    for (int i = 0; i < n; ++i) {
        fprintf(f, "%c%c", a, b);
        
        ++b;
        if (b == 0) {
            ++a;
        }
    }

    fclose(f);
}
