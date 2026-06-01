#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void dgemm(int n, double* A, double* B, double* C) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double cij = C[i + j * n];
            for (int k = 0; k < n; ++k) {
                cij += A[i + k * n] * B[k + j * n];
            }
            C[i + j * n] = cij;
        }
    }
}

int main() {
    FILE *arquivo = fopen("resultados.txt", "a");

    if (arquivo == NULL) {
        printf("Erro ao abrir resultados.txt\n");
        return 1;
    }

    // Cabeçalho
    printf("%-15s %-15s\n", "Tamanho (N)", "Tempo (s)");
    printf("---------------------------------\n");

    // Espaço e cabeçalho da execução
    fprintf(arquivo, "\n\n");
    fprintf(arquivo, "===== DGEMM_CAP3 =====\n");
    fprintf(arquivo, "%-15s %-15s\n", "Tamanho (N)", "Tempo (s)");
    fprintf(arquivo, "---------------------------------\n");

    for (int n = 32; n <= 4096; n *= 2) {

        double* A = (double*)malloc(n * n * sizeof(double));
        double* B = (double*)malloc(n * n * sizeof(double));
        double* C = (double*)malloc(n * n * sizeof(double));

        if (A == NULL || B == NULL || C == NULL) {
            fprintf(stderr, "Erro ao alocar memória para N = %d\n", n);
            break;
        }

        for (int i = 0; i < n * n; i++) {
            A[i] = 1.0;
            B[i] = 1.0;
            C[i] = 0.0;
        }

        clock_t start = clock();
        dgemm(n, A, B, C);
        clock_t end = clock();

        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

        printf("%-15d %-15.4f\n", n, elapsed);
        fprintf(arquivo, "%-15d %-15.4f\n", n, elapsed);

        free(A);
        free(B);
        free(C);
    }

    fclose(arquivo);

    return 0;
}