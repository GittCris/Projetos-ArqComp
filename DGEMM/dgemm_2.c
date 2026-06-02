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

    // Cabeçalho no console
    printf("%-15s %-15s\n", "Tamanho (N)", "Tempo Médio (s)");
    printf("---------------------------------\n");

    // Espaço e cabeçalho no arquivo de saída
    fprintf(arquivo, "\n\n");
    fprintf(arquivo, "===== DGEMM_CAP2 (Média de 10 execuções) =====\n");
    fprintf(arquivo, "%-15s %-15s\n", "Tamanho (N)", "Tempo Médio (s)");
    fprintf(arquivo, "---------------------------------\n");

    for (int n = 32; n <= 4096; n *= 2) {

        double* A = (double*)malloc(n * n * sizeof(double));
        double* B = (double*)malloc(n * n * sizeof(double));
        double* C = (double*)malloc(n * n * sizeof(double));

        if (A == NULL || B == NULL || C == NULL) {
            fprintf(stderr, "Erro ao alocar memória para N = %d\n", n);
            if (A) free(A);
            if (B) free(B);
            if (C) free(C);
            break;
        }

        double tempo_total = 0.0;
        int repeticoes = 10;

        for (int rep = 0; rep < repeticoes; rep++) {
            for (int i = 0; i < n * n; i++) {
                A[i] = 1.0;
                B[i] = 1.0;
                C[i] = 0.0; 
            }

            clock_t start = clock();
            dgemm(n, A, B, C);
            clock_t end = clock();

            tempo_total += (double)(end - start) / CLOCKS_PER_SEC;
        }

        double tempo_medio = tempo_total / repeticoes;

        // Mantém exatamente o mesmo formato de output solicitado
        printf("%-15d %-15.4f\n", n, tempo_medio);
        fprintf(arquivo, "%-15d %-15.4f\n", n, tempo_medio);

        free(A);
        free(B);
        free(C);
    }

    fclose(arquivo);

    return 0;
}