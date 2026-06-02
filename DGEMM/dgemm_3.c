#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

void dgemm(size_t n, double *A, double *B, double *C)
{
    // O loop avança de 4 em 4 porque o vetor AVX processa 4 doubles por vez
    for (size_t i = 0; i < n; i += 4)
    {
        for (size_t j = 0; j < n; ++j)
        {
            __m256d c0 = _mm256_load_pd(C + i + j * n);
            for (size_t k = 0; k < n; ++k)
            {
                __m256d a = _mm256_load_pd(A + i + k * n);
                __m256d b = _mm256_broadcast_sd(B + k + j * n);
                c0 = _mm256_add_pd(c0, _mm256_mul_pd(a, b));
            }
            _mm256_store_pd(C + i + j * n, c0);
        }
    }
}

int main()
{
    FILE *arquivo = fopen("resultados.txt", "a");

    if (arquivo == NULL) {
        fprintf(stderr, "Erro ao abrir resultados.txt\n");
        return 1;
    }

    // Espaço e cabeçalho da execução no arquivo (mantido idêntico)
    fprintf(arquivo, "\n\n");
    fprintf(arquivo, "===== DGEMM_CAP3 =====\n");
    fprintf(arquivo, "%-15s %-15s\n", "Tamanho (N)", "Tempo (s)");
    fprintf(arquivo, "---------------------------------\n");

    // Cabeçalho dos resultados no console
    printf("%-15s %-15s\n", "Tamanho (N)", "Tempo (s)");
    printf("---------------------------------\n");

    // Loop de 32 até 4096 dobrando de tamanho
    for (int n = 32; n <= 4096; n *= 2)
    {
        size_t size = n * n * sizeof(double);

        double *A = (double *)aligned_alloc(32, size);
        double *B = (double *)aligned_alloc(32, size);
        double *C = (double *)aligned_alloc(32, size);

        if (A == NULL || B == NULL || C == NULL)
        {
            fprintf(stderr, "Erro ao alocar memória para N = %d\n", n);
            if (A) free(A);
            if (B) free(B);
            if (C) free(C);
            break;
        }

        double tempo_total = 0.0;
        int repeticoes = 10;

        for (int rep = 0; rep < repeticoes; rep++)
        {
            // Garante dados limpos e C zerada antes de CADA uma das 10 corridas
            for (int i = 0; i < n * n; i++)
            {
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

        // Imprime e grava o tempo médio mantendo a formatação antiga %-15.6f
        printf("%-15d %-15.6f\n", n, tempo_medio);
        fprintf(arquivo, "%-15d %-15.6f\n", n, tempo_medio);

        free(A);
        free(B);
        free(C);
    }

    fclose(arquivo);

    return 0;
}


//gcc -O3 -mavx dgemm_3.c -o dgemm_avx 
// ./dgemm_avx