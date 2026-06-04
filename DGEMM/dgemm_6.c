/*
 * DGEMM - Capitulo 6 (Going Faster: Multiple Processors / OpenMP)
 *
 * Adaptacao da Figura 6.31 do Patterson & Hennessy: a versao com cache
 * blocking + unrolling + AVX do cap5, com UMA linha adicional:
 *   #pragma omp parallel for
 * no laco externo (sj). Isso espalha os blocos-coluna de C entre as
 * threads. Como cada iteracao de sj escreve um conjunto disjunto de
 * colunas de C, nao ha condicao de corrida.
 *
 * AVX2 (VEC=4). Kernel com UNROLL acumuladores (cap4) + blocking (cap5).
 * Defaults do livro: BLOCKSIZE=32, UNROLL=4.
 *
 * Parametros: -DBLOCKSIZE=k (default 32), -DUNROLL=k (default 4).
 *   BLOCKSIZE deve ser multiplo de UNROLL*VEC; n multiplo de BLOCKSIZE.
 * Numero de threads vem de OMP_NUM_THREADS (ambiente).
 *
 * Imprime checksum, time_sec (clock_gettime wall-clock, ns) e num_threads.
 */

#define _POSIX_C_SOURCE 199309L

#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VEC 4

#ifndef UNROLL
#define UNROLL 4
#endif

#ifndef BLOCKSIZE
#define BLOCKSIZE 32
#endif

#if (BLOCKSIZE % (UNROLL * VEC)) != 0
#error "BLOCKSIZE deve ser multiplo de UNROLL*VEC"
#endif

static void do_block(int n, int si, int sj, int sk,
                     const double *A, const double *B, double *C)
{
    for (int i = si; i < si + BLOCKSIZE; i += UNROLL * VEC) {
        for (int j = sj; j < sj + BLOCKSIZE; ++j) {
            __m256d c[UNROLL];
            for (int r = 0; r < UNROLL; ++r)
                c[r] = _mm256_load_pd(C + i + r * VEC + j * n);

            for (int k = sk; k < sk + BLOCKSIZE; ++k) {
                __m256d bb = _mm256_broadcast_sd(B + k + j * n);
                for (int r = 0; r < UNROLL; ++r)
                    c[r] = _mm256_fmadd_pd(
                        _mm256_load_pd(A + i + r * VEC + k * n), bb, c[r]);
            }

            for (int r = 0; r < UNROLL; ++r)
                _mm256_store_pd(C + i + r * VEC + j * n, c[r]);
        }
    }
}

static void dgemm(int n, const double *A, const double *B, double *C)
{
    #pragma omp parallel for
    for (int sj = 0; sj < n; sj += BLOCKSIZE)
        for (int si = 0; si < n; si += BLOCKSIZE)
            for (int sk = 0; sk < n; sk += BLOCKSIZE)
                do_block(n, si, sj, sk, A, B, C);
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "uso: %s N\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    if (n <= 0 || (n % BLOCKSIZE) != 0) {
        fprintf(stderr, "N deve ser multiplo de BLOCKSIZE=%d (recebido %d)\n",
                BLOCKSIZE, n);
        return 1;
    }

    size_t total = (size_t)n * (size_t)n;
    size_t bytes = total * sizeof(double);
    double *A = (double *)aligned_alloc(32, bytes);
    double *B = (double *)aligned_alloc(32, bytes);
    double *C = (double *)aligned_alloc(32, bytes);
    if (!A || !B || !C) {
        fprintf(stderr, "falha de alocacao para n=%d\n", n);
        return 2;
    }

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            A[i + j * n] = (double)(i + j);
            B[i + j * n] = (double)(i - j);
            C[i + j * n] = 0.0;
        }
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    dgemm(n, A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (double)(t1.tv_sec - t0.tv_sec)
                   + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;

    double s = 0.0;
    for (size_t k = 0; k < total; ++k) {
        s += C[k];
    }
    fprintf(stderr, "checksum=%.6f\n", s);
    fprintf(stderr, "time_sec=%.9f\n", elapsed);
    fprintf(stderr, "num_threads=%d\n", omp_get_max_threads());

    free(A);
    free(B);
    free(C);
    return 0;
}
