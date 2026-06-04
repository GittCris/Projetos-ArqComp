/*
 * DGEMM - otimizacoes: "DGEMM final" (apex consolidado).
 *
 * Base = kernel do cap6 (blocked + unroll + AVX2 + OpenMP), com:
 *   - __restrict em A,B,C: o compilador prova nao-aliasing entre os stores de
 *     C e os loads de A/B no hotspot do_block() (apontado pelo VTune),
 *     melhorando o escalonamento de load/store. Ganho esperado: poucos %.
 *   - BLOCKSIZE=32, UNROLL=4 fixos (seguranca de L1; achados dos caps 4/5).
 *
 * Switches de compilacao (variantes isoladas):
 *   -DPREFETCH   secao 5.3: _mm_prefetch da proxima coluna de A no laco k.
 *   -DHUGEPAGES  secao 5.7: madvise(MADV_HUGEPAGE) nas matrizes -> menos TLB
 *                misses (impacto cresce com n).
 *
 * Numero de threads vem de OMP_NUM_THREADS. Imprime checksum, time_sec (ns)
 * e num_threads em stderr.
 */

#define _GNU_SOURCE

#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef HUGEPAGES
#include <sys/mman.h>
#endif

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

/* distancia de prefetch (em iteracoes de k) */
#ifndef PFDIST
#define PFDIST 4
#endif

static void do_block(int n, int si, int sj, int sk,
                     const double *restrict A, const double *restrict B,
                     double *restrict C)
{
    for (int i = si; i < si + BLOCKSIZE; i += UNROLL * VEC) {
        for (int j = sj; j < sj + BLOCKSIZE; ++j) {
            __m256d c[UNROLL];
            for (int r = 0; r < UNROLL; ++r)
                c[r] = _mm256_load_pd(C + i + r * VEC + j * n);

            for (int k = sk; k < sk + BLOCKSIZE; ++k) {
#ifdef PREFETCH
                /* traz a coluna de A de PFDIST iteracoes a frente */
                if (k + PFDIST < sk + BLOCKSIZE)
                    _mm_prefetch((const char *)(A + i + (k + PFDIST) * n),
                                 _MM_HINT_T0);
#endif
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

static void dgemm(int n, const double *restrict A, const double *restrict B,
                  double *restrict C)
{
    #pragma omp parallel for
    for (int sj = 0; sj < n; sj += BLOCKSIZE)
        for (int si = 0; si < n; si += BLOCKSIZE)
            for (int sk = 0; sk < n; sk += BLOCKSIZE)
                do_block(n, si, sj, sk, A, B, C);
}

static double *alloc_matrix(size_t bytes)
{
    double *p = (double *)aligned_alloc(32, bytes);
#ifdef HUGEPAGES
    if (p)
        madvise(p, bytes, MADV_HUGEPAGE);
#endif
    return p;
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
    double *A = alloc_matrix(bytes);
    double *B = alloc_matrix(bytes);
    double *C = alloc_matrix(bytes);
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
    for (size_t k = 0; k < total; ++k)
        s += C[k];

    fprintf(stderr, "checksum=%.6f\n", s);
    fprintf(stderr, "time_sec=%.9f\n", elapsed);
    fprintf(stderr, "num_threads=%d\n", omp_get_max_threads());

    free(A);
    free(B);
    free(C);
    return 0;
}
