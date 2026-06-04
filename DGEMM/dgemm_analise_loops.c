/*
 * DGEMM - otimizacoes: estudo de ordem de lacos + transpor B + ponteiros.
 *
 * Kernel ESCALAR (nao SIMD) - forma honesta de reproduzir o efeito de cache
 * da Fig. 5.19 do livro: so o laco triplo escalar deixa testar as 6
 * permutacoes e expoe o efeito de "stride" do acesso a memoria.
 *
 * Switches de compilacao:
 *   -DORDER=0..5   ->  ijk, ikj, jik, jki, kij, kji   (#if ORDER==n)
 *   -DTRANSPOSE_B  ->  usa Bt[j+k*n] (B contiguo em k) - secao 5.4
 *   -DPOINTERS     ->  versao jki com aritmetica de PONTEIROS - secao 2.14
 *                      (ignora ORDER; compara com a versao por indice)
 *
 * Layout column-major: C[i+j*n], A[i+k*n], B[k+j*n]. Init A=i+j, B=i-j.
 * Imprime checksum e time_sec (clock_gettime, ns) em stderr.
 *
 * Previsao (column-major): i no laco interno -> stride-1 em C e A ->
 *   jki/kji otimos; k interno -> A vira stride-n -> ijk/jik ruins;
 *   j interno -> kij/ikj piores.
 */

#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef ORDER
#define ORDER 0
#endif

/* Leitura de B: transposta (Bt[j+k*n]) ou normal (B[k+j*n]) */
#ifdef TRANSPOSE_B
  #define BVAL(B, k, j, n) (B)[(j) + (k) * (n)]
#else
  #define BVAL(B, k, j, n) (B)[(k) + (j) * (n)]
#endif

static void dgemm(int n, const double *A, const double *B, double *C)
{
#ifdef POINTERS
    /* Secao 2.14: jki percorrido com ponteiros incrementais. */
    for (int j = 0; j < n; ++j) {
        double *c = C + (size_t)j * n;          /* coluna j de C */
        for (int k = 0; k < n; ++k) {
            double bkj = BVAL(B, k, j, n);
            const double *a = A + (size_t)k * n; /* coluna k de A */
            double *cc = c;
            const double *aa = a;
            for (int i = 0; i < n; ++i)
                *cc++ += *aa++ * bkj;
        }
    }
#else
  #if ORDER == 0   /* ijk */
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k)
          C[i + j * n] += A[i + k * n] * BVAL(B, k, j, n);
  #elif ORDER == 1 /* ikj */
    for (int i = 0; i < n; ++i)
      for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
          C[i + j * n] += A[i + k * n] * BVAL(B, k, j, n);
  #elif ORDER == 2 /* jik */
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
          C[i + j * n] += A[i + k * n] * BVAL(B, k, j, n);
  #elif ORDER == 3 /* jki */
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k)
        for (int i = 0; i < n; ++i)
          C[i + j * n] += A[i + k * n] * BVAL(B, k, j, n);
  #elif ORDER == 4 /* kij */
    for (int k = 0; k < n; ++k)
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          C[i + j * n] += A[i + k * n] * BVAL(B, k, j, n);
  #elif ORDER == 5 /* kji */
    for (int k = 0; k < n; ++k)
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
          C[i + j * n] += A[i + k * n] * BVAL(B, k, j, n);
  #else
    #error "ORDER deve ser 0..5"
  #endif
#endif
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "uso: %s N\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    if (n <= 0) {
        fprintf(stderr, "N invalido: %s\n", argv[1]);
        return 1;
    }

    size_t total = (size_t)n * (size_t)n;
    size_t bytes = total * sizeof(double);
    double *A = (double *)malloc(bytes);
    double *B = (double *)malloc(bytes);
    double *C = (double *)calloc(total, sizeof(double));
    if (!A || !B || !C) {
        fprintf(stderr, "falha de alocacao para n=%d\n", n);
        return 2;
    }

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            A[i + j * n] = (double)(i + j);
            B[i + j * n] = (double)(i - j);
        }
    }

    const double *Buse = B;
#ifdef TRANSPOSE_B
    /* Bt[j+k*n] = B[k+j*n] (transposicao - custo O(n^2) de SETUP, fora do
       cronometro, como o init). */
    double *Bt = (double *)malloc(bytes);
    if (!Bt) { fprintf(stderr, "alloc Bt falhou\n"); return 2; }
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            Bt[j + k * n] = B[k + j * n];
    Buse = Bt;
#endif

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    dgemm(n, A, Buse, C);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (double)(t1.tv_sec - t0.tv_sec)
                   + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;

    double s = 0.0;
    for (size_t k = 0; k < total; ++k)
        s += C[k];

    fprintf(stderr, "checksum=%.6f\n", s);
    fprintf(stderr, "time_sec=%.9f\n", elapsed);

    free(A);
    free(B);
    free(C);
#ifdef TRANSPOSE_B
    free(Bt);
#endif
    return 0;
}
