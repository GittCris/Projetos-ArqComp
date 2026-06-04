#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>

#define UNROLL (4)
#define BLOCKSIZE 32

#define N 1024

void do_block(int n, int si, int sj, int sk, double *A, double *B, double *C) {
    for (int i = si; i < si + BLOCKSIZE; i += UNROLL * 2) {
        for (int j = sj; j < sj + BLOCKSIZE; j++) {
            float64x2_t c[UNROLL];

            for (int r = 0; r < UNROLL; r++) {
                c[r] = vld1q_f64(C + (i + r * 2) + j * n);
            }

            for (int k = sk; k < sk + BLOCKSIZE; k++) {
                // Broadcast the B element to both lanes of the 128-bit register
                float64x2_t bb = vdupq_n_f64(*(B + k + j * n));

                // Parallel fused multiply-add
                for (int r = 0; r < UNROLL; r++) {
                    float64x2_t a_vec = vld1q_f64(A + n * k + (i + r * 2));
                    c[r] = vfmaq_f64(c[r], a_vec, bb);
                }
            }

            for (int r = 0; r < UNROLL; r++) {
                vst1q_f64(C + (i + r * 2) + j * n, c[r]);
            }
        }
    }
}

void dgemm(int n, double* A, double* B, double* C) {
    for (int sj = 0; sj < n; sj += BLOCKSIZE) {
        for (int si = 0; si < n; si += BLOCKSIZE) {
            for (int sk = 0; sk < n; sk += BLOCKSIZE) {
                do_block(n, si, sj, sk, A, B, C);
            }
        }
    }
}

int main() {
    int sizes[] = {256, 512, 1024, 2048, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];

        double* A = (double*)malloc(n * n * sizeof(double));
        double* B = (double*)malloc(n * n * sizeof(double));
        double* C = (double*)malloc(n * n * sizeof(double));

        if (A == NULL || B == NULL || C == NULL) {
            printf("Memory allocation failed for N=%d!\n", n);
            return 1;
        }

        for (int i = 0; i < n * n; i++) {
            A[i] = 1.0;
            B[i] = 1.0;
            C[i] = 0.0;
        }

        printf("Starting Cache Blocked DGEMM (ARM NEON) for matrix order %d...\n", n);

        clock_t start = clock();

        dgemm(n, A, B, C);

        clock_t end = clock();

        double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Tempo de execução (N=%d): %f segundos\n\n", n, elapsed_time);

        free(A);
        free(B);
        free(C);
    }

    return 0;
}