#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>

#define UNROLL (4)

#define N 1024

void dgemm(int n, double* A, double* B, double* C) {
    for (int i = 0; i < n; i += UNROLL * 2) {
        for (int j = 0; j < n; ++j) {
            float64x2_t c[UNROLL];

            for (int r = 0; r < UNROLL; r++) {
                c[r] = vld1q_f64(C + i + r * 2 + j * n);
            }

            for (int k = 0; k < n; k++) {
                // Broadcast the single B element to both lanes of a 128-bit register
                float64x2_t bb = vdupq_n_f64(*(B + j * n + k));

                // Parallel fused multiply-add instructions
                for (int r = 0; r < UNROLL; r++) {
                    float64x2_t a_vec = vld1q_f64(A + n * k + r * 2 + i);
                    c[r] = vfmaq_f64(c[r], a_vec, bb);
                }
            }

            for (int r = 0; r < UNROLL; r++) {
                vst1q_f64(C + i + r * 2 + j * n, c[r]);
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

        printf("Starting DGEMM (ARM NEON) for matrix order %d...\n", n);

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