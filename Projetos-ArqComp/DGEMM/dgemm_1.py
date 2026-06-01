import time

def dgemm(n):
    A = [[1.0 for _ in range(n)] for _ in range(n)]
    B = [[1.0 for _ in range(n)] for _ in range(n)]
    C = [[0.0 for _ in range(n)] for _ in range(n)]

    start = time.perf_counter()

    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]

    end = time.perf_counter()

    return end - start


def benchmark(valores, repeticoes=10):
    resultados = []

    for n in valores:
        tempos = []

        for _ in range(repeticoes):
            tempos.append(dgemm(n))

        media = sum(tempos) / repeticoes
        resultados.append((n, media))

        print(f"Matriz: {n}x{n}")
        print(f"Tempo médio ({repeticoes} execuções): {media:.6f}s\n")

    return resultados


def imprimir_resultados(resultados):
    print("+----------------------+----------------------+")
    print("| Ordem da Matriz (N)  | Tempo de Execução(s) |")
    print("+----------------------+----------------------+")
    
    for n, tempo in resultados:
        print(f"| {n:<20} | {tempo:>20.6f} |")
    
    print("+----------------------+----------------------+")


tamanhos = [32, 64, 128, 256, 512, 1024, 2048]

resultados = benchmark(tamanhos)
resultados.extend(benchmark([4096], 1))

imprimir_resultados(resultados)