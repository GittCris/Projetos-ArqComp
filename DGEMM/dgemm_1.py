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


def inicializar_arquivo_resultados():
    # Cria ou limpa o arquivo e adiciona o cabeçalho inicial
    with open("resultados.txt", "w", encoding="utf-8") as f:
        f.write("+----------------------+----------------------+\n")
        f.write("| Ordem da Matriz (N)  | Tempo de Execução(s) |\n")
        f.write("+----------------------+----------------------+\n")


def salvar_resultado_no_arquivo(n, tempo):
    # Abre o arquivo no modo 'a' (append) para adicionar a linha sem apagar o que já existe
    with open("resultados.txt", "a", encoding="utf-8") as f:
        f.write(f"| {n:<20} | {tempo:>20.6f} |\n")
        # Força a escrita no disco imediatamente
        f.flush()


def fechar_moldura_arquivo():
    # Adiciona a linha de fechamento da tabela no final do arquivo
    with open("resultados.txt", "a", encoding="utf-8") as f:
        f.write("+----------------------+----------------------+\n")


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

        # MODIFICAÇÃO: Salva no arquivo imediatamente após terminar este tamanho
        salvar_resultado_no_arquivo(n, media)

    return resultados


def imprimir_resultados(resultados):
    print("+----------------------+----------------------+")
    print("| Ordem da Matriz (N)  | Tempo de Execução(s) |")
    print("+----------------------+----------------------+")

    for n, tempo in resultados:
        print(f"| {n:<20} | {tempo:>20.6f} |")

    print("+----------------------+----------------------+")


# --- Execução do Programa ---

# 1. Prepara o arquivo com o cabeçalho antes de começar os testes
inicializar_arquivo_resultados()

tamanhos = [32, 64, 128, 256, 512, 1024, 2048]

# 2. Roda o benchmark (ele vai salvando linha por linha no TXT)
resultados = benchmark(tamanhos)
resultados.extend(benchmark([4096], 1))

# 3. Fecha a moldura da tabela no arquivo TXT
fechar_moldura_arquivo()

# 4. Mostra o resultado final na tela
print("\nBenchmark concluído! Resultado final salvo em 'resultados.txt'.\n")
imprimir_resultados(resultados)