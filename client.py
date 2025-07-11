import numpy as np
import Pyro5.api
import time
import itertools
import math

# --- Configuração do Serializador Otimizado ---
Pyro5.config.SERIALIZER = "msgpack"

# --- Serialização para NumPy (necessária para o Pyro) ---
def numpy_array_to_dict(obj):
    return {
        "__class__": "numpy.ndarray",
        "dtype": obj.dtype.str,
        "data": obj.tolist(),
    }

def dict_to_numpy_array(classname, d):
    if classname == "numpy.ndarray":
        return np.array(d["data"], dtype=np.dtype(d["dtype"]))
    return d

Pyro5.api.register_class_to_dict(np.ndarray, numpy_array_to_dict)
Pyro5.api.register_dict_to_class("numpy.ndarray", dict_to_numpy_array)


# --- Configurações ---
MATRIX_SIZE = 1024
OUTPUT_FILENAME = "relatorio_desempenho.txt"
ORIGINAL_MATRIX_FILE = "matriz_original.txt"
INVERSE_MATRIX_FILE = "matriz_inversa.txt"


class WorkerPool:
    """Gerencia a conexão e distribuição de tarefas para os workers."""
    def __init__(self):
        ns = Pyro5.api.locate_ns()
        worker_names = sorted(ns.list(prefix="matrix.calculator.").keys())
        if not worker_names:
            raise RuntimeError("Nenhum worker disponível encontrado no Name Server.")
        
        print(f"Workers encontrados: {', '.join(worker_names)}")
        self.workers = [Pyro5.api.Proxy(f"PYRONAME:{name}") for name in worker_names]
        for w in self.workers:
            w._pyroBind()
        self._cycle = itertools.cycle(self.workers)

    def get_worker(self):
        """Obtém o próximo worker em um ciclo (round-robin)."""
        return next(self._cycle)

    def count(self):
        return len(self.workers)


def generate_invertible_matrix(size):
    """Gera uma matriz quadrada garantidamente invertível."""
    print(f"Gerando matriz invertível de tamanho {size}x{size}...")
    matrix = np.random.rand(size, size)
    matrix += np.eye(size) * size
    print("Matriz gerada com sucesso.")
    return matrix

def format_determinant(sign, logdet):
    """Formata o determinante em notação científica a partir do seu sinal e logaritmo."""
    if sign == 0:
        return "0.0"
    
    # Converte o log natural (base e) para log base 10
    log10_det = logdet / math.log(10)
    
    # Separa a mantissa do expoente
    exponent = math.floor(log10_det)
    mantissa = 10**(log10_det - exponent)
    
    return f"{sign * mantissa:.4f}e+{exponent}"


def main():
    """Função principal do cliente."""
    if (MATRIX_SIZE & (MATRIX_SIZE - 1)) != 0 or MATRIX_SIZE == 0:
        print(f"Erro: O tamanho da matriz ({MATRIX_SIZE}) deve ser uma potência de 2.")
        return

    try:
        pool = WorkerPool()
    except Exception as e:
        print(f"Erro ao conectar aos workers: {e}")
        return

    matrix = generate_invertible_matrix(MATRIX_SIZE)
    
    print("\nIniciando análise de desempenho...")

    # --- Cálculo Serial ---
    print("\n--- Medindo tempo Serial (NumPy Local) ---")
    start_time_serial_det = time.perf_counter()
    sign_serial, logdet_serial = np.linalg.slogdet(matrix)
    time_serial_det = time.perf_counter() - start_time_serial_det

    start_time_serial_inv = time.perf_counter()
    inv_serial = np.linalg.inv(matrix)
    time_serial_inv = time.perf_counter() - start_time_serial_inv
    print(f"Log-Determinante Serial: {time_serial_det:.6f}s | Inversa Serial: {time_serial_inv:.6f}s")

    # --- Cálculo Paralelo ---
    print("\n--- Medindo tempo Paralelo (Pyro5 Distribuído) ---")
    worker_det = pool.get_worker()
    start_time_parallel_det = time.perf_counter()
    (sign_parallel, logdet_parallel) = worker_det.log_determinant(matrix)
    time_parallel_det = time.perf_counter() - start_time_parallel_det

    worker_inv = pool.get_worker()
    start_time_parallel_inv = time.perf_counter()
    inv_parallel = worker_inv.invert(matrix)
    time_parallel_inv = time.perf_counter() - start_time_parallel_inv
    print(f"Log-Determinante Paralelo: {time_parallel_det:.6f}s | Inversa Paralela: {time_parallel_inv:.6f}s")

    # --- Validação e Salvamento ---
    np.savetxt(ORIGINAL_MATRIX_FILE, matrix, fmt='%.4f')
    np.savetxt(INVERSE_MATRIX_FILE, inv_parallel, fmt='%.4f')
    valid = np.allclose(inv_serial, inv_parallel)
    print("\nValidação da Inversa:", "OK" if valid else "FALHOU")

    # --- Geração do Relatório ---
    speedup_det = time_serial_det / time_parallel_det if time_parallel_det > 0 else 0
    speedup_inv = time_serial_inv / time_parallel_inv if time_parallel_inv > 0 else 0

    det_str_serial = format_determinant(sign_serial, logdet_serial)
    det_str_parallel = format_determinant(sign_parallel, logdet_parallel)

    report = f"""
============================================================
    ANÁLISE DE DESEMPENHO - MATRIZES DISTRIBUÍDAS
============================================================
Data: {time.strftime('%Y-%m-%d %H:%M:%S')}

MATRIZ:
  - Tamanho: {MATRIX_SIZE}x{MATRIX_SIZE}
  - Workers: {pool.count()}

TEMPOS (Log-Determinante):
  - Serial:   {time_serial_det:.6f}s
  - Paralelo: {time_parallel_det:.6f}s
  - Speedup:  {speedup_det:.2f}x

TEMPOS (Inversa):
  - Serial:   {time_serial_inv:.6f}s
  - Paralelo: {time_parallel_inv:.6f}s
  - Speedup:  {speedup_inv:.2f}x

RESULTADOS:
  - Log-Determinante Serial:   (sinal: {sign_serial}, log: {logdet_serial:.4f})
  - Log-Determinante Paralelo: (sinal: {sign_parallel}, log: {logdet_parallel:.4f})
  - Determinante (Aprox.):     {det_str_parallel}
  - Validação da Inversa:      {"OK" if valid else "FALHOU"}

Arquivos: '{ORIGINAL_MATRIX_FILE}', '{INVERSE_MATRIX_FILE}'
============================================================
"""
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nRelatório gerado com sucesso em '{OUTPUT_FILENAME}'.")
    print(report)

if __name__ == "__main__":
    main()
