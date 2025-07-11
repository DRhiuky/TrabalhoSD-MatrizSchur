import numpy as np
import Pyro5.api
import Pyro5.server
import threading
import queue
import random
import sys
import os

# --- Configuração do Serializador Otimizado ---
Pyro5.config.SERIALIZER = "msgpack"

# --- Utilitários de cache ---
def matrix_hash(matrix):
    """Gera um hash único para uma matriz baseada no conteúdo de seus bytes."""
    return hash(matrix.tobytes())

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

# --- Configuração ---
BASE_CASE_SIZE = 64

def get_random_worker():
    """Obtém um proxy para um worker aleatório registrado no Name Server."""
    ns = Pyro5.api.locate_ns()
    workers = [name for name in ns.list(prefix="matrix.calculator.").keys()]
    if not workers:
        raise RuntimeError("Nenhum worker disponível registrado no Name Server.")
    chosen = random.choice(workers)
    return Pyro5.api.Proxy(f"PYRONAME:{chosen}")

@Pyro5.api.expose
@Pyro5.api.behavior(instance_mode="single") # "single" é o modo correto para uma instância por processo.
class ParallelMatrixCalculator:
    """
    Worker que executa os cálculos. Com a instância única, o cache agora funciona corretamente.
    """
    def __init__(self):
        self.inv_cache = {}
        self.log_det_cache = {}
        print(f"Worker iniciado. PID: {os.getpid()}. Cache está vazio.")

    def multiply(self, A, B):
        print(f"  PID {os.getpid()}: Multiplicando {A.shape} x {B.shape}")
        return np.dot(A, B)

    def invert(self, M):
        print(f"PID {os.getpid()}: Recebida tarefa INVERSA para matriz {M.shape[0]}x{M.shape[0]}")
        key = matrix_hash(M)
        if key in self.inv_cache:
            print(f"  PID {os.getpid()}: CACHE HIT para INVERSA de {M.shape[0]}x{M.shape[0]}")
            return self.inv_cache[key]

        n = M.shape[0]
        if n <= BASE_CASE_SIZE:
            print(f"  PID {os.getpid()}: INVERSA - Caso base para {n}x{n}")
            result = np.linalg.inv(M)
            self.inv_cache[key] = result
            return result

        mid = n // 2
        A, B, C, D = M[:mid, :mid], M[:mid, mid:], M[mid:, :mid], M[mid:, mid:]

        proxy = get_random_worker()
        A_inv = proxy.invert(A)
        
        S = D - self.multiply(self.multiply(C, A_inv), B)
        
        proxy = get_random_worker()
        S_inv = proxy.invert(S)
        
        A_inv_B = self.multiply(A_inv, B)
        C_A_inv = self.multiply(C, A_inv)
        
        block11 = A_inv + self.multiply(self.multiply(A_inv_B, S_inv), C_A_inv)
        block12 = -self.multiply(A_inv_B, S_inv)
        block21 = -self.multiply(S_inv, C_A_inv)
        block22 = S_inv

        inv_M = np.block([[block11, block12], [block21, block22]])
        self.inv_cache[key] = inv_M
        return inv_M

    def log_determinant(self, M):
        """Calcula o sinal e o logaritmo natural do determinante."""
        print(f"PID {os.getpid()}: Recebida tarefa LOG-DETERMINANTE para matriz {M.shape[0]}x{M.shape[0]}")
        key = matrix_hash(M)
        if key in self.log_det_cache:
            print(f"  PID {os.getpid()}: CACHE HIT para LOG-DETERMINANTE de {M.shape[0]}x{M.shape[0]}")
            return self.log_det_cache[key]

        n = M.shape[0]
        if n <= BASE_CASE_SIZE:
            print(f"  PID {os.getpid()}: LOG-DETERMINANTE - Caso base para {n}x{n}")
            sign, logdet = np.linalg.slogdet(M)
            result = (sign, logdet)
            self.log_det_cache[key] = result
            return result

        mid = n // 2
        A, B, C, D = M[:mid, :mid], M[:mid, mid:], M[mid:, :mid], M[mid:, mid:]
        
        results_queue = queue.Queue()

        def get_log_det_A():
            proxy = get_random_worker()
            result = proxy.log_determinant(A)
            results_queue.put(('log_det_A', result))

        def get_inv_A():
            proxy = get_random_worker()
            result = proxy.invert(A)
            results_queue.put(('A_inv', result))

        thread_det = threading.Thread(target=get_log_det_A)
        thread_inv = threading.Thread(target=get_inv_A)
        thread_det.start()
        thread_inv.start()

        results = {}
        for _ in range(2):
            key_result, value = results_queue.get()
            results[key_result] = value

        (sign_A, log_det_A) = results['log_det_A']
        A_inv = results['A_inv']

        S = D - self.multiply(self.multiply(C, A_inv), B)

        proxy = get_random_worker()
        (sign_S, log_det_S) = proxy.log_determinant(S)

        final_sign = sign_A * sign_S
        final_log_det = log_det_A + log_det_S
        
        result = (final_sign, final_log_det)
        self.log_det_cache[key] = result
        return result

def main():
    """Função principal para iniciar o worker."""
    if len(sys.argv) < 2:
        print("Erro: Forneça um ID único para este worker.")
        print("Uso: python worker.py <id>")
        sys.exit(1)

    worker_id = sys.argv[1]
    worker_name = f"matrix.calculator.{worker_id}"

    Pyro5.config.SERVERTYPE = "thread"
    daemon = Pyro5.server.Daemon()
    ns = Pyro5.api.locate_ns()
    
    # Registra a classe com o comportamento de instância única
    uri = daemon.register(ParallelMatrixCalculator)
    ns.register(worker_name, uri)

    print(f"Worker '{worker_name}' pronto. PID: {os.getpid()}")
    daemon.requestLoop()

if __name__ == "__main__":
    main()
