# -*- coding: utf-8 -*-
"""
Constantes e configurações do projeto
"""

import numpy as np

# Configurações de visualização
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_SIZE = (14, 10)
COLOR_PALETTE = "husl"

# Parâmetros da Mega-Sena
MIN_NUMBER = 1
MAX_NUMBER = 60
NUMBERS_PER_DRAW = 6
VOLANTE_ROWS = 6
VOLANTE_COLS = 10

# Números especiais
def is_prime(n):
    """Verifica se número é primo"""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def is_fibonacci(n):
    """Verifica se número é Fibonacci"""
    def is_perfect_square(x):
        s = int(np.sqrt(x))
        return s*s == x
    return is_perfect_square(5*n*n + 4) or is_perfect_square(5*n*n - 4)

# Listas de números especiais
PRIMES = [n for n in range(MIN_NUMBER, MAX_NUMBER + 1) if is_prime(n)]
FIBONACCI_NUMS = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
EVEN_NUMS = [n for n in range(2, MAX_NUMBER + 1, 2)]
ODD_NUMS = [n for n in range(1, MAX_NUMBER + 1, 2)]
MULT_3 = [n for n in range(3, MAX_NUMBER + 1, 3)]
MULT_6 = [n for n in range(6, MAX_NUMBER + 1, 6)]
MULT_9 = [n for n in range(9, MAX_NUMBER + 1, 9)]

# Divisores para análise de múltiplos
DIVISORS = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Configurações de modelagem
MIN_HISTORY = 20
TEST_SIZE = 15
RANDOM_STATE = 42

# Parâmetros de modelos
RF_PARAMS = {
    'n_estimators': 600,
    'max_depth': 35,
    'min_samples_split': 4,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'class_weight': 'balanced'
}

GB_PARAMS = {
    'n_estimators': 250,
    'learning_rate': 0.08,
    'max_depth': 12,
    'min_samples_split': 4,
    'subsample': 0.8,
    'random_state': RANDOM_STATE
}

LR_PARAMS = {
    'max_iter': 8000,
    'C': 0.15,
    'solver': 'saga',
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced',
    'n_jobs': -1
}

# Bins para análise de somas
SOMA_BINS = [0, 120, 150, 180, 210, 240, 300]
SOMA_LABELS = ['<120', '120-150', '150-180', '180-210', '210-240', '>240']