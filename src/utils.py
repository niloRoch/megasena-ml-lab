# -*- coding: utf-8 -*-
"""
Funções utilitárias gerais
"""

import numpy as np
from .constants import VOLANTE_ROWS, VOLANTE_COLS, PRIMES, FIBONACCI_NUMS


def get_quadrante(num):
    """Divide volante 60 números em 4 quadrantes"""
    if num <= 15:
        return 1
    elif num <= 30:
        return 2
    elif num <= 45:
        return 3
    else:
        return 4


def get_linha_coluna_megasena(num):
    """Posição no volante Mega-Sena (6x10 layout)"""
    linha = (num - 1) // VOLANTE_COLS + 1
    coluna = (num - 1) % VOLANTE_COLS + 1
    return linha, coluna


def get_zona(num):
    """Divide em 3 zonas: Baixa (1-20), Média (21-40), Alta (41-60)"""
    if num <= 20:
        return 'baixa'
    elif num <= 40:
        return 'media'
    else:
        return 'alta'


def get_par_impar_pattern(row):
    """Retorna padrão par-ímpar de um jogo"""
    sorted_nums = sorted(row)
    pattern = ''.join(['P' if n % 2 == 0 else 'I' for n in sorted_nums])
    return pattern


def calculate_jump_features(row):
    """Calcula features relacionadas aos saltos entre números"""
    sorted_nums = sorted(row)
    jumps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
    return {
        'salto_min': min(jumps),
        'salto_max': max(jumps),
        'salto_medio': np.mean(jumps),
        'salto_std': np.std(jumps),
        'saltos_1': sum(1 for j in jumps if j == 1),
        'saltos_2_5': sum(1 for j in jumps if 2 <= j <= 5),
        'saltos_grandes': sum(1 for j in jumps if j > 10)
    }


def calculate_concentration(row):
    """Calcula concentração/dispersão dos números"""
    sorted_nums = sorted(row)
    n = len(sorted_nums)
    cumsum = np.cumsum(sorted_nums)
    return (2 * np.sum((i+1) * val for i, val in enumerate(sorted_nums))) / (n * cumsum[-1]) - (n+1) / n