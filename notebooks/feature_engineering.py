# -*- coding: utf-8 -*-
"""
Engenharia de features para modelagem
"""

import numpy as np
import pandas as pd
from scipy import stats
from .constants import PRIMES, FIBONACCI_NUMS, DIVISORS
from .utils import (get_quadrante, get_zona, get_linha_coluna_megasena,
                    get_par_impar_pattern, calculate_jump_features, calculate_concentration)


def calculate_cycle_features(binary_series, current_idx):
    """Calcula features avançadas de ciclos"""
    occurrences = binary_series[:current_idx]
    if occurrences.sum() == 0:
        return {
            'gap_atual': current_idx,
            'gap_medio': current_idx,
            'gap_std': 0,
            'gap_min': current_idx,
            'gap_max': current_idx,
            'ciclo_regular': 0,
            'prob_ciclo': 0,
            'tendencia_ciclo': 0,
            'aceleracao_ciclo': 0
        }

    indices = occurrences[occurrences == 1].index.tolist()

    gaps = []
    for i in range(len(indices) - 1):
        gaps.append(indices[i+1] - indices[i])

    if not gaps:
        gaps = [current_idx - indices[0]]

    gap_atual = current_idx - indices[-1] if indices else current_idx
    gap_medio = np.mean(gaps)
    gap_std = np.std(gaps) if len(gaps) > 1 else 0

    ciclo_regular = 1 / (1 + gap_std) if gap_std > 0 else 1
    prob_ciclo = 1 - (gap_atual / (gap_medio + gap_std + 1))
    prob_ciclo = max(0, min(1, prob_ciclo))

    tendencia_ciclo = 0
    if len(gaps) >= 3:
        ultimos_3 = gaps[-3:]
        tendencia_ciclo = (ultimos_3[-1] - ultimos_3[0]) / 3

    aceleracao_ciclo = 0
    if len(gaps) >= 4:
        ultimos_4 = gaps[-4:]
        aceleracao_ciclo = (ultimos_4[-1] - 2*ultimos_4[-2] + ultimos_4[-3])

    return {
        'gap_atual': gap_atual,
        'gap_medio': gap_medio,
        'gap_std': gap_std,
        'gap_min': min(gaps),
        'gap_max': max(gaps),
        'ciclo_regular': ciclo_regular,
        'prob_ciclo': prob_ciclo,
        'tendencia_ciclo': tendencia_ciclo,
        'aceleracao_ciclo': aceleracao_ciclo
    }


def calculate_momentum(binary_series, windows=[5, 10, 20]):
    """Calcula momentum (tendência) de aparições"""
    momentum = {}
    for w in windows:
        recent = binary_series.tail(w).mean()
        overall = binary_series.mean()
        momentum[f'momentum_{w}'] = recent - overall
    return momentum


def calculate_behavioral_score(binary_series, current_idx, window=30):
    """
    Calcula score comportamental do número
    """
    if current_idx < window:
        window = current_idx

    recent = binary_series[current_idx-window:current_idx]

    volatilidade = recent.std()

    indices_aparicoes = recent[recent == 1].index.tolist()
    if len(indices_aparicoes) > 1:
        intervalos = np.diff(indices_aparicoes)
        consistencia = 1 - (np.std(intervalos) / (np.mean(intervalos) + 1))
    else:
        consistencia = 0

    if current_idx >= 20:
        freq_10 = binary_series[current_idx-10:current_idx].mean()
        freq_20 = binary_series[current_idx-20:current_idx-10].mean()
        tendencia = freq_10 - freq_20
    else:
        tendencia = 0

    return {
        'volatilidade': volatilidade,
        'consistencia': consistencia,
        'tendencia_recente': tendencia
    }


class FeatureEngineer:
    """Classe para engenharia de features"""
    
    def __init__(self, df, df_balls):
        self.df = df
        self.df_balls = df_balls
        
    def create_basic_features(self):
        """Cria features básicas por jogo"""
        print("\n🔧 Calculando features básicas...")
        
        self.df['soma'] = self.df_balls.sum(axis=1)
        self.df['media'] = self.df_balls.mean(axis=1)
        self.df['mediana'] = self.df_balls.median(axis=1)
        self.df['std'] = self.df_balls.std(axis=1)
        self.df['amplitude'] = self.df_balls.max(axis=1) - self.df_balls.min(axis=1)
        self.df['q1'] = self.df_balls.quantile(0.25, axis=1)
        self.df['q3'] = self.df_balls.quantile(0.75, axis=1)
        self.df['iqr'] = self.df['q3'] - self.df['q1']
        
        return self
    
    def create_distribution_features(self):
        """Cria features de distribuição"""
        self.df['pares'] = self.df_balls.apply(lambda x: sum(n % 2 == 0 for n in x), axis=1)
        self.df['impares'] = 6 - self.df['pares']
        self.df['primos'] = self.df_balls.apply(lambda x: sum(n in PRIMES for n in x), axis=1)
        self.df['fibonacci'] = self.df_balls.apply(lambda x: sum(n in FIBONACCI_NUMS for n in x), axis=1)
        
        for divisor in DIVISORS:
            self.df[f'mult_{divisor}'] = self.df_balls.apply(
                lambda x: sum(n % divisor == 0 for n in x), axis=1
            )
        
        return self
    
    def create_position_features(self):
        """Cria features de posição no volante"""
        # Quadrantes
        for q in range(1, 5):
            self.df[f'quadrante_{q}'] = self.df_balls.apply(
                lambda x: sum(get_quadrante(n) == q for n in x), axis=1
            )
        
        # Zonas
        for zona in ['baixa', 'media', 'alta']:
            self.df[f'zona_{zona}'] = self.df_balls.apply(
                lambda x: sum(get_zona(n) == zona for n in x), axis=1
            )
        
        # Linhas
        for linha in range(1, 7):
            self.df[f'linha_{linha}'] = self.df_balls.apply(
                lambda x: sum(get_linha_coluna_megasena(n)[0] == linha for n in x), axis=1
            )
        
        # Colunas
        for coluna in range(1, 11):
            self.df[f'coluna_{coluna}'] = self.df_balls.apply(
                lambda x: sum(get_linha_coluna_megasena(n)[1] == coluna for n in x), axis=1
            )
        
        return self
    
    def create_pattern_features(self):
        """Cria features de padrões"""
        # Saltos
        saltos_info = self.df_balls.apply(calculate_jump_features, axis=1)
        for key in saltos_info[0].keys():
            self.df[key] = [info[key] for info in saltos_info]
        
        # Sequências
        self.df['sequencias'] = self.df_balls.apply(
            lambda x: sum(1 for i in range(len(sorted(x))-1) if sorted(x)[i+1] - sorted(x)[i] == 1),
            axis=1
        )
        
        # Padrão par-ímpar
        self.df['padrao_par_impar'] = self.df_balls.apply(get_par_impar_pattern, axis=1)
        
        # Repetições entre concursos
        repeticoes = []
        for i in range(1, len(self.df_balls)):
            atual = set(self.df_balls.iloc[i])
            anterior = set(self.df_balls.iloc[i-1])
            repeticoes.append(len(atual & anterior))
        self.df['repeticoes'] = [0] + repeticoes
        
        # Distribuição espacial
        self.df['dist_espacial'] = self.df_balls.apply(
            lambda x: np.mean([sorted(x)[i+1] - sorted(x)[i] for i in range(5)]),
            axis=1
        )
        
        # Assimetria e curtose
        self.df['assimetria'] = self.df_balls.apply(lambda x: stats.skew(x), axis=1)
        self.df['curtose'] = self.df_balls.apply(lambda x: stats.kurtosis(x), axis=1)
        
        # Concentração
        self.df['concentracao'] = self.df_balls.apply(calculate_concentration, axis=1)
        
        return self
    
    def create_all_features(self):
        """Cria todas as features"""
        self.create_basic_features()
        self.create_distribution_features()
        self.create_position_features()
        self.create_pattern_features()
        
        print("✅ Features calculadas com sucesso!")
        return self.df