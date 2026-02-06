# -*- coding: utf-8 -*-
"""
Análises estatísticas básicas
"""

import pandas as pd
import numpy as np
from scipy import stats
from .constants import SOMA_BINS, SOMA_LABELS


class StatisticalAnalyzer:
    """Classe para análises estatísticas dos dados"""
    
    def __init__(self, df_balls, binary_matrix):
        self.df_balls = df_balls
        self.binary_matrix = binary_matrix
        self.freq_abs = None
        self.freq_rel = None
        
    def calculate_frequencies(self):
        """Calcula frequências absolutas e relativas"""
        self.freq_abs = self.binary_matrix.sum()
        self.freq_rel = self.freq_abs / len(self.binary_matrix)
        total_aparicoes = self.freq_abs.sum()
        
        print("\n" + "="*60)
        print("ESTATÍSTICAS BÁSICAS")
        print("="*60)
        print(f"Total de aparições: {total_aparicoes}")
        print(f"Média por dezena: {self.freq_abs.mean():.2f}")
        print(f"Desvio padrão: {self.freq_abs.std():.2f}")
        print(f"Dezena mais frequente: {self.freq_abs.idxmax()} ({self.freq_abs.max()} vezes)")
        print(f"Dezena menos frequente: {self.freq_abs.idxmin()} ({self.freq_abs.min()} vezes)")
        
        return self.freq_abs, self.freq_rel
    
    def pareto_analysis(self):
        """Análise de Pareto (80/20)"""
        freq_sorted = self.freq_abs.sort_values(ascending=False)
        freq_cumsum = freq_sorted.cumsum()
        freq_cumsum_pct = (freq_cumsum / freq_cumsum.max()) * 100
        
        pareto_80_count = (freq_cumsum_pct <= 80).sum()
        pareto_80_nums = freq_cumsum_pct[freq_cumsum_pct <= 80].index.tolist()
        
        print(f"\n📊 Análise de Pareto (Princípio 80/20)...")
        print(f"🎯 {pareto_80_count} números representam 80% das aparições:")
        print(f"   {pareto_80_nums}")
        
        # Classificação Pareto
        pareto_class = {}
        for num in range(1, 61):
            if num in pareto_80_nums[:int(pareto_80_count * 0.5)]:
                pareto_class[num] = 'A'
            elif num in pareto_80_nums:
                pareto_class[num] = 'B'
            else:
                pareto_class[num] = 'C'
        
        return pareto_class
    
    def sum_analysis(self, df):
        """Análise de somas dos jogos"""
        somas_historico = self.df_balls.sum(axis=1)
        soma_media = somas_historico.mean()
        soma_std = somas_historico.std()
        soma_min = somas_historico.min()
        soma_max = somas_historico.max()
        
        print(f"\n➕ Analisando padrões de somas...")
        print(f"Soma média: {soma_media:.1f}")
        print(f"Desvio padrão: {soma_std:.1f}")
        print(f"Intervalo: [{soma_min}, {soma_max}]")
        print(f"Intervalo 1σ: [{soma_media-soma_std:.1f}, {soma_media+soma_std:.1f}]")
        print(f"Intervalo 2σ: [{soma_media-2*soma_std:.1f}, {soma_media+2*soma_std:.1f}]")
        
        df['faixa_soma'] = pd.cut(somas_historico, bins=SOMA_BINS, labels=SOMA_LABELS)
        print("\nDistribuição de somas:")
        print(df['faixa_soma'].value_counts().sort_index())
        
        return soma_media, soma_std