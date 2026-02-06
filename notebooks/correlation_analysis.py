# -*- coding: utf-8 -*-
"""
Análise de correlações, pares e trincas
"""

from collections import Counter, defaultdict
from itertools import combinations


class CorrelationAnalyzer:
    """Classe para análise de correlações entre números"""
    
    def __init__(self, df_balls, binary_matrix):
        self.df_balls = df_balls
        self.binary_matrix = binary_matrix
        self.correlation_matrix = None
        self.pares_freq = None
        self.trincas_freq = None
        self.pares_por_numero = None
        self.trincas_por_numero = None
        
    def calculate_correlations(self):
        """Calcula matriz de correlações"""
        print("\n📊 Calculando correlações entre dezenas...")
        self.correlation_matrix = self.binary_matrix.corr()
        
        corr_pairs = []
        for i in range(1, 61):
            for j in range(i+1, 61):
                corr_pairs.append((i, j, self.correlation_matrix.loc[i, j]))
        
        corr_pairs_sorted = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
        print("\nTop 10 pares mais correlacionados:")
        for i, (n1, n2, corr) in enumerate(corr_pairs_sorted[:10], 1):
            print(f"  {i:2d}. {n1:2d}-{n2:2d}: {corr:.4f}")
        
        return self.correlation_matrix
    
    def analyze_pairs(self):
        """Analisa pares mais frequentes"""
        print("\n🔍 Analisando pares correlacionais...")
        
        self.pares_freq = Counter()
        for _, row in self.df_balls.iterrows():
            nums = sorted(row.values)
            for par in combinations(nums, 2):
                self.pares_freq[par] += 1
        
        top_pares = self.pares_freq.most_common(20)
        print("\nTop 10 pares mais frequentes:")
        for i, (par, freq) in enumerate(top_pares[:10], 1):
            print(f"  {i:2d}. {par}: {freq} vezes")
        
        # Dicionário de pares por número
        self.pares_por_numero = defaultdict(list)
        for par, freq in self.pares_freq.items():
            for num in par:
                self.pares_por_numero[num].append((par, freq))
        
        for num in self.pares_por_numero:
            self.pares_por_numero[num] = sorted(
                self.pares_por_numero[num], 
                key=lambda x: x[1], 
                reverse=True
            )
        
        return self.pares_freq, self.pares_por_numero
    
    def analyze_triples(self):
        """Analisa trincas mais frequentes"""
        print("\n🔍 Analisando trincas correlacionais...")
        
        self.trincas_freq = Counter()
        for _, row in self.df_balls.iterrows():
            nums = sorted(row.values)
            for trinca in combinations(nums, 3):
                self.trincas_freq[trinca] += 1
        
        top_trincas = self.trincas_freq.most_common(20)
        print("\nTop 10 trincas mais frequentes:")
        for i, (trinca, freq) in enumerate(top_trincas[:10], 1):
            print(f"  {i:2d}. {trinca}: {freq} vezes")
        
        # Dicionário de trincas por número
        self.trincas_por_numero = defaultdict(list)
        for trinca, freq in self.trincas_freq.items():
            for num in trinca:
                self.trincas_por_numero[num].append((trinca, freq))
        
        for num in self.trincas_por_numero:
            self.trincas_por_numero[num] = sorted(
                self.trincas_por_numero[num], 
                key=lambda x: x[1], 
                reverse=True
            )
        
        return self.trincas_freq, self.trincas_por_numero