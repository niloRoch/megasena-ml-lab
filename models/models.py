# -*- coding: utf-8 -*-
"""
Treinamento e gerenciamento de modelos
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import RobustScaler
from .constants import RF_PARAMS, GB_PARAMS, LR_PARAMS, MIN_HISTORY, TEST_SIZE
from .feature_engineering import (calculate_cycle_features, calculate_momentum,
                                  calculate_behavioral_score)
from .utils import get_quadrante, get_zona, get_linha_coluna_megasena
from .constants import PRIMES, FIBONACCI_NUMS


class ModelPipeline:
    """Pipeline completo de modelagem"""
    
    def __init__(self, binary_matrix, correlation_matrix, pareto_class,
                 trincas_por_numero, trincas_freq, pares_por_numero, pares_freq, df):
        self.binary_matrix = binary_matrix
        self.correlation_matrix = correlation_matrix
        self.pareto_class = pareto_class
        self.trincas_por_numero = trincas_por_numero
        self.trincas_freq = trincas_freq
        self.pares_por_numero = pares_por_numero
        self.pares_freq = pares_freq
        self.df = df
        
        self.X = None
        self.y = None
        self.scaler = None
        self.models = {}
        
    def prepare_dataset(self):
        """Prepara dataset para modelagem"""
        print("\n🤖 Preparando dados para Machine Learning...")
        
        X = []
        y = []

        for i in range(MIN_HISTORY, len(self.df)):
            features_concurso = []

            for num in range(1, 61):
                # Features de ciclo
                cycle_features = calculate_cycle_features(self.binary_matrix[num], i)

                # Features de frequência
                freq_total = self.binary_matrix[num][:i].mean()
                freq_recent_5 = self.binary_matrix[num][i-5:i].mean()
                freq_recent_10 = self.binary_matrix[num][i-10:i].mean()
                freq_recent_20 = self.binary_matrix[num][i-20:i].mean()

                # Momentum
                momentum_features = calculate_momentum(self.binary_matrix[num][:i], [5, 10, 20])

                # Comportamento
                behavioral = calculate_behavioral_score(self.binary_matrix[num], i, window=30)

                # Características estáticas
                is_par = 1 if num % 2 == 0 else 0
                is_prime_num = 1 if num in PRIMES else 0
                is_fib = 1 if num in FIBONACCI_NUMS else 0
                quadrante = get_quadrante(num)
                linha, coluna = get_linha_coluna_megasena(num)
                zona_num = {'baixa': 1, 'media': 2, 'alta': 3}[get_zona(num)]

                # Múltiplos
                mult_features = [1 if num % d == 0 else 0 for d in [3, 5, 6, 7, 9]]

                # Correlação
                avg_corr = self.correlation_matrix.iloc[num-1, :].mean()
                max_corr = self.correlation_matrix.iloc[num-1, :].max()

                # Hot/Cold
                recent_avg = self.binary_matrix[num][max(0,i-30):i].mean()
                overall_avg = self.binary_matrix[num][:i].mean()
                hot_cold_score = recent_avg - overall_avg

                # Pareto
                pareto_score = {'A': 3, 'B': 2, 'C': 1}[self.pareto_class[num]]

                # Trincas
                if num in self.trincas_por_numero and len(self.trincas_por_numero[num]) > 0:
                    top_trincas_num = self.trincas_por_numero[num][:5]
                    trinca_score = sum(freq for _, freq in top_trincas_num) / len(top_trincas_num)
                    trinca_max = top_trincas_num[0][1]
                else:
                    trinca_score = 0
                    trinca_max = 0

                # Pares
                if num in self.pares_por_numero and len(self.pares_por_numero[num]) > 0:
                    top_pares_num = self.pares_por_numero[num][:5]
                    par_score = sum(freq for _, freq in top_pares_num) / len(top_pares_num)
                    par_max = top_pares_num[0][1]
                else:
                    par_score = 0
                    par_max = 0

                # Atraso normalizado
                atraso_norm = cycle_features['gap_atual'] / (cycle_features['gap_medio'] + 1)

                # Tendência de linha/coluna
                linha_freq = self.binary_matrix[[n for n in range(1, 61) 
                                                 if get_linha_coluna_megasena(n)[0] == linha]][:i].sum().sum()
                coluna_freq = self.binary_matrix[[n for n in range(1, 61) 
                                                  if get_linha_coluna_megasena(n)[1] == coluna]][:i].sum().sum()
                linha_score = linha_freq / (i * 10)
                coluna_score = coluna_freq / (i * 6)

                # Adicionar features
                features_concurso.extend([
                    freq_total, freq_recent_5, freq_recent_10, freq_recent_20,
                    cycle_features['gap_atual'], cycle_features['gap_medio'],
                    cycle_features['gap_std'], cycle_features['ciclo_regular'],
                    cycle_features['prob_ciclo'], cycle_features['tendencia_ciclo'],
                    cycle_features['aceleracao_ciclo'], atraso_norm,
                    cycle_features['gap_max'] - cycle_features['gap_min'],
                    momentum_features['momentum_5'], momentum_features['momentum_10'],
                    momentum_features['momentum_20'],
                    behavioral['volatilidade'], behavioral['consistencia'],
                    behavioral['tendencia_recente'],
                    is_par, is_prime_num, is_fib, quadrante, linha, coluna, zona_num,
                    *mult_features,
                    avg_corr, max_corr,
                    hot_cold_score,
                    pareto_score,
                    trinca_score, trinca_max, par_score, par_max,
                    linha_score, coluna_score
                ])

            X.append(features_concurso)
            y.append(self.binary_matrix.iloc[i, :].values)

        self.X = np.array(X)
        self.y = np.array(y)

        num_features_per_number = len(features_concurso) // 60

        print(f"✅ Dataset preparado:")
        print(f"   Amostras: {self.X.shape[0]}")
        print(f"   Features por número: {num_features_per_number}")
        print(f"   Total de features: {self.X.shape[1]}")
        
        return self
    
    def split_data(self, test_size=TEST_SIZE):
        """Divide dados em treino e teste"""
        X_train, X_test = self.X[:-test_size], self.X[-test_size:]
        y_train, y_test = self.y[:-test_size], self.y[-test_size:]

        print(f"\n📈 Divisão dos dados:")
        print(f"   Treino: {X_train.shape[0]} concursos")
        print(f"   Teste: {X_test.shape[0]} concursos")

        # Normalização
        self.scaler = RobustScaler()
        X_train_scaled