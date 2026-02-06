# -*- coding: utf-8 -*-
"""
Previsão para próximos concursos
"""

import numpy as np
from collections import Counter
from itertools import combinations
from .feature_engineering import (calculate_cycle_features, calculate_momentum,
                                  calculate_behavioral_score)
from .utils import get_quadrante, get_zona, get_linha_coluna_megasena
from .constants import PRIMES, FIBONACCI_NUMS


class Predictor:
    """Classe para fazer previsões"""
    
    def __init__(self, model, scaler, binary_matrix, correlation_matrix,
                 pareto_class, trincas_por_numero, trincas_freq,
                 pares_por_numero, pares_freq, df):
        self.model = model
        self.scaler = scaler
        self.binary_matrix = binary_matrix
        self.correlation_matrix = correlation_matrix
        self.pareto_class = pareto_class
        self.trincas_por_numero = trincas_por_numero
        self.trincas_freq = trincas_freq
        self.pares_por_numero = pares_por_numero
        self.pares_freq = pares_freq
        self.df = df
        
    def predict_next_game_top10(self):
        """Prevê top 10 números com análise multi-critério"""
        next_features = []
        current_idx = len(self.df)

        for num in range(1, 61):
            # Todas as features (mesmo código do prepare_dataset)
            cycle_features = calculate_cycle_features(self.binary_matrix[num], current_idx)

            freq_total = self.binary_matrix[num].mean()
            freq_recent_5 = self.binary_matrix[num].tail(5).mean()
            freq_recent_10 = self.binary_matrix[num].tail(10).mean()
            freq_recent_20 = self.binary_matrix[num].tail(20).mean()

            momentum_features = calculate_momentum(self.binary_matrix[num], [5, 10, 20])
            behavioral = calculate_behavioral_score(self.binary_matrix[num], current_idx, window=30)

            is_par = 1 if num % 2 == 0 else 0
            is_prime_num = 1 if num in PRIMES else 0
            is_fib = 1 if num in FIBONACCI_NUMS else 0
            quadrante = get_quadrante(num)
            linha, coluna = get_linha_coluna_megasena(num)
            zona_num = {'baixa': 1, 'media': 2, 'alta': 3}[get_zona(num)]

            mult_features = [1 if num % d == 0 else 0 for d in [3, 5, 6, 7, 9]]

            avg_corr = self.correlation_matrix.iloc[num-1, :].mean()
            max_corr = self.correlation_matrix.iloc[num-1, :].max()

            recent_avg = self.binary_matrix[num].tail(30).mean()
            overall_avg = self.binary_matrix[num].mean()
            hot_cold_score = recent_avg - overall_avg

            pareto_score = {'A': 3, 'B': 2, 'C': 1}[self.pareto_class[num]]

            if num in self.trincas_por_numero and len(self.trincas_por_numero[num]) > 0:
                top_trincas_num = self.trincas_por_numero[num][:5]
                trinca_score = sum(freq for _, freq in top_trincas_num) / len(top_trincas_num)
                trinca_max = top_trincas_num[0][1]
            else:
                trinca_score = 0
                trinca_max = 0

            if num in self.pares_por_numero and len(self.pares_por_numero[num]) > 0:
                top_pares_num = self.pares_por_numero[num][:5]
                par_score = sum(freq for _, freq in top_pares_num) / len(top_pares_num)
                par_max = top_pares_num[0][1]
            else:
                par_score = 0
                par_max = 0

            atraso_norm = cycle_features['gap_atual'] / (cycle_features['gap_medio'] + 1)

            linha_freq = self.binary_matrix[[n for n in range(1, 61) 
                                             if get_linha_coluna_megasena(n)[0] == linha]].sum().sum()
            coluna_freq = self.binary_matrix[[n for n in range(1, 61) 
                                              if get_linha_coluna_megasena(n)[1] == coluna]].sum().sum()
            linha_score = linha_freq / (current_idx * 10)
            coluna_score = coluna_freq / (current_idx * 6)

            next_features.extend([
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

        next_features = np.array([next_features])
        next_features_scaled = self.scaler.transform(next_features)

        # Obter probabilidades
        try:
            probabilities = []
            for estimator in self.model.estimators_:
                if hasattr(estimator, 'predict_proba'):
                    prob = estimator.predict_proba(next_features_scaled)[0]
                    probabilities.append(prob[1] if len(prob) > 1 else prob[0])
                else:
                    probabilities.append(estimator.predict(next_features_scaled)[0])
            probabilities = np.array(probabilities)
        except:
            prediction = self.model.predict(next_features_scaled)[0]
            probabilities = prediction.astype(float)

        # Criar score combinado
        scores = {}
        for num in range(1, 61):
            idx = num - 1

            ml_score = probabilities[idx]

            cycle_info = calculate_cycle_features(self.binary_matrix[num], current_idx)
            cycle_score = cycle_info['prob_ciclo'] * 0.3

            if num in self.trincas_por_numero and len(self.trincas_por_numero[num]) > 0:
                trinca_strength = self.trincas_por_numero[num][0][1] / max(1, self.trincas_freq.most_common(1)[0][1])
            else:
                trinca_strength = 0

            if num in self.pares_por_numero and len(self.pares_por_numero[num]) > 0:
                par_strength = self.pares_por_numero[num][0][1] / max(1, self.pares_freq.most_common(1)[0][1])
            else:
                par_strength = 0

            pareto_bonus = {'A': 0.2, 'B': 0.1, 'C': 0}[self.pareto_class[num]]

            combined_score = (
                ml_score * 0.5 +
                cycle_score * 0.2 +
                trinca_strength * 0.15 +
                par_strength * 0.1 +
                pareto_bonus * 0.05
            )

            scores[num] = combined_score

        # Selecionar top 10
        top_10_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        top_10_numbers = sorted([num for num, _ in top_10_sorted])
        top_10_scores = [scores[num] for num in top_10_numbers]

        return top_10_numbers, top_10_scores, probabilities, scores
    
    def analyze_prediction(self, predicted_top10, all_scores, df_balls):
        """Análise detalhada da previsão"""
        print(f"\n" + "="*60)
        print("ANÁLISE DETALHADA DA PREVISÃO")
        print("="*60)

        pred_pares = sum(1 for n in predicted_top10 if n % 2 == 0)
        pred_impares = 10 - pred_pares
        pred_primos = sum(1 for n in predicted_top10 if n in PRIMES)
        pred_mult_3 = sum(1 for n in predicted_top10 if n % 3 == 0)
        pred_mult_6 = sum(1 for n in predicted_top10 if n % 6 == 0)
        pred_mult_9 = sum(1 for n in predicted_top10 if n % 9 == 0)

        print(f"\n📊 Composição:")
        print(f"   Pares: {pred_pares} | Ímpares: {pred_impares}")
        print(f"   Primos: {pred_primos}")
        print(f"   Múltiplos de 3: {pred_mult_3} | de 6: {pred_mult_6} | de 9: {pred_mult_9}")
        print(f"   Soma: {sum(predicted_top10)}")
        print(f"   Média: {np.mean(predicted_top10):.1f}")

        # Distribuição por quadrantes
        print(f"\n📍 Distribuição por Quadrantes:")
        for q in range(1, 5):
            nums_q = [n for n in predicted_top10 if get_quadrante(n) == q]
            faixa = f"{(q-1)*15+1}-{q*15}"
            print(f"   Q{q} ({faixa}): {len(nums_q)} números {nums_q if nums_q else '-'}")

        # Análise de atrasos
        gaps_atuais = {}
        for num in range(1, 61):
            if self.binary_matrix[num].any():
                last_idx = self.binary_matrix[num][::-1].idxmax()
                gaps_atuais[num] = len(self.binary_matrix) - last_idx
            else:
                gaps_atuais[num] = len(self.binary_matrix)

        nums_atrasados = sorted(gaps_atuais.items(), key=lambda x: x[1], reverse=True)[:20]

        print(f"\n⏰ Top 20 números mais atrasados:")
        for i, (num, gap) in enumerate(nums_atrasados, 1):
            marcador = "⭐" if num in predicted_top10 else "  "
            pareto_mark = f"[{self.pareto_class[num]}]"
            print(f"   {i:2d}. {marcador} {pareto_mark} Dezena {num:2d}: {gap:3d} concursos")

        return gaps_atuais