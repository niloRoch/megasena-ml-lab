# -*- coding: utf-8 -*-
"""
Avaliação de modelos
"""

import numpy as np


def evaluate_model(y_true, y_pred, model_name, df_len, test_size):
    """Avalia modelo com métricas específicas para Mega-Sena"""
    print(f"\n{'='*60}")
    print(f"AVALIAÇÃO - {model_name.upper()}")
    print(f"{'='*60}")

    acertos_por_jogo = []
    for i in range(len(y_true)):
        true_nums = set(np.where(y_true[i] == 1)[0] + 1)
        pred_nums = set(np.where(y_pred[i] == 1)[0] + 1)
        acertos = len(true_nums & pred_nums)
        acertos_por_jogo.append(acertos)

        concurso_num = df_len - test_size + i
        print(f"  Concurso {concurso_num}: {acertos}/6 acertos | "
              f"Previstos: {len(pred_nums)}")

    media_acertos = np.mean(acertos_por_jogo)
    print(f"\n📊 Estatísticas:")
    print(f"   Média de acertos: {media_acertos:.2f}/6")
    print(f"   Mínimo: {min(acertos_por_jogo)}/6")
    print(f"   Máximo: {max(acertos_por_jogo)}/6")
    print(f"   Desvio padrão: {np.std(acertos_por_jogo):.2f}")

    return {
        'acertos_medio': media_acertos,
        'acertos_lista': acertos_por_jogo
    }


def select_best_model(models_metrics, models):
    """Seleciona o melhor modelo baseado nas métricas"""
    best_score = max([m['acertos_medio'] for m in models_metrics.values()])
    
    for name, metrics in models_metrics.items():
        if metrics['acertos_medio'] == best_score:
            best_model_name = name
            break
    
    print(f"\n🏆 Melhor modelo: {best_model_name.upper()}")
    
    return models[best_model_name], best_model_name