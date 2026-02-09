# -*- coding: utf-8 -*-
"""
Modelo Mega-Sena Aprimorado - Machine Learning
Universo: 60 dezenas | Previsão: Top 10 números mais prováveis
Features expandidas: Ciclos, Quadrantes, Padrões
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.multioutput import MultiOutputClassifier
from scipy import stats
from scipy.stats import chi2_contingency
from google.colab import files
from itertools import combinations
import warnings
from collections import Counter, defaultdict, deque
warnings.filterwarnings('ignore')

# Upload do arquivo
print("Por favor, faça o upload do arquivo CSV da Mega-Sena")
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

# Configuração de visualização
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
sns.set_palette("husl")

# Carregar dados
df = pd.read_csv(file_name, sep=';')
print(f"✅ Dados carregados: {len(df)} concursos")
print(f"Colunas disponíveis: {df.columns.tolist()}")

# Identificar colunas das bolas (ajustar conforme o CSV)
ball_columns = [col for col in df.columns if 'Bola' in col or 'Dezena' in col or col.isdigit()]
if not ball_columns:
    # Tentar identificar automaticamente
    ball_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and df[col].between(1, 60).all()]

if len(ball_columns) < 6:
    print("⚠️ Atenção: Por favor, ajuste as colunas das bolas manualmente")
    print("Exemplo: ball_columns = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']")

balls = ball_columns[:6]  # Mega-Sena
df_balls = df[balls].copy()

print(f"Colunas das bolas: {balls}")
print(f"Amostra dos dados:")
print(df_balls.head())

# ==================== FUNÇÕES AUXILIARES ====================

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

def get_quadrante(num):
    """
    Divide volante 60 números em 4 quadrantes
    Q1: 1-15, Q2: 16-30, Q3: 31-45, Q4: 46-60
    """
    if num <= 15:
        return 1
    elif num <= 30:
        return 2
    elif num <= 45:
        return 3
    else:
        return 4

def get_linha_coluna_megasena(num):
    """
    Posição no volante Mega-Sena (6x10 layout)
    Linhas: 6 | Colunas: 10
    """
    linha = (num - 1) // 10 + 1
    coluna = (num - 1) % 10 + 1
    return linha, coluna

def get_zona(num):
    """
    Divide em 3 zonas: Baixa (1-20), Média (21-40), Alta (41-60)
    """
    if num <= 20:
        return 'baixa'
    elif num <= 40:
        return 'media'
    else:
        return 'alta'

def calculate_cycle_features(binary_series, current_idx):
    """
    Calcula features avançadas de ciclos
    """
    occurrences = binary_series[:current_idx]
    if occurrences.sum() == 0:
        return {
            'gap_atual': current_idx,
            'gap_medio': current_idx,
            'gap_std': 0,
            'gap_min': current_idx,
            'gap_max': current_idx,
            'ciclo_regular': 0,
            'prob_ciclo': 0
        }

    # Encontrar índices de ocorrências
    indices = occurrences[occurrences == 1].index.tolist()

    # Calcular gaps
    gaps = []
    for i in range(len(indices) - 1):
        gaps.append(indices[i+1] - indices[i])

    if not gaps:
        gaps = [current_idx - indices[0]]

    gap_atual = current_idx - indices[-1] if indices else current_idx
    gap_medio = np.mean(gaps)
    gap_std = np.std(gaps) if len(gaps) > 1 else 0

    # Regularidade do ciclo (menor desvio = mais regular)
    ciclo_regular = 1 / (1 + gap_std) if gap_std > 0 else 1

    # Probabilidade baseada no ciclo
    prob_ciclo = 1 - (gap_atual / (gap_medio + gap_std + 1))
    prob_ciclo = max(0, min(1, prob_ciclo))

    return {
        'gap_atual': gap_atual,
        'gap_medio': gap_medio,
        'gap_std': gap_std,
        'gap_min': min(gaps),
        'gap_max': max(gaps),
        'ciclo_regular': ciclo_regular,
        'prob_ciclo': prob_ciclo
    }

def calculate_momentum(binary_series, windows=[5, 10, 20]):
    """
    Calcula momentum (tendência) de aparições
    """
    momentum = {}
    for w in windows:
        recent = binary_series.tail(w).mean()
        overall = binary_series.mean()
        momentum[f'momentum_{w}'] = recent - overall
    return momentum

# ==================== CONSTANTES ====================
primes = [n for n in range(1, 61) if is_prime(n)]
fibonacci_nums = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
pares = [n for n in range(2, 61, 2)]
impares = [n for n in range(1, 61, 2)]

print(f"\n📊 Números primos: {len(primes)} números")
print(f"📊 Números Fibonacci: {fibonacci_nums}")
print(f"📊 Pares: {len(pares)} | Ímpares: {len(impares)}")

# ==================== MATRIZ BINÁRIA ====================
binary_matrix = pd.DataFrame(index=df.index, columns=range(1, 61), dtype=int)
for num in range(1, 61):
    binary_matrix[num] = df_balls.isin([num]).any(axis=1).astype(int)

print("\n✅ Matriz binária criada (60 dezenas)")

# ==================== ESTATÍSTICAS BÁSICAS ====================
freq_abs = binary_matrix.sum()
freq_rel = freq_abs / len(df)
total_aparicoes = freq_abs.sum()

print("\n" + "="*60)
print("ESTATÍSTICAS BÁSICAS")
print("="*60)
print(f"Total de aparições: {total_aparicoes}")
print(f"Média por dezena: {freq_abs.mean():.2f}")
print(f"Desvio padrão: {freq_abs.std():.2f}")
print(f"Dezena mais frequente: {freq_abs.idxmax()} ({freq_abs.max()} vezes)")
print(f"Dezena menos frequente: {freq_abs.idxmin()} ({freq_abs.min()} vezes)")

# ==================== FEATURES EXPANDIDAS ====================
print("\n🔧 Calculando features expandidas...")

# 1. ESTATÍSTICAS BÁSICAS POR JOGO
df['soma'] = df_balls.sum(axis=1)
df['media'] = df_balls.mean(axis=1)
df['mediana'] = df_balls.median(axis=1)
df['std'] = df_balls.std(axis=1)
df['amplitude'] = df_balls.max(axis=1) - df_balls.min(axis=1)
df['q1'] = df_balls.quantile(0.25, axis=1)
df['q3'] = df_balls.quantile(0.75, axis=1)
df['iqr'] = df['q3'] - df['q1']

# 2. DISTRIBUIÇÕES
df['pares'] = df_balls.apply(lambda x: sum(n % 2 == 0 for n in x), axis=1)
df['impares'] = 6 - df['pares']
df['primos'] = df_balls.apply(lambda x: sum(n in primes for n in x), axis=1)
df['fibonacci'] = df_balls.apply(lambda x: sum(n in fibonacci_nums for n in x), axis=1)

# 3. MÚLTIPLOS
for divisor in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    df[f'mult_{divisor}'] = df_balls.apply(lambda x: sum(n % divisor == 0 for n in x), axis=1)

# 4. QUADRANTES
for q in range(1, 5):
    df[f'quadrante_{q}'] = df_balls.apply(
        lambda x: sum(get_quadrante(n) == q for n in x), axis=1
    )

# 5. ZONAS
for zona in ['baixa', 'media', 'alta']:
    df[f'zona_{zona}'] = df_balls.apply(
        lambda x: sum(get_zona(n) == zona for n in x), axis=1
    )

# 6. POSIÇÕES NO VOLANTE (6x10)
for linha in range(1, 7):
    df[f'linha_{linha}'] = df_balls.apply(
        lambda x: sum(get_linha_coluna_megasena(n)[0] == linha for n in x), axis=1
    )

for coluna in range(1, 11):
    df[f'coluna_{coluna}'] = df_balls.apply(
        lambda x: sum(get_linha_coluna_megasena(n)[1] == coluna for n in x), axis=1
    )

# 7. ANÁLISE DE SALTOS
def calculate_jump_features(row):
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

saltos_info = df_balls.apply(calculate_jump_features, axis=1)
for key in saltos_info[0].keys():
    df[key] = [info[key] for info in saltos_info]

# 8. SEQUÊNCIAS
df['sequencias'] = df_balls.apply(
    lambda x: sum(1 for i in range(len(sorted(x))-1) if sorted(x)[i+1] - sorted(x)[i] == 1),
    axis=1
)

# 9. PADRÕES PAR-ÍMPAR
def get_par_impar_pattern(row):
    sorted_nums = sorted(row)
    pattern = ''.join(['P' if n % 2 == 0 else 'I' for n in sorted_nums])
    return pattern

df['padrao_par_impar'] = df_balls.apply(get_par_impar_pattern, axis=1)

# 10. REPETIÇÕES ENTRE CONCURSOS
repeticoes = []
for i in range(1, len(df_balls)):
    atual = set(df_balls.iloc[i])
    anterior = set(df_balls.iloc[i-1])
    repeticoes.append(len(atual & anterior))
df['repeticoes'] = [0] + repeticoes

# 11. DISTRIBUIÇÃO ESPACIAL (Distância média entre números)
df['dist_espacial'] = df_balls.apply(
    lambda x: np.mean([sorted(x)[i+1] - sorted(x)[i] for i in range(5)]),
    axis=1
)

# 12. ASSIMETRIA E CURTOSE
df['assimetria'] = df_balls.apply(lambda x: stats.skew(x), axis=1)
df['curtose'] = df_balls.apply(lambda x: stats.kurtosis(x), axis=1)

# 13. CONCENTRAÇÃO (índice de Gini simplificado)
def calculate_concentration(row):
    sorted_nums = sorted(row)
    n = len(sorted_nums)
    cumsum = np.cumsum(sorted_nums)
    return (2 * np.sum((i+1) * val for i, val in enumerate(sorted_nums))) / (n * cumsum[-1]) - (n+1) / n

df['concentracao'] = df_balls.apply(calculate_concentration, axis=1)

print("✅ Features calculadas com sucesso!")

# ==================== ANÁLISE DE CORRELAÇÕES ====================
print("\n📊 Calculando correlações entre dezenas...")
correlation_matrix = binary_matrix.corr()

# Pares mais correlacionados
corr_pairs = []
for i in range(1, 61):
    for j in range(i+1, 61):
        corr_pairs.append((i, j, correlation_matrix.loc[i, j]))

corr_pairs_sorted = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
print("\nTop 10 pares mais correlacionados:")
for i, (n1, n2, corr) in enumerate(corr_pairs_sorted[:10], 1):
    print(f"  {i:2d}. {n1:2d}-{n2:2d}: {corr:.4f}")

# ==================== PREPARAÇÃO PARA MODELAGEM ====================
print("\n🤖 Preparando dados para Machine Learning...")

X = []
y = []

for i in range(10, len(df)):  # Usar histórico mínimo de 10 concursos
    features_concurso = []

    for num in range(1, 61):
        # Features de ciclo
        cycle_features = calculate_cycle_features(binary_matrix[num], i)

        # Features de frequência
        freq_total = binary_matrix[num][:i].mean()
        freq_recent_5 = binary_matrix[num][i-5:i].mean()
        freq_recent_10 = binary_matrix[num][i-10:i].mean()
        freq_recent_20 = binary_matrix[num][max(0,i-20):i].mean()

        # Momentum
        momentum_features = calculate_momentum(binary_matrix[num][:i], [5, 10, 20])

        # Características estáticas
        is_par = 1 if num % 2 == 0 else 0
        is_prime_num = 1 if num in primes else 0
        is_fib = 1 if num in fibonacci_nums else 0
        quadrante = get_quadrante(num)
        linha, coluna = get_linha_coluna_megasena(num)
        zona_num = {'baixa': 1, 'media': 2, 'alta': 3}[get_zona(num)]

        # Múltiplos
        mult_features = [1 if num % d == 0 else 0 for d in [3, 5, 7]]

        # Correlação média com outros números
        avg_corr = correlation_matrix.iloc[num-1, :].mean()

        # Hot/Cold
        recent_avg = binary_matrix[num][max(0,i-30):i].mean()
        overall_avg = binary_matrix[num][:i].mean()
        hot_cold_score = recent_avg - overall_avg

        # Adicionar features
        features_concurso.extend([
            # Frequências
            freq_total, freq_recent_5, freq_recent_10, freq_recent_20,
            # Ciclos
            cycle_features['gap_atual'], cycle_features['gap_medio'],
            cycle_features['gap_std'], cycle_features['ciclo_regular'],
            cycle_features['prob_ciclo'],
            # Momentum
            momentum_features['momentum_5'], momentum_features['momentum_10'],
            momentum_features['momentum_20'],
            # Estáticas
            is_par, is_prime_num, is_fib, quadrante, linha, coluna, zona_num,
            # Múltiplos
            *mult_features,
            # Correlação e temperatura
            avg_corr, hot_cold_score
        ])

    X.append(features_concurso)
    y.append(binary_matrix.iloc[i, :].values)

X = np.array(X)
y = np.array(y)

print(f"✅ Dataset preparado:")
print(f"   Amostras: {X.shape[0]}")
print(f"   Features por número: {X.shape[1] // 60}")
print(f"   Total de features: {X.shape[1]}")

# ==================== DIVISÃO TREINO/TESTE ====================
test_size = 15  # Últimos 15 concursos para teste
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

print(f"\n📈 Divisão dos dados:")
print(f"   Treino: {X_train.shape[0]} concursos")
print(f"   Teste: {X_test.shape[0]} concursos")

# Normalização robusta
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== TREINAMENTO DOS MODELOS ====================
print("\n" + "="*60)
print("TREINAMENTO DOS MODELOS")
print("="*60)

# Modelo 1: Random Forest
print("\n🌲 Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf = MultiOutputClassifier(rf_model)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)

# Modelo 2: Gradient Boosting
print("🚀 Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
gb = MultiOutputClassifier(gb_model)
gb.fit(X_train_scaled, y_train)
gb_pred = gb.predict(X_test_scaled)

# Modelo 3: Logistic Regression
print("📊 Logistic Regression...")
lr_model = LogisticRegression(
    max_iter=5000,
    C=0.1,
    solver='saga',
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
lr = MultiOutputClassifier(lr_model)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

# ==================== AVALIAÇÃO ====================
def evaluate_model(y_true, y_pred, model_name):
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

        concurso_num = len(df) - test_size + i
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

# Avaliar todos os modelos
rf_metrics = evaluate_model(y_test, rf_pred, "Random Forest")
gb_metrics = evaluate_model(y_test, gb_pred, "Gradient Boosting")
lr_metrics = evaluate_model(y_test, lr_pred, "Logistic Regression")

# Selecionar melhor modelo
melhor_score = max(rf_metrics['acertos_medio'],
                   gb_metrics['acertos_medio'],
                   lr_metrics['acertos_medio'])

if rf_metrics['acertos_medio'] == melhor_score:
    best_model = rf
    best_name = "Random Forest"
elif gb_metrics['acertos_medio'] == melhor_score:
    best_model = gb
    best_name = "Gradient Boosting"
else:
    best_model = lr
    best_name = "Logistic Regression"

print(f"\n🏆 Melhor modelo: {best_name}")

# ==================== PREVISÃO PRÓXIMO CONCURSO ====================
print("\n" + "="*60)
print(f"PREVISÃO PARA O PRÓXIMO CONCURSO ({len(df) + 1})")
print("="*60)

def predict_next_game_top10(model, scaler):
    """Prevê top 10 números mais prováveis"""
    next_features = []
    current_idx = len(df)

    for num in range(1, 61):
        # Calcular todas as features
        cycle_features = calculate_cycle_features(binary_matrix[num], current_idx)

        freq_total = binary_matrix[num].mean()
        freq_recent_5 = binary_matrix[num].tail(5).mean()
        freq_recent_10 = binary_matrix[num].tail(10).mean()
        freq_recent_20 = binary_matrix[num].tail(20).mean()

        momentum_features = calculate_momentum(binary_matrix[num], [5, 10, 20])

        is_par = 1 if num % 2 == 0 else 0
        is_prime_num = 1 if num in primes else 0
        is_fib = 1 if num in fibonacci_nums else 0
        quadrante = get_quadrante(num)
        linha, coluna = get_linha_coluna_megasena(num)
        zona_num = {'baixa': 1, 'media': 2, 'alta': 3}[get_zona(num)]

        mult_features = [1 if num % d == 0 else 0 for d in [3, 5, 7]]

        avg_corr = correlation_matrix.iloc[num-1, :].mean()

        recent_avg = binary_matrix[num].tail(30).mean()
        overall_avg = binary_matrix[num].mean()
        hot_cold_score = recent_avg - overall_avg

        next_features.extend([
            freq_total, freq_recent_5, freq_recent_10, freq_recent_20,
            cycle_features['gap_atual'], cycle_features['gap_medio'],
            cycle_features['gap_std'], cycle_features['ciclo_regular'],
            cycle_features['prob_ciclo'],
            momentum_features['momentum_5'], momentum_features['momentum_10'],
            momentum_features['momentum_20'],
            is_par, is_prime_num, is_fib, quadrante, linha, coluna, zona_num,
            *mult_features,
            avg_corr, hot_cold_score
        ])

    next_features = np.array([next_features])
    next_features_scaled = scaler.transform(next_features)

    # Obter probabilidades
    try:
        probabilities = []
        for estimator in model.estimators_:
            if hasattr(estimator, 'predict_proba'):
                prob = estimator.predict_proba(next_features_scaled)[0]
                probabilities.append(prob[1] if len(prob) > 1 else prob[0])
            else:
                probabilities.append(estimator.predict(next_features_scaled)[0])
        probabilities = np.array(probabilities)
    except:
        prediction = model.predict(next_features_scaled)[0]
        probabilities = prediction.astype(float)

    # Selecionar top 10
    top_10_indices = np.argsort(probabilities)[-10:]
    top_10_numbers = sorted([idx + 1 for idx in top_10_indices])
    top_10_probs = [probabilities[idx] for idx in top_10_indices]

    return top_10_numbers, top_10_probs, probabilities

# Fazer previsão
predicted_top10, predicted_probs, all_probs = predict_next_game_top10(best_model, scaler)

print(f"\n🎯 TOP 10 DEZENAS MAIS PROVÁVEIS:")
print(f"   {predicted_top10}")

# Análise da previsão
print(f"\n📊 ANÁLISE DA PREVISÃO:")
pred_pares = sum(1 for n in predicted_top10 if n % 2 == 0)
pred_impares = 10 - pred_pares
pred_primos = sum(1 for n in predicted_top10 if n in primes)

print(f"   Pares: {pred_pares} | Ímpares: {pred_impares}")
print(f"   Primos: {pred_primos}")
print(f"   Soma: {sum(predicted_top10)}")
print(f"   Média: {np.mean(predicted_top10):.1f}")

# Distribuição por quadrantes
print(f"\n   Distribuição por Quadrantes:")
for q in range(1, 5):
    nums_q = [n for n in predicted_top10 if get_quadrante(n) == q]
    print(f"      Q{q} (1-15, 16-30, 31-45, 46-60): {len(nums_q)} números {nums_q}")

# Distribuição por zonas
print(f"\n   Distribuição por Zonas:")
for zona in ['baixa', 'media', 'alta']:
    nums_zona = [n for n in predicted_top10 if get_zona(n) == zona]
    print(f"      {zona.capitalize()}: {len(nums_zona)} números {nums_zona}")

# Números mais atrasados (top 15)
gaps_atuais = {}
for num in range(1, 61):
    if binary_matrix[num].any():
        last_idx = binary_matrix[num][::-1].idxmax()
        gaps_atuais[num] = len(binary_matrix) - last_idx
    else:
        gaps_atuais[num] = len(binary_matrix)

nums_atrasados = sorted(gaps_atuais.items(), key=lambda x: x[1], reverse=True)[:15]

print(f"\n⏰ TOP 15 NÚMEROS MAIS ATRASADOS:")
for i, (num, gap) in enumerate(nums_atrasados, 1):
    marcador = "⭐" if num in predicted_top10 else "  "
    print(f"   {i:2d}. {marcador} Dezena {num:2d}: {gap:3d} concursos")

print(f"\n💡 SUGESTÃO DE JOGO:")
print(f"   Escolha 6 números entre os 10 sugeridos: {predicted_top10}")
