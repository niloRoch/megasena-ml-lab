# -*- coding: utf-8 -*-
"""
Modelo Mega-Sena - Machine Learning v2.0
Universo: 60 dezenas | Previsão: Top 10 números mais prováveis
Features expandidas: Trincas, Pareto, Ciclos Avançados, Comportamento, Sequências
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
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

# Identificar colunas das bolas
ball_columns = [col for col in df.columns if 'Bola' in col or 'Dezena' in col or col.isdigit()]
if not ball_columns:
    ball_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and df[col].between(1, 60).all()]

if len(ball_columns) < 6:
    print("⚠️ Atenção: Por favor, ajuste as colunas das bolas manualmente")

balls = ball_columns[:6]
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
    linha = (num - 1) // 10 + 1
    coluna = (num - 1) % 10 + 1
    return linha, coluna

def get_zona(num):
    """Divide em 3 zonas: Baixa (1-20), Média (21-40), Alta (41-60)"""
    if num <= 20:
        return 'baixa'
    elif num <= 40:
        return 'media'
    else:
        return 'alta'

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

    # Tendência dos últimos 3 ciclos (está diminuindo ou aumentando?)
    tendencia_ciclo = 0
    if len(gaps) >= 3:
        ultimos_3 = gaps[-3:]
        tendencia_ciclo = (ultimos_3[-1] - ultimos_3[0]) / 3

    # Aceleração (mudança na tendência)
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
    - Consistência de aparições
    - Padrão de distribuição temporal
    - Volatilidade
    """
    if current_idx < window:
        window = current_idx
    
    recent = binary_series[current_idx-window:current_idx]
    
    # Volatilidade (quão irregular são as aparições)
    volatilidade = recent.std()
    
    # Consistência (aparições uniformemente distribuídas)
    indices_aparicoes = recent[recent == 1].index.tolist()
    if len(indices_aparicoes) > 1:
        intervalos = np.diff(indices_aparicoes)
        consistencia = 1 - (np.std(intervalos) / (np.mean(intervalos) + 1))
    else:
        consistencia = 0
    
    # Tendência recente (últimos 10 vs últimos 20)
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

# ==================== CONSTANTES ====================
primes = [n for n in range(1, 61) if is_prime(n)]
fibonacci_nums = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
pares = [n for n in range(2, 61, 2)]
impares = [n for n in range(1, 61, 2)]
mult_3 = [n for n in range(3, 61, 3)]
mult_6 = [n for n in range(6, 61, 6)]
mult_9 = [n for n in range(9, 61, 9)]

print(f"\n📊 Números primos: {len(primes)} números")
print(f"📊 Números Fibonacci: {fibonacci_nums}")
print(f"📊 Pares: {len(pares)} | Ímpares: {len(impares)}")
print(f"📊 Múltiplos de 3: {len(mult_3)} | Múltiplos de 6: {len(mult_6)} | Múltiplos de 9: {len(mult_9)}")

# ==================== MATRIZ BINÁRIA ====================
binary_matrix = pd.DataFrame(index=df.index, columns=range(1, 61), dtype=int)
for num in range(1, 61):
    binary_matrix[num] = df_balls.isin([num]).any(axis=1).astype(int)

print("\n✅ Matriz binária criada (60 dezenas)")

# ==================== ANÁLISE DE TRINCAS CORRELACIONAIS ====================
print("\n🔍 Analisando trincas correlacionais...")

trincas_freq = Counter()
for _, row in df_balls.iterrows():
    nums = sorted(row.values)
    for trinca in combinations(nums, 3):
        trincas_freq[trinca] += 1

# Top 20 trincas mais frequentes
top_trincas = trincas_freq.most_common(20)
print("\nTop 10 trincas mais frequentes:")
for i, (trinca, freq) in enumerate(top_trincas[:10], 1):
    print(f"  {i:2d}. {trinca}: {freq} vezes")

# Dicionário de trincas por número
trincas_por_numero = defaultdict(list)
for trinca, freq in trincas_freq.items():
    for num in trinca:
        trincas_por_numero[num].append((trinca, freq))

# Ordenar trincas de cada número por frequência
for num in trincas_por_numero:
    trincas_por_numero[num] = sorted(trincas_por_numero[num], key=lambda x: x[1], reverse=True)

# ==================== ANÁLISE DE PARES CORRELACIONAIS ====================
print("\n🔍 Analisando pares correlacionais...")

pares_freq = Counter()
for _, row in df_balls.iterrows():
    nums = sorted(row.values)
    for par in combinations(nums, 2):
        pares_freq[par] += 1

top_pares = pares_freq.most_common(20)
print("\nTop 10 pares mais frequentes:")
for i, (par, freq) in enumerate(top_pares[:10], 1):
    print(f"  {i:2d}. {par}: {freq} vezes")

# Dicionário de pares por número
pares_por_numero = defaultdict(list)
for par, freq in pares_freq.items():
    for num in par:
        pares_por_numero[num].append((par, freq))

for num in pares_por_numero:
    pares_por_numero[num] = sorted(pares_por_numero[num], key=lambda x: x[1], reverse=True)

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

# ==================== ANÁLISE DE PARETO (80/20) ====================
print("\n📊 Análise de Pareto (Princípio 80/20)...")

freq_sorted = freq_abs.sort_values(ascending=False)
freq_cumsum = freq_sorted.cumsum()
freq_cumsum_pct = (freq_cumsum / freq_cumsum.max()) * 100

# Encontrar quantos números representam 80% das aparições
pareto_80_count = (freq_cumsum_pct <= 80).sum()
pareto_80_nums = freq_cumsum_pct[freq_cumsum_pct <= 80].index.tolist()

print(f"\n🎯 {pareto_80_count} números representam 80% das aparições:")
print(f"   {pareto_80_nums}")

# Classificação Pareto para cada número
pareto_class = {}
for num in range(1, 61):
    if num in pareto_80_nums[:int(pareto_80_count * 0.5)]:
        pareto_class[num] = 'A'  # Top 50% do Pareto
    elif num in pareto_80_nums:
        pareto_class[num] = 'B'  # Restante do Pareto 80%
    else:
        pareto_class[num] = 'C'  # Fora do Pareto

# ==================== ANÁLISE DE SOMAS ====================
print("\n➕ Analisando padrões de somas...")

somas_historico = df_balls.sum(axis=1)
soma_media = somas_historico.mean()
soma_std = somas_historico.std()
soma_min = somas_historico.min()
soma_max = somas_historico.max()

print(f"Soma média: {soma_media:.1f}")
print(f"Desvio padrão: {soma_std:.1f}")
print(f"Intervalo: [{soma_min}, {soma_max}]")
print(f"Intervalo 1σ: [{soma_media-soma_std:.1f}, {soma_media+soma_std:.1f}]")
print(f"Intervalo 2σ: [{soma_media-2*soma_std:.1f}, {soma_media+2*soma_std:.1f}]")

# Distribuição de somas por faixas
soma_bins = [0, 120, 150, 180, 210, 240, 300]
soma_labels = ['<120', '120-150', '150-180', '180-210', '210-240', '>240']
df['faixa_soma'] = pd.cut(somas_historico, bins=soma_bins, labels=soma_labels)
print("\nDistribuição de somas:")
print(df['faixa_soma'].value_counts().sort_index())

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

# 3. MÚLTIPLOS (incluindo 3, 6, 9)
for divisor in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    df[f'mult_{divisor}'] = df_balls.apply(lambda x: sum(n % divisor == 0 for n in x), axis=1)

# Análise específica de múltiplos de 3, 6, 9
mult_3_historico = df['mult_3']
mult_6_historico = df['mult_6']
mult_9_historico = df['mult_9']

print(f"\nMúltiplos de 3 - Média: {mult_3_historico.mean():.2f}, Moda: {mult_3_historico.mode()[0]}")
print(f"Múltiplos de 6 - Média: {mult_6_historico.mean():.2f}, Moda: {mult_6_historico.mode()[0]}")
print(f"Múltiplos de 9 - Média: {mult_9_historico.mean():.2f}, Moda: {mult_9_historico.mode()[0]}")

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

# 6. POSIÇÕES NO VOLANTE (linhas e colunas)
for linha in range(1, 7):
    df[f'linha_{linha}'] = df_balls.apply(
        lambda x: sum(get_linha_coluna_megasena(n)[0] == linha for n in x), axis=1
    )

linha_counts = {i: df[f'linha_{i}'].sum() for i in range(1, 7)}
print(f"\nDistribuição histórica por linhas:")
for linha, count in linha_counts.items():
    print(f"  Linha {linha}: {count} números ({count/total_aparicoes*100:.1f}%)")

for coluna in range(1, 11):
    df[f'coluna_{coluna}'] = df_balls.apply(
        lambda x: sum(get_linha_coluna_megasena(n)[1] == coluna for n in x), axis=1
    )

coluna_counts = {i: df[f'coluna_{i}'].sum() for i in range(1, 11)}
print(f"\nDistribuição histórica por colunas:")
for coluna, count in coluna_counts.items():
    print(f"  Coluna {coluna}: {count} números ({count/total_aparicoes*100:.1f}%)")

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

sequencias_historico = df['sequencias']
print(f"\nSequências - Média: {sequencias_historico.mean():.2f}, Moda: {sequencias_historico.mode()[0]}")

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

# 11. DISTRIBUIÇÃO ESPACIAL
df['dist_espacial'] = df_balls.apply(
    lambda x: np.mean([sorted(x)[i+1] - sorted(x)[i] for i in range(5)]),
    axis=1
)

# 12. ASSIMETRIA E CURTOSE
df['assimetria'] = df_balls.apply(lambda x: stats.skew(x), axis=1)
df['curtose'] = df_balls.apply(lambda x: stats.kurtosis(x), axis=1)

# 13. CONCENTRAÇÃO
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

for i in range(20, len(df)):  # Histórico mínimo de 20 concursos
    features_concurso = []

    for num in range(1, 61):
        # 1. Features de ciclo (expandidas)
        cycle_features = calculate_cycle_features(binary_matrix[num], i)

        # 2. Features de frequência
        freq_total = binary_matrix[num][:i].mean()
        freq_recent_5 = binary_matrix[num][i-5:i].mean()
        freq_recent_10 = binary_matrix[num][i-10:i].mean()
        freq_recent_20 = binary_matrix[num][i-20:i].mean()

        # 3. Momentum
        momentum_features = calculate_momentum(binary_matrix[num][:i], [5, 10, 20])

        # 4. Comportamento
        behavioral = calculate_behavioral_score(binary_matrix[num], i, window=30)

        # 5. Características estáticas
        is_par = 1 if num % 2 == 0 else 0
        is_prime_num = 1 if num in primes else 0
        is_fib = 1 if num in fibonacci_nums else 0
        quadrante = get_quadrante(num)
        linha, coluna = get_linha_coluna_megasena(num)
        zona_num = {'baixa': 1, 'media': 2, 'alta': 3}[get_zona(num)]

        # 6. Múltiplos (incluindo 3, 6, 9)
        mult_features = [1 if num % d == 0 else 0 for d in [3, 5, 6, 7, 9]]

        # 7. Correlação média
        avg_corr = correlation_matrix.iloc[num-1, :].mean()
        max_corr = correlation_matrix.iloc[num-1, :].max()

        # 8. Hot/Cold
        recent_avg = binary_matrix[num][max(0,i-30):i].mean()
        overall_avg = binary_matrix[num][:i].mean()
        hot_cold_score = recent_avg - overall_avg

        # 9. Pareto
        pareto_score = {'A': 3, 'B': 2, 'C': 1}[pareto_class[num]]

        # 10. Score de trincas (força das trincas que contêm este número)
        if num in trincas_por_numero and len(trincas_por_numero[num]) > 0:
            top_trincas_num = trincas_por_numero[num][:5]
            trinca_score = sum(freq for _, freq in top_trincas_num) / len(top_trincas_num)
            trinca_max = top_trincas_num[0][1]
        else:
            trinca_score = 0
            trinca_max = 0

        # 11. Score de pares
        if num in pares_por_numero and len(pares_por_numero[num]) > 0:
            top_pares_num = pares_por_numero[num][:5]
            par_score = sum(freq for _, freq in top_pares_num) / len(top_pares_num)
            par_max = top_pares_num[0][1]
        else:
            par_score = 0
            par_max = 0

        # 12. Atraso normalizado
        atraso_norm = cycle_features['gap_atual'] / (cycle_features['gap_medio'] + 1)

        # 13. Tendência de linha/coluna
        linha_freq = binary_matrix[[n for n in range(1, 61) if get_linha_coluna_megasena(n)[0] == linha]][:i].sum().sum()
        coluna_freq = binary_matrix[[n for n in range(1, 61) if get_linha_coluna_megasena(n)[1] == coluna]][:i].sum().sum()
        linha_score = linha_freq / (i * 10)  # Normalizado
        coluna_score = coluna_freq / (i * 6)  # Normalizado

        # Adicionar features
        features_concurso.extend([
            # Frequências (4)
            freq_total, freq_recent_5, freq_recent_10, freq_recent_20,
            # Ciclos expandidos (9)
            cycle_features['gap_atual'], cycle_features['gap_medio'],
            cycle_features['gap_std'], cycle_features['ciclo_regular'],
            cycle_features['prob_ciclo'], cycle_features['tendencia_ciclo'],
            cycle_features['aceleracao_ciclo'], atraso_norm,
            cycle_features['gap_max'] - cycle_features['gap_min'],  # Variação de gap
            # Momentum (3)
            momentum_features['momentum_5'], momentum_features['momentum_10'],
            momentum_features['momentum_20'],
            # Comportamento (3)
            behavioral['volatilidade'], behavioral['consistencia'],
            behavioral['tendencia_recente'],
            # Estáticas (7)
            is_par, is_prime_num, is_fib, quadrante, linha, coluna, zona_num,
            # Múltiplos (5)
            *mult_features,
            # Correlação (2)
            avg_corr, max_corr,
            # Hot/Cold (1)
            hot_cold_score,
            # Pareto (1)
            pareto_score,
            # Trincas e Pares (4)
            trinca_score, trinca_max, par_score, par_max,
            # Linha/Coluna tendência (2)
            linha_score, coluna_score
        ])

    X.append(features_concurso)
    y.append(binary_matrix.iloc[i, :].values)

X = np.array(X)
y = np.array(y)

num_features_per_number = len(features_concurso) // 60

print(f"✅ Dataset preparado:")
print(f"   Amostras: {X.shape[0]}")
print(f"   Features por número: {num_features_per_number}")
print(f"   Total de features: {X.shape[1]}")

# ==================== DIVISÃO TREINO/TESTE ====================
test_size = 15
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
    n_estimators=600,
    max_depth=35,
    min_samples_split=4,
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
    n_estimators=250,
    learning_rate=0.08,
    max_depth=12,
    min_samples_split=4,
    subsample=0.8,
    random_state=42
)
gb = MultiOutputClassifier(gb_model)
gb.fit(X_train_scaled, y_train)
gb_pred = gb.predict(X_test_scaled)

# Modelo 3: Logistic Regression
print("📊 Logistic Regression...")
lr_model = LogisticRegression(
    max_iter=8000,
    C=0.15,
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

def predict_next_game_top10_advanced(model, scaler):
    """Prevê top 10 números com análise multi-critério"""
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
        behavioral = calculate_behavioral_score(binary_matrix[num], current_idx, window=30)

        is_par = 1 if num % 2 == 0 else 0
        is_prime_num = 1 if num in primes else 0
        is_fib = 1 if num in fibonacci_nums else 0
        quadrante = get_quadrante(num)
        linha, coluna = get_linha_coluna_megasena(num)
        zona_num = {'baixa': 1, 'media': 2, 'alta': 3}[get_zona(num)]

        mult_features = [1 if num % d == 0 else 0 for d in [3, 5, 6, 7, 9]]

        avg_corr = correlation_matrix.iloc[num-1, :].mean()
        max_corr = correlation_matrix.iloc[num-1, :].max()

        recent_avg = binary_matrix[num].tail(30).mean()
        overall_avg = binary_matrix[num].mean()
        hot_cold_score = recent_avg - overall_avg

        pareto_score = {'A': 3, 'B': 2, 'C': 1}[pareto_class[num]]

        if num in trincas_por_numero and len(trincas_por_numero[num]) > 0:
            top_trincas_num = trincas_por_numero[num][:5]
            trinca_score = sum(freq for _, freq in top_trincas_num) / len(top_trincas_num)
            trinca_max = top_trincas_num[0][1]
        else:
            trinca_score = 0
            trinca_max = 0

        if num in pares_por_numero and len(pares_por_numero[num]) > 0:
            top_pares_num = pares_por_numero[num][:5]
            par_score = sum(freq for _, freq in top_pares_num) / len(top_pares_num)
            par_max = top_pares_num[0][1]
        else:
            par_score = 0
            par_max = 0

        atraso_norm = cycle_features['gap_atual'] / (cycle_features['gap_medio'] + 1)

        linha_freq = binary_matrix[[n for n in range(1, 61) if get_linha_coluna_megasena(n)[0] == linha]].sum().sum()
        coluna_freq = binary_matrix[[n for n in range(1, 61) if get_linha_coluna_megasena(n)[1] == coluna]].sum().sum()
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

    # Criar score combinado
    scores = {}
    for num in range(1, 61):
        idx = num - 1
        
        # Score do modelo ML
        ml_score = probabilities[idx]
        
        # Score de ciclo/atraso
        cycle_info = calculate_cycle_features(binary_matrix[num], current_idx)
        cycle_score = cycle_info['prob_ciclo'] * 0.3
        
        # Score de trincas
        if num in trincas_por_numero and len(trincas_por_numero[num]) > 0:
            trinca_strength = trincas_por_numero[num][0][1] / max(1, trincas_freq.most_common(1)[0][1])
        else:
            trinca_strength = 0
        
        # Score de pares
        if num in pares_por_numero and len(pares_por_numero[num]) > 0:
            par_strength = pares_por_numero[num][0][1] / max(1, pares_freq.most_common(1)[0][1])
        else:
            par_strength = 0
        
        # Score Pareto
        pareto_bonus = {'A': 0.2, 'B': 0.1, 'C': 0}[pareto_class[num]]
        
        # Score combinado
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

# Fazer previsão
predicted_top10, predicted_scores, all_probs, all_scores = predict_next_game_top10_advanced(best_model, scaler)

print(f"\n🎯 TOP 10 DEZENAS MAIS PROVÁVEIS:")
print(f"   {predicted_top10}")

print(f"\nScores individuais:")
for num in predicted_top10:
    print(f"   {num:2d}: {all_scores[num]:.4f}")

# ==================== ANÁLISE DETALHADA DA PREVISÃO ====================
print(f"\n" + "="*60)
print("ANÁLISE DETALHADA DA PREVISÃO")
print("="*60)

# Análise básica
pred_pares = sum(1 for n in predicted_top10 if n % 2 == 0)
pred_impares = 10 - pred_pares
pred_primos = sum(1 for n in predicted_top10 if n in primes)
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

# Distribuição por zonas
print(f"\n🌡️ Distribuição por Zonas:")
for zona in ['baixa', 'media', 'alta']:
    nums_zona = [n for n in predicted_top10 if get_zona(n) == zona]
    print(f"   {zona.capitalize()}: {len(nums_zona)} números {nums_zona if nums_zona else '-'}")

# Distribuição por linhas
print(f"\n📏 Distribuição por Linhas (volante 6x10):")
for linha in range(1, 7):
    nums_linha = [n for n in predicted_top10 if get_linha_coluna_megasena(n)[0] == linha]
    print(f"   Linha {linha}: {len(nums_linha)} números {nums_linha if nums_linha else '-'}")

# Distribuição por colunas
print(f"\n📐 Distribuição por Colunas:")
for coluna in range(1, 11):
    nums_coluna = [n for n in predicted_top10 if get_linha_coluna_megasena(n)[1] == coluna]
    if nums_coluna:
        print(f"   Coluna {coluna}: {len(nums_coluna)} números {nums_coluna}")

# Sequências
pred_sorted = sorted(predicted_top10)
pred_sequencias = sum(1 for i in range(len(pred_sorted)-1) if pred_sorted[i+1] - pred_sorted[i] == 1)
print(f"\n🔢 Sequências consecutivas: {pred_sequencias}")

# Saltos
saltos = [pred_sorted[i+1] - pred_sorted[i] for i in range(len(pred_sorted)-1)]
print(f"   Salto médio: {np.mean(saltos):.1f}")
print(f"   Salto mínimo: {min(saltos)}")
print(f"   Salto máximo: {max(saltos)}")

# Números atrasados
print(f"\n⏰ ANÁLISE DE ATRASOS:")
gaps_atuais = {}
for num in range(1, 61):
    if binary_matrix[num].any():
        last_idx = binary_matrix[num][::-1].idxmax()
        gaps_atuais[num] = len(binary_matrix) - last_idx
    else:
        gaps_atuais[num] = len(binary_matrix)

nums_atrasados = sorted(gaps_atuais.items(), key=lambda x: x[1], reverse=True)[:20]

print(f"\nTop 20 números mais atrasados:")
for i, (num, gap) in enumerate(nums_atrasados, 1):
    marcador = "⭐" if num in predicted_top10 else "  "
    pareto_mark = f"[{pareto_class[num]}]"
    print(f"   {i:2d}. {marcador} {pareto_mark} Dezena {num:2d}: {gap:3d} concursos")

# Análise de trincas previstas
print(f"\n🔗 TRINCAS PRESENTES NA PREVISÃO:")
trincas_previstas = list(combinations(predicted_top10, 3))
trincas_previstas_freq = [(t, trincas_freq.get(t, 0)) for t in trincas_previstas]
trincas_previstas_freq.sort(key=lambda x: x[1], reverse=True)

print(f"   Top 5 trincas históricas presentes:")
for i, (trinca, freq) in enumerate(trincas_previstas_freq[:5], 1):
    if freq > 0:
        print(f"      {i}. {trinca}: {freq} vezes")

# Análise de pares previstos
print(f"\n👥 PARES PRESENTES NA PREVISÃO:")
pares_previstos = list(combinations(predicted_top10, 2))
pares_previstos_freq = [(p, pares_freq.get(p, 0)) for p in pares_previstos]
pares_previstos_freq.sort(key=lambda x: x[1], reverse=True)

print(f"   Top 5 pares históricos presentes:")
for i, (par, freq) in enumerate(pares_previstos_freq[:5], 1):
    if freq > 0:
        print(f"      {i}. {par}: {freq} vezes")

# Classificação Pareto
pareto_distribution = Counter(pareto_class[n] for n in predicted_top10)
print(f"\n📊 Distribuição Pareto:")
print(f"   Classe A (top performers): {pareto_distribution.get('A', 0)} números")
print(f"   Classe B (médio): {pareto_distribution.get('B', 0)} números")
print(f"   Classe C (baixo): {pareto_distribution.get('C', 0)} números")

# Validação com padrões históricos
print(f"\n✅ VALIDAÇÃO COM PADRÕES HISTÓRICOS:")
print(f"   Soma da previsão: {sum(predicted_top10)} (histórico: {soma_media:.0f} ± {soma_std:.0f})")
print(f"   Pares/Ímpares: {pred_pares}/{pred_impares} (moda histórica: {df['pares'].mode()[0]}/{6-df['pares'].mode()[0]})")
print(f"   Múltiplos de 3: {pred_mult_3} (média histórica: {mult_3_historico.mean():.1f})")
print(f"   Sequências: {pred_sequencias} (moda histórica: {sequencias_historico.mode()[0]})")

print(f"\n" + "="*60)
print(f"💡 SUGESTÕES DE JOGOS ")
print(f"="*60)

# Sugestão 1: Balanceado
print(f"\n1️⃣ JOGO BALANCEADO (3 pares, 3 ímpares):")
pares_top10 = [n for n in predicted_top10 if n % 2 == 0]
impares_top10 = [n for n in predicted_top10 if n % 2 != 0]
if len(pares_top10) >= 3 and len(impares_top10) >= 3:
    jogo1 = sorted(pares_top10[:3] + impares_top10[:3])
    print(f"   {jogo1}")
else:
    print(f"   Não há combinação 3-3 disponível")

# Sugestão 2: Baseado em atrasos
print(f"\n2️⃣ JOGO COM NÚMEROS ATRASADOS:")
nums_atrasados_top10 = [n for n, _ in nums_atrasados if n in predicted_top10][:6]
if len(nums_atrasados_top10) >= 6:
    print(f"   {sorted(nums_atrasados_top10)}")
else:
    print(f"   Números disponíveis: {sorted(nums_atrasados_top10)}")

# Sugestão 3: Baseado em Pareto A
print(f"\n3️⃣ JOGO FOCADO EM PARETO CLASSE A:")
pareto_a_nums = [n for n in predicted_top10 if pareto_class[n] == 'A']
if len(pareto_a_nums) >= 4:
    outros = [n for n in predicted_top10 if pareto_class[n] != 'A'][:2]
    jogo3 = sorted(pareto_a_nums[:4] + outros)
    print(f"   {jogo3}")
else:
    print(f"   Números Pareto A disponíveis: {sorted(pareto_a_nums)}")

print(f"\nFaz Teu nome!!!")
