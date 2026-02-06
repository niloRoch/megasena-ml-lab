# 🎰 Megasena ML Lab

Mega-Sena usando Machine Learning, engenharia massiva de features, análise estatística e modelos ensemble multioutput.

## 📋 Índice

- [Características Principais](#-características-principais)
- [Análises Implementadas](#-análises-implementadas)
- [Como Usar](#-como-usar)
- [Output do Modelo](#-output-do-modelo)
- [Arquitetura do Modelo](#-arquitetura-do-modelo)
- [Visualizações](#-visualizações)
- [Configurações](#️-configurações-ajustáveis)
- [Interpretação dos Resultados](#-interpretação-dos-resultados)
- [Notas Importantes](#-notas-importantes)
- [Disclaimer](#️-disclaimer)

---

## 🎯 Características Principais

- **Universo completo**: Análise de todas as 60 dezenas
- **Previsão Top 10**: Identifica os 10 números mais prováveis para o próximo sorteio
- **Machine Learning**: Ensemble de 3 algoritmos (Random Forest, Gradient Boosting, Logistic Regression)
- **Features expandidas**: 40+ características por número incluindo ciclos, momentum, comportamento e correlações

---

## 📊 Análises Implementadas

### Estatísticas Avançadas
- Frequências absolutas e relativas
- Análise de ciclos e atrasos
- Momentum e tendências
- Padrões comportamentais
- Volatilidade e consistência

### Padrões Numéricos
- **Trincas correlacionais**: Identifica combinações de 3 números frequentes
- **Pares correlacionais**: Analisa duplas que aparecem juntas
- **Princípio de Pareto (80/20)**: Classifica números por performance
- **Distribuição espacial**: Quadrantes, zonas, linhas e colunas do volante
- **Sequências e saltos**: Padrões de números consecutivos

### Características Matemáticas
- Números primos e Fibonacci
- Múltiplos (2, 3, 4, 5, 6, 7, 8, 9, 10)
- Pares e ímpares
- Somas e distribuições estatísticas
- Assimetria e curtose

---

## 🚀 Como Usar

### Requisitos
```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
```


### Execução no Google Colab

1. Abra o arquivo no Google Colab
2. Execute a primeira célula para fazer upload do arquivo CSV
3. O arquivo deve estar no formato:
   - Separador: `;` (ponto e vírgula)
   - Colunas das bolas sorteadas (6 colunas com números de 1 a 60)

### Formato do CSV
```csv
Concurso;Bola1;Bola2;Bola3;Bola4;Bola5;Bola6
1;4;5;30;33;41;52
2;10;27;40;46;49;58
...
```

---

## 📈 Output do Modelo

### 1. Estatísticas Básicas
- Total de aparições por dezena
- Números mais e menos frequentes
- Médias e desvios padrão

### 2. Análises Especializadas
- **Top 10 Trincas**: Combinações de 3 números mais frequentes
- **Top 10 Pares**: Duplas que aparecem juntas
- **Análise de Pareto**: Classificação A/B/C de performance
- **Padrões de Somas**: Distribuição histórica e intervalos sigma
- **Múltiplos**: Análise detalhada de múltiplos de 3, 6 e 9

### 3. Avaliação dos Modelos
- Acertos médios por jogo (nos últimos 15 concursos)
- Métricas individuais de cada modelo
- Seleção automática do melhor modelo

### 4. Previsão para Próximo Concurso
```
🎯 TOP 10 DEZENAS MAIS PROVÁVEIS:
   [5, 10, 23, 33, 37, 41, 44, 51, 53, 60]

Scores individuais:
   05: 0.8234
   10: 0.7891
   ...
```

### 5. Análise Detalhada da Previsão
- Composição (pares/ímpares, primos, múltiplos)
- Distribuição por quadrantes, zonas e linhas
- Análise de atrasos (números atrasados presentes)
- Trincas e pares históricos na previsão
- Validação com padrões históricos

### 6. Sugestões de Jogos
- **Jogo Balanceado**: 3 pares + 3 ímpares
- **Jogo com Atrasados**: Foca em números atrasados
- **Jogo Pareto A**: Prioriza números classe A

---

## 🧠 Arquitetura do Modelo

### Features por Número (40+)

#### 1. Frequências (4)
- Total, últimos 5, 10 e 20 concursos

#### 2. Ciclos Avançados (9)
- Gap atual, médio, desvio padrão
- Regularidade do ciclo
- Probabilidade baseada em ciclo
- Tendência e aceleração

#### 3. Momentum (3)
- Janelas de 5, 10 e 20 concursos

#### 4. Comportamento (3)
- Volatilidade
- Consistência
- Tendência recente

#### 5. Características Estáticas (7)
- Par/ímpar, primo, Fibonacci
- Quadrante, linha, coluna, zona

#### 6. Múltiplos (5)
- Divisibilidade por 3, 5, 6, 7, 9

#### 7. Correlações (2)
- Média e máxima com outros números

#### 8. Hot/Cold (1)
- Score de aquecimento/esfriamento

#### 9. Pareto (1)
- Classificação A/B/C

#### 10. Trincas e Pares (4)
- Força das trincas e pares associados

#### 11. Posicionamento (2)
- Tendência de linha e coluna

### Ensemble de Modelos
```python
Random Forest: 600 árvores, depth=35
Gradient Boosting: 250 estimadores, learning_rate=0.08
Logistic Regression: C=0.15, max_iter=8000
```

**O modelo final combina as previsões com pesos:**
- ML Score: 50%
- Ciclo Score: 20%
- Trinca Strength: 15%
- Par Strength: 10%
- Pareto Bonus: 5%

---

## 📊 Visualizações

O código gera análises visuais configuradas com:
- **Estilo**: `seaborn-v0_8-darkgrid`
- **Paleta**: `husl`
- **Figuras**: 14x10 inches

---

## ⚙️ Configurações Ajustáveis
```python
# Tamanho do conjunto de teste
test_size = 15

# Janelas de análise
momentum_windows = [5, 10, 20]
behavioral_window = 30

# Normalização
scaler = RobustScaler()  # Robusto a outliers
```

---

## 🎲 Interpretação dos Resultados

### Scores
| Faixa | Interpretação |
|-------|---------------|
| 0.8 - 1.0 | Muito provável |
| 0.6 - 0.8 | Provável |
| 0.4 - 0.6 | Médio |
| < 0.4 | Menos provável |

### Classificação Pareto
- **Classe A**: 50% dos números que geram 40% das aparições
- **Classe B**: Restante dos números do Pareto 80%
- **Classe C**: Fora do Pareto (menos frequentes)

---

## 📝 Notas Importantes

- O modelo utiliza histórico mínimo de 20 concursos para treinamento
- Todas as features são normalizadas com RobustScaler
- A previsão é baseada em padrões históricos e não garante acertos
- Recomenda-se atualizar o dataset regularmente

---

## 🤝 Contribuições

Melhorias sugeridas:
- [ ] Adicionar análise de redes neurais (LSTM)
- [ ] Implementar validação cruzada temporal
- [ ] Criar dashboard interativo
- [ ] Adicionar análise de estações/meses
- [ ] Implementar otimização bayesiana de hiperparâmetros


## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

