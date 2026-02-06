# -*- coding: utf-8 -*-
"""
Script principal do projeto Mega-Sena ML
"""

import warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_megasena_data
from src.statistical_analysis import StatisticalAnalyzer
from src.correlation_analysis import CorrelationAnalyzer
from src.feature_engineering import FeatureEngineer
from src.models import MegaSenaMLModels
from src.evaluation import ModelEvaluator
from src.prediction import MegaSenaPredictor
from src.utils import print_section_header


def main(file_path, sep=';'):
    """
    Função principal do pipeline
    
    Args:
        file_path (str): Caminho para o arquivo CSV
        sep (str): Separador do CSV
    """
    print_section_header("🎰 MEGA-SENA MACHINE LEARNING v2.0", char='=', width=70)
    
    # 1. CARREGAR DADOS
    print("\n📂 Carregando dados...")
    df, df_balls, binary_matrix = load_megasena_data(file_path, sep, verbose=True)
    
    # 2. ANÁLISES ESTATÍSTICAS
    stat_analyzer = StatisticalAnalyzer(df, df_balls, binary_matrix)
    df = stat_analyzer.run_all_analyses()
    
    # 3. ANÁLISES DE CORRELAÇÃO
    corr_analyzer = CorrelationAnalyzer(df_balls, binary_matrix)
    correlation_results = corr_analyzer.run_all_analyses()
    
    # 4. ENGENHARIA DE FEATURES
    feature_engineer = FeatureEngineer(df, binary_matrix, correlation_results)
    X, y = feature_engineer.build_dataset()
    
    # 5. TREINAMENTO DOS MODELOS
    ml_models = MegaSenaMLModels(X, y)
    models = ml_models.train_all_models()
    
    # 6. AVALIAÇÃO
    evaluator = ModelEvaluator(
        ml_models.y_test, 
        len(df), 
        ml_models.test_size
    )
    
    predictions = ml_models.get_predictions()
    results = evaluator.evaluate_all_models(predictions)
    best_model_name, scores = evaluator.compare_models(results)
    
    # 7. SELECIONAR MELHOR MODELO
    ml_models.select_best_model(scores)
    
    # 8. PREVISÃO DO PRÓXIMO CONCURSO
    print_section_header(
        f"PREVISÃO PARA O PRÓXIMO CONCURSO ({len(df) + 1})", 
        char='=', 
        width=70
    )
    
    predictor = MegaSenaPredictor(
        df, df_balls, binary_matrix, 
        correlation_results, feature_engineer
    )
    
    predicted_top10, scores, probabilities, all_scores = predictor.predict_top10(
        ml_models.best_model, 
        ml_models.scaler
    )
    
    predictor.analyze_prediction(predicted_top10, all_scores)
    
    print("\n" + "="*70)
    print("✅ PIPELINE CONCLUÍDO COM SUCESSO!")
    print("="*70)
    
    return {
        'df': df,
        'df_balls': df_balls,
        'binary_matrix': binary_matrix,
        'correlation_results': correlation_results,
        'models': models,
        'best_model': ml_models.best_model,
        'best_model_name': ml_models.best_name,
        'predicted_top10': predicted_top10,
        'all_scores': all_scores
    }


if __name__ == "__main__":
    # Exemplo de uso
    # Descomente e ajuste o caminho do arquivo
    
    # Para Google Colab
    # from google.colab import files
    # print("Por favor, faça o upload do arquivo CSV da Mega-Sena")
    # uploaded = files.upload()
    # file_path = list(uploaded.keys())[0]
    
    # Para uso local
    file_path = "data/raw/megasena_historico.csv"
    
    results = main(file_path, sep=';')
    
    print("\n🎯 Top 10 números previstos:")
    print(results['predicted_top10'])
