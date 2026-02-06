# -*- coding: utf-8 -*-
"""
Carregamento e preparação inicial dos dados
"""

import pandas as pd
import numpy as np
from .constants import MIN_NUMBER, MAX_NUMBER


class MegaSenaDataLoader:
    """Classe para carregar e processar dados da Mega-Sena"""
    
    def __init__(self, filepath):
        """
        Inicializa o loader
        
        Args:
            filepath: Caminho para o arquivo CSV
        """
        self.filepath = filepath
        self.df = None
        self.df_balls = None
        self.ball_columns = None
        self.binary_matrix = None
        
    def load_data(self, sep=';'):
        """Carrega dados do CSV"""
        self.df = pd.read_csv(self.filepath, sep=sep)
        print(f"✅ Dados carregados: {len(self.df)} concursos")
        print(f"Colunas disponíveis: {self.df.columns.tolist()}")
        return self
    
    def identify_ball_columns(self):
        """Identifica as colunas que contêm as bolas sorteadas"""
        # Tentar identificar por nome
        self.ball_columns = [col for col in self.df.columns 
                            if 'Bola' in col or 'Dezena' in col or col.isdigit()]
        
        # Se não encontrar, tentar por tipo e valores
        if not self.ball_columns:
            self.ball_columns = [col for col in self.df.columns 
                                if self.df[col].dtype in ['int64', 'float64'] 
                                and self.df[col].between(MIN_NUMBER, MAX_NUMBER).all()]
        
        if len(self.ball_columns) < 6:
            raise ValueError("⚠️ Não foi possível identificar as 6 colunas das bolas")
        
        self.ball_columns = self.ball_columns[:6]
        self.df_balls = self.df[self.ball_columns].copy()
        
        print(f"Colunas das bolas: {self.ball_columns}")
        print(f"Amostra dos dados:")
        print(self.df_balls.head())
        
        return self
    
    def create_binary_matrix(self):
        """Cria matriz binária de presença/ausência de números"""
        self.binary_matrix = pd.DataFrame(
            index=self.df.index, 
            columns=range(MIN_NUMBER, MAX_NUMBER + 1), 
            dtype=int
        )
        
        for num in range(MIN_NUMBER, MAX_NUMBER + 1):
            self.binary_matrix[num] = self.df_balls.isin([num]).any(axis=1).astype(int)
        
        print(f"\n✅ Matriz binária criada ({MAX_NUMBER} dezenas)")
        return self
    
    def get_data(self):
        """Retorna os dados processados"""
        return {
            'df': self.df,
            'df_balls': self.df_balls,
            'binary_matrix': self.binary_matrix,
            'ball_columns': self.ball_columns
        }