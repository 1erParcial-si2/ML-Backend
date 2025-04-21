import pandas as pd
import os
import numpy as np
import io
from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from django.conf import settings

class RecommendationSystem:
    def __init__(self):
        self.model = None
        self.mlb = None
        self.is_trained = False
        # Nuevo: almacenar datos CSV en memoria
        self.csv_data = None
        
    def load_data(self, file_path=None):
        """Load and preprocess the training data"""
        # Si tenemos datos CSV en memoria, usarlos
        if self.csv_data is not None:
            # Leer CSV desde la cadena en memoria
            df = pd.read_csv(io.StringIO(self.csv_data))
        else:
            # Usar archivo CSV local como fallback
            if file_path is None:
                # Usar un nombre de archivo compatible con Windows
                file_path = os.path.join(settings.BASE_DIR, 'datasets', 'train_input_target_m2_n_=1.csv')
            
            # 1) Leer CSV desde archivo
            df = pd.read_csv(file_path)
        
        # 2) Convertir cadenas a listas de ints
        df['input'] = df['input'].apply(literal_eval)
        df['target'] = df['target'].apply(literal_eval)
        
        return df
    
    def set_csv_data(self, csv_data):
        """Establece los datos CSV en memoria y marca el modelo como no entrenado"""
        self.csv_data = csv_data
        self.is_trained = False
        return True
    
    def train(self, file_path=None):
        """Train the recommendation model"""
        df = self.load_data(file_path)
        
        # 3) One-hot vectorización
        self.mlb = MultiLabelBinarizer()
        
        # X: (#ejemplos × #productos_totales)
        X = self.mlb.fit_transform(df['input'])
        
        # Y: (#ejemplos × #productos_totales)
        Y = self.mlb.transform(df['target'])
        
        # Entrenar modelo (Random Forest como ejemplo, pero podrías usar otros)
        base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model = MultiOutputClassifier(base_model)
        self.model.fit(X, Y)
        
        self.is_trained = True
        return True
    
    def predict(self, input_products):
        """Generate recommendations for input products"""
        if not self.is_trained:
            self.train()
        
        # Vectorizar
        x_vec = self.mlb.transform([input_products])  # shape (1, #productos_totales)
        
        # En lugar de usar predict_proba que es complejo para MultiOutputClassifier
        # Usaremos una simplificación para el caso de prueba
        Y_pred = self.model.predict(x_vec)[0]
        
        # Obtener índices donde hay 1 (productos recomendados)
        indices = np.where(Y_pred > 0)[0]
        
        if len(indices) == 0:
            # Si no hay recomendaciones, usar el producto más común del dataset
            df = self.load_data()
            # Obtener todos los productos recomendados (target) y contar frecuencias
            all_targets = [item for sublist in df['target'].tolist() for item in sublist]
            if all_targets:
                # Usar el producto más frecuentemente recomendado como valor por defecto
                from collections import Counter
                most_common = Counter(all_targets).most_common(1)[0][0]
                return [most_common]
            else:
                # Si no hay productos en el dataset, devolver el primero de la lista de clases
                if len(self.mlb.classes_) > 0:
                    return [int(self.mlb.classes_[0])]
                # Si no hay clases (muy improbable), devolver un valor seguro
                return [1]
        else:
            # Convertir numpy.int64 a int nativo de Python para evitar problemas de serialización
            candidatos = [int(self.mlb.classes_[i]) for i in indices]
            
        # No recomendar productos que ya están en el input
        candidatos = [p for p in candidatos if p not in input_products]
        
        # Si no hay candidatos después de filtrar, devolver un valor del dataset
        if not candidatos:
            df = self.load_data()
            # Buscar productos que no estén en input_products
            all_products = set(self.mlb.classes_.tolist())
            available_products = list(all_products - set(input_products))
            if available_products:
                # Devolver un producto disponible
                return [int(available_products[0])]
            else:
                # Si todos los productos ya están en input, devolver el primero de input como fallback
                return [int(input_products[0])]
            
        return candidatos
    
    def get_all_products(self):
        """Return all product IDs seen during training"""
        if not self.is_trained:
            self.train()
        
        return self.mlb.classes_.tolist()

# Singleton instance
recommendation_system = RecommendationSystem() 