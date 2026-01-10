"""
Módulo de detección de objetos usando CNN
"""
import tensorflow as tf
import numpy as np

class DetectorCNN:
    def __init__(self, model_path=None):
        self.classes = ['arma', 'gorro', 'mascara', 'persona']
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """ Carga el modelo si existe, si no, devuelve None para entrenar """
        if model_path:
            return tf.keras.models.load_model(model_path)
        return None

    def detect_objects(self, frame, threshold=0.5):
        """ Lógica principal de predicción """
        # Preprocesamiento rápido (esto debería venir de lo que haga D'Alessandro)
        # Predicción: self.model.predict(frame)
        pass

    def draw_detections(self, frame, detections):
        """ Esta función podría ir aquí o en src/utils.py (D'Alessandro) """
        pass
