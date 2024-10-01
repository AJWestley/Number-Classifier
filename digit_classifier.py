import numpy as np
from pickle import load
import tensorflow as tf

class DigitClassifier:
    def __init__(self, model_path: str) -> None:
        self.__model = tf.keras.models.load_model(model_path)
    
    def predict_probabilities(self, img: np.ndarray) -> np.ndarray:
        img = self.transform(img)
        return self.__model.predict(img)[0]
    
    def predict(self, img: np.ndarray):
        img = self.transform(img)
        return np.argmax(self.__model.predict(img)[0])
    
    def transform(self, img: np.ndarray) -> np.ndarray:
        return ((255 - img) / 255).reshape(-1, 28, 28, 1)
    
    def load_model(self, filepath: str):
        with open(filepath, 'rb') as mfile:
            self.__model = load(mfile)
    