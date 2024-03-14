import numpy as np
from sklearn.linear_model import LogisticRegression
from pickle import load

class DigitClassifier:
    def __init__(self, model_path: str) -> None:
        self.__model: LogisticRegression = None
        self.load_model(model_path)
    
    def predict_probabilities(self, img: np.ndarray) -> np.ndarray:
        flat_img = self.png_to_flat_mnist(img)
        return self.__model.predict_proba(flat_img)[0]
    
    def predict(self, img: np.ndarray):
        flat_img = self.png_to_flat_mnist(img)
        return self.__model.predict(flat_img)[0]
    
    def png_to_flat_mnist(self, img: np.ndarray) -> np.ndarray:
        return ((255 - img) / 255).reshape((1, -1))
    
    def load_model(self, filepath: str):
        with open(filepath, 'rb') as mfile:
            self.__model = load(mfile)
    