"""Clase CharacterPredictor"""
import matplotlib.pyplot as plt
import matplotlib
import imageio
from src.networks.custom import customCNN
from src.networks.resnet import resnet
from src.networks.lenet import lenet
from src.models.character_model import Character_Model
from typing import Tuple, Union

import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

matplotlib.use('TkAgg')


class CharacterPredictor:
    """Ante una imagen de un solo carácter escrito a mano, lo reconoce."""

    def __init__(self):
        self.model = Character_Model(customCNN)
        self.model.load_weights()

    def predict(self, image_or_filename: Union[np.ndarray, str]) -> Tuple[str, float]:
        """Predecir en una sola imagen."""
        if isinstance(image_or_filename, str):
            image = imageio.imread(image_or_filename, pilmode='L')
        else:
            image = image_or_filename
        return self.model.predict_on_image(image)

    def evaluate(self, dataset):
        """Evaluar en un conjunto de datos."""
        return self.model.evaluate(dataset.x_test, dataset.y_test)
