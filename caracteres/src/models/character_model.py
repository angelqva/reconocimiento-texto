"""
Character Model class
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
from src.models.base_model import Model
from src.data.emnist_dataset import EMNIST
from src.networks.lenet import lenet


class Character_Model(Model):
    """
    Caracteres Model Class
    """

    def __init__(self,
                 network_fn: Callable = lenet,
                 dataset: type = EMNIST):
        """Definir la clase de red predeterminada y la clase de conjunto de datos"""
        super().__init__(network_fn, dataset)

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)
        pred_raw = self.network.predict(
            np.expand_dims(image, 0), batch_size=1).flatten()
        ind = np.argmax(pred_raw)
        confidence_of_prediction = pred_raw[ind]
        # el diccionario de mapeo de entero a carácter es self.data.mapping[integer]
        predicted_character = self.data.mapping[ind]
        return predicted_character, confidence_of_prediction
