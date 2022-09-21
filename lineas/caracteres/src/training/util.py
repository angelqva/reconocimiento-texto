"""Function to train a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
from src.training.lr_find import LearningRateFinder
from src.training.clr_callback import CyclicLR
from src.visualization.visualize import plot_loss, plot_acc, save_model


import time

from keras.callbacks import EarlyStopping
from src.data.dataset import Dataset
from src.models.base_model import Model
from pathlib import Path
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

EARLY_STOPPING = True
CYCLIC_LR = True

MIN_LR = 1e-5
MAX_LR = 1e-3
STEP_SIZE = 8
MODE = "triangular2"
SAVE_LR_PLOT = '../models/'


def train_model(
        model: Model,
        dataset: Dataset,
        epochs: int,
        batch_size: int,
        name: str,
        FIND_LR: bool = False) -> Model:
    """Train model."""
    callbacks = []

    if FIND_LR:
        # inicialice el buscador de tasa de aprendizaje y luego entrene con el aprendizaje
        # tasas que van desde 1e-10 hasta 1e+1
        print("[INFO] encontrar la tasa de aprendizaje...")
        lrf = LearningRateFinder(model)
        lrf.find(
            dataset,
            1e-10,
            1e+1,
            stepsPerEpoch=np.ceil(
                (len(dataset['x_train']) / float(batch_size))),
            batchSize=batch_size)

        # trazar la pérdida para las distintas tasas de aprendizaje y guardar el
        # gráfico resultante en el disco
        lrf.plot_loss(name)

        # salga con gracia del script para que podamos ajustar nuestras tasas de aprendizaje
        # en la configuración y luego entrenar la red para nuestro conjunto completo de
        # épocas
        print("[INFO] buscador de tasa de aprendizaje completo")
        print(
            "[INFO] examinar la trama y ajustar las tasas de aprendizaje antes del entrenamiento")
        sys.exit(0)

    else:
        if EARLY_STOPPING:
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01,
                                           patience=3, verbose=1, restore_best_weights=True)
            callbacks.append(early_stopping)

        if CYCLIC_LR:
            cyclic_lr = CyclicLR(base_lr=MIN_LR, max_lr=MAX_LR,
                                 step_size=STEP_SIZE *
                                 (dataset['x_train'].shape[0] // batch_size),
                                 mode=MODE)
            callbacks.append(cyclic_lr)

        model.network.summary()

        t = time.time()
        _history = model.fit(dataset=dataset,
                             batch_size=batch_size,
                             epochs=epochs,
                             callbacks=callbacks,
                             lr=MIN_LR)
        print('[INFO] El entrenamiento tomó {:2f} s'.format(time.time() - t))

        plot_acc(_history, name)
        plot_loss(_history, name)
        save_model(model.network, name)

        return model
