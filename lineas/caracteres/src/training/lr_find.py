from keras.callbacks import LambdaCallback
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tempfile
SAVE_LR_PLOT = 'caracteres/models/'


class LearningRateFinder:
    def __init__(self, model, stopFactor=4, beta=0.98):
        # almacenar el modelo, el factor de parada y el valor beta (para calcular
        # una pérdida media suavizada)
        self.model = model
        self.stopFactor = stopFactor
        self.beta = beta

        # inicializar nuestra lista de tasas de aprendizaje y pérdidas,
        # respectivamente
        self.lrs = []
        self.losses = []

        # inicializar nuestro multiplicador de tasa de aprendizaje, pérdida promedio, mejor
        # pérdida encontrada hasta el momento, número de lote actual y archivo de pesos
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def reset(self):
        # reinicia todas las variables de nuestro constructor(masculino)
        self.lrs = []
        self.losses = []
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def on_batch_end(self, batch, logs):
        # tome la tasa de aprendizaje actual y agréguela a la lista de
        # tasas de aprendizaje que hemos probado
        lr = K.get_value(self.model.network.optimizer.lr)
        self.lrs.append(lr)

        # tomar la pérdida al final de este lote, incrementar el total
        # número de lotes procesados, calcule el promedio promedio
        # pérdida, suavizarla y actualizar la lista de pérdidas con el
        # valor suavizado
        l = logs["loss"]
        self.batchNum += 1
        self.avgLoss = (self.beta * self.avgLoss) + ((1 - self.beta) * l)
        smooth = self.avgLoss / (1 - (self.beta ** self.batchNum))
        self.losses.append(smooth)

        # calcular el valor del factor de parada de pérdida máxima
        stopLoss = self.stopFactor * self.bestLoss

        # verificar para ver si la pérdida ha crecido demasiado
        if self.batchNum > 1 and smooth > stopLoss:
            # stop returning and return from the method
            self.model.network.stop_training = True
            return

        # verificar para ver si la mejor pérdida debe actualizarse
        if self.batchNum == 1 or smooth < self.bestLoss:
            self.bestLoss = smooth

        # aumentar la tasa de aprendizaje
        lr *= self.lrMult
        K.set_value(self.model.network.optimizer.lr, lr)

    def find(self, dataset, startLR, endLR, epochs=None,
             stepsPerEpoch=None, batchSize=32, sampleSize=2048):
        # restablecer nuestras variables específicas de clase
        self.reset()

        # si no se proporciona un número de épocas de entrenamiento, calcule el
        # épocas de entrenamiento basadas en un tamaño de muestra predeterminado
        if epochs is None:
            epochs = int(np.ceil(sampleSize / float(stepsPerEpoch)))

        # calcule el número total de actualizaciones por lotes que tomarán
        # lugar mientras intentamos encontrar un buen comienzo
        # tasa de aprendizaje
        numBatchUpdates = epochs * stepsPerEpoch

        # derivar el multiplicador de tasa de aprendizaje basado en el final
        # tasa de aprendizaje, tasa de aprendizaje inicial y número total de
        # actualizaciones por lotes
        self.lrMult = (endLR / startLR) ** (1.0 / numBatchUpdates)

        # construir una devolución de llamada que se llamará al final de cada
        # lote, lo que nos permite aumentar nuestra tasa de aprendizaje como entrenamiento
        # progresa
        callback = LambdaCallback(
            on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

        _ = self.model.fit(dataset=dataset,
                           batch_size=batchSize,
                           epochs=epochs,
                           callbacks=[callback],
                           lr=startLR)

    def plot_loss(self, name, skipBegin=10, skipEnd=1, title=""):
        # tomar la tasa de aprendizaje y los valores de pérdidas para graficar
        lrs = self.lrs[skipBegin:-skipEnd]
        losses = self.losses[skipBegin:-skipEnd]

        # trazar la tasa de aprendizaje frente a la pérdida
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Loss")

        # si el título no está vacío, agréguelo a la trama
        if title != "":
            plt.title(title)
        plt.savefig(SAVE_LR_PLOT + str(name) + '_lr.png')
