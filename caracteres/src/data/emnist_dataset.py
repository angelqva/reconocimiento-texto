"""
Conjunto de datos EMNIST. Se descarga desde el sitio web de NIST y se guarda como archivo .npz si aún no está presente.
"""

from __future__ import absolute_import
from src import util
from src.data.dataset import Dataset
from __future__ import division
from __future__ import print_function

from urllib.request import urlretrieve
import zipfile
from scipy.io import loadmat
import os
import shutil
import h5py
import errno
import numpy as np
import json
from pathlib import Path
from tensorflow.keras.utils import to_categorical
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

root_folder = Path(__file__).resolve().parents[2]/'data'
raw_folder = root_folder/'raw'
processed_folder = root_folder/'processed'
url = 'https://s3-us-west-2.amazonaws.com/fsdl-public-assets/matlab.zip'
filename = raw_folder/'matlab.zip'
ESSENTIALS_FILENAME = raw_folder/'emnist_essentials.json'
SAMPLE_TO_BALANCE = True


class EMNIST(Dataset):
    """
    "El conjunto de datos EMNIST es un conjunto de dígitos de caracteres escritos a mano derivados de la base de datos especial NIST 19
    y convertido a un formato de imagen de 28x28 píxeles y una estructura de conjunto de datos que coincide directamente con el conjunto de datos MNIST".
    De https://www.nist.gov/itl/iad/image-group/emnist-dataset
    La división de datos que usaremos es
    EMNIST ByClass: 814.255 caracteres. 62 clases desequilibradas.
    """

    def __init__(self):

        if os.path.exists(ESSENTIALS_FILENAME):
            with open(ESSENTIALS_FILENAME) as f:
                essentials = json.load(f)
            self.mapping = dict(essentials['mapping'])
            self.num_classes = len(self.mapping)
            self.input_shape = essentials['input_shape']
            self.output_shape = (self.num_classes,)

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def download(self):
        """Descargar conjunto de datos EMNIST"""

        try:
            os.makedirs(raw_folder)
            os.makedirs(processed_folder)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        print('[INFO] Descargando conjunto de datos sin procesar...')
        util.download_url(url, filename)
        print('[INFO] Descarga completa..')

        print('[INFO] Descomprimiendo conjunto de datos sin procesar...')
        zip_file = zipfile.ZipFile(filename, 'r')
        zip_file.extract('matlab/emnist-byclass.mat', processed_folder)
        print('[INFO] Descompresión completada')

        print('[INFO] Cargando datos de prueba y entrenamiento desde un archivo .mat...')
        data = loadmat(processed_folder/'matlab/emnist-byclass.mat')
        x_train = data['dataset']['train'][0, 0]['images'][0,
                                                           0].reshape(-1, 28, 28).swapaxes(1, 2)
        y_train = data['dataset']['train'][0, 0]['labels'][0, 0]
        x_test = data['dataset']['test'][0, 0]['images'][0,
                                                         0].reshape(-1, 28, 28).swapaxes(1, 2)
        y_test = data['dataset']['test'][0, 0]['labels'][0, 0]

        if SAMPLE_TO_BALANCE:
            print('[INFO] Equilibrar clases para reducir la cantidad de datos...')
            x_train, y_train = _sample_to_balance(x_train, y_train)
            x_test, y_test = _sample_to_balance(x_test, y_test)

        print('[INFO] Guardando en HDF5 en un formato comprimido...')
        PROCESSED_DATA_DIRNAME = processed_folder
        PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME/'byclass.h5'

        with h5py.File(PROCESSED_DATA_FILENAME, 'w') as f:
            f.create_dataset('x_train', data=x_train,
                             dtype='u1', compression='lzf')
            f.create_dataset('y_train', data=y_train,
                             dtype='u1', compression='lzf')
            f.create_dataset('x_test', data=x_test,
                             dtype='u1', compression='lzf')
            f.create_dataset('y_test', data=y_test,
                             dtype='u1', compression='lzf')

        print('[INFO] Guardando parámetros esenciales del conjunto de datos...')
        mapping = {int(k): chr(v) for k, v in data['dataset']['mapping'][0, 0]}
        essentials = {'mapping': list(
            mapping.items()), 'input_shape': list(x_train.shape[1:])}
        self.mapping = mapping
        self.num_classes = len(self.mapping)
        self.input_shape = essentials['input_shape']
        self.output_shape = (self.num_classes,)

        with open(ESSENTIALS_FILENAME, 'w') as f:
            json.dump(essentials, f)

        print('[INFO] Limpiar...')
        os.remove(filename)
        shutil.rmtree(processed_folder/'matlab')

    def load_data(self):
        """ Cargar conjunto de datos EMNIST"""

        PROCESSED_DATA_DIRNAME = processed_folder
        PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME/'byclass.h5'

        if not os.path.exists(PROCESSED_DATA_FILENAME):
            self.download()
        with h5py.File(PROCESSED_DATA_FILENAME, 'r') as f:
            self.x_train = f['x_train'][:]
            self.y_train = f['y_train'][:]
            self.x_test = f['x_test'][:]
            self.y_test = f['y_test'][:]

        self.y_train = to_categorical(
            self.y_train, num_classes=self.num_classes)
        self.y_test = to_categorical(self.y_test, num_classes=self.num_classes)

        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    def __repr__(self):
        return (
            'EMNIST Dataset\n'
            f'Num classes: {self.num_classes}\n'
            f'Mapping: {self.mapping}\n'
            f'Input shape: {self.input_shape}\n'
        )


def _sample_to_balance(x, y):
    """Debido a que el conjunto de datos no está equilibrado, tomamos como máximo el número medio de instancias por clase."""
    np.random.seed(42)
    num_to_sample = int(np.bincount(y.flatten()).mean())
    all_sampled_inds = []
    for label in np.unique(y.flatten()):
        inds = np.where(y == label)[0]
        sampled_inds = np.unique(np.random.choice(inds, num_to_sample))
        all_sampled_inds.append(sampled_inds)
    ind = np.concatenate(all_sampled_inds)
    x_sampled = x[ind]
    y_sampled = y[ind]
    return x_sampled, y_sampled


def main():
    """Cargue el conjunto de datos EMNIST e imprima INFO."""

    dataset = EMNIST()
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    print(dataset)
    print('Entrenamiento shape:', x_train.shape, y_train.shape)
    print('Prueba shape:', x_test.shape, y_test.shape)


if __name__ == '__main__':
    main()
