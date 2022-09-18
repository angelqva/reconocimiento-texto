"""
Train model
"""

from __future__ import absolute_import
import argparse
from src.networks.custom import customCNN
from src.networks.resnet import resnet
from src.networks.lenet import lenet
from src.models.character_model import Character_Model
from src.data.emnist_dataset import EMNIST
from src.training.util import train_model
from sklearn.model_selection import train_test_split
from __future__ import division
from __future__ import print_function
from comet_ml import Experiment

from pathlib import Path
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save-model", type=int, default=False,
                        help="si el modelo debe guardarse o no")
    parser.add_argument("-w", "--weights", type=str, default=True,
                        help="si se deben guardar los pesos o no")
    parser.add_argument("-save_model", "--save_model", type=str, default=False,
                        help="si el modelo debe guardarse o no")
    parser.add_argument("-m", '--model', type=str, default="Character_Model",
                        help="que modelo usar")
    parser.add_argument("-n", '--network', type=str, default="lenet",
                        help="qué arquitectura de red usar")
    parser.add_argument("-d", '--dataset', type=str, default="EMNIST",
                        help="qué conjunto de datos usar")
    parser.add_argument("-e", '--epochs', type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("-b", '--batch_size', type=int, default=32,
                        help="Batch size")
    parser.add_argument("-find_lr", '--find_lr', type=bool, default=False,
                        help="Find lr")
    args = vars(parser.parse_args())

    return args


funcs = {'EMNIST': EMNIST, 'lenet': lenet, 'resnet': resnet,
         'customCNN': customCNN, 'Character_Model': Character_Model}


def train(args, use_comet: bool = True):

    data_cls = funcs[args['dataset']]
    model_cls = funcs[args['model']]
    network = funcs[args['network']]

    print('[INFO] Obteniendo dataset...')
    data = data_cls()
    (x_train, y_train), (x_test, y_test) = data.load_data()
    classes = data.mapping

    y_test_labels = [np.where(y_test[idx] == 1)[0][0]
                     for idx in range(len(y_test))]
    # distribute 90% test 10% val dataset with equal class distribution
    (x_test, x_valid, y_test, y_valid) = train_test_split(x_test, y_test, test_size=0.1,
                                                          stratify=y_test_labels, random_state=42)

    print('[INFO] Entrenamiento shape: ', x_train.shape, y_train.shape)
    print('[INFO] Validacion shape: ', x_valid.shape, y_valid.shape)
    print('[INFO] Prueba shape: ', x_test.shape, y_test.shape)

    print('[INFO] Configurando el modelo..')
    model = model_cls(network, data_cls)
    print(model)

    dataset = dict({
        'x_train': x_train,
        'y_train': y_train,
        'x_valid': x_valid,
        'y_valid': y_valid,
        'x_test': x_test,
        'y_test': y_test
    })

    if use_comet and args['find_lr'] == False:
        # crea un experimento con tu clave api
        experiment = Experiment(api_key='INSERT API KEY',
                                project_name='emnist',
                                auto_param_logging=False)

        print('[INFO] Empezar a entrenar...')
        # registrará métricas con el prefijo 'train_'
        with experiment.train():
            _ = train_model(
                model,
                dataset,
                batch_size=args['batch_size'],
                epochs=args['epochs'],
                name=args['network']
            )

        print('[INFO] Comenzando la prueba...')
        # registrará métricas con el prefijo 'test_'
        with experiment.test():
            loss, score = model.evaluate(dataset, args['batch_size'])
            print(f'[INFO] Evaluación de prueba: {score*100}')
            metrics = {
                'loss': loss,
                'accuracy': score
            }
            experiment.log_metrics(metrics)

        experiment.log_parameters(args)
        experiment.log_dataset_hash(x_train)
        experiment.end()

    elif use_comet and args['find_lr'] == True:

        _ = train_model(
            model,
            dataset,
            batch_size=args['batch_size'],
            epochs=args['epochs'],
            FIND_LR=args['find_lr'],
            name=args['network']
        )

    else:

        print('[INFO] Empezar a entrenar...')
        train_model(
            model,
            dataset,
            batch_size=args['batch_size'],
            epochs=args['epochs'],
            name=args['network']
        )
        print('[INFO] Comenzando la prueba...')
        loss, score = model.evaluate(dataset, args['batch_size'])
        print(f'[INFO] Evaluación de prueba: {score*100}')

    if args['weights']:
        model.save_weights()

    if args['save_model']:
        model.save_model()


def main():
    """Ejecutar experimento."""
    args = _parse_args()
    train(args)


if __name__ == '__main__':
    main()
