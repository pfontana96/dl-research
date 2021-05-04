import argparse
from pathlib import Path
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

from .core import YOLO_v2, yolo_loss, LearningRateScheduler, BatchGenerator
from .utils import load_config, parse_annotations, normalize
from .graphic_tools import plot_image

parser = argparse.ArgumentParser(prog = "YOLO v2")
sb_parser_action = parser.add_subparsers()
sb_parser_action.add_argument("-t", "--train", help="Train model with specified data path.\n\
                                Path should containg a folder named /JPEGImages (containing\
                                 *.jpg files) and another named /Annotations (containing *.xml\
                                  annotations files).", nargs=1)
sb_parser_action.add_argument("-p", "--predict", help="Predict image (path to .jpg file).", nargs=1)

parser.add_argument("-c", "--config", help="Path to config file (defaults to config.yaml in \
                                            current directory).", nargs=1)
parser.add_argument("-w", "--weights", help="Load weights from file (defaults to weights.h5 in \
                                             current, random initialization if file does not \
                                             exist).")
parser.add_argument("-s", "--show", help="Flag to plot figure")

args = parser.parse_args()

def get_model(config, weights_file=None):
    callbacks = []
    # Define learning rate or Schedule
    if not 'learning_rate' in config.keys():
        lr = 0.5e-4 # Default values
    else:
        if type(config['learning_rate']) == np.ndarray:
            callbacks.append(LearningRateScheduler(config['learning_rate']))
            # Get first learning rate
            lr = config['learning_rate'][np.argmin(config['learning_rate'][:,0], 1)        

    # Define optimizer
    if not 'optimizer' in config.keys():
        # Default optimizer is adam
        optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    else:
        if not all (k in foo for foo in ("adam", "sgd", "rmsprop")):
            raise ValueError("Not supported optimizer in config file. Currently supported\
                              options:\n'adam' 'sgd' 'rmsprop'")
        
        if config['optimizer'] == 'adam':
            optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        
        if config['optimizer'] == 'sgd':
            momentum = config['momentum'] if 'momentum' in config.keys() else 0.0
            optimizer = SGD(learning_rate=lr, momentum=momentum)

        if config['optimizer'] == 'rmsprop':
            momentum = config['momentum'] if 'momentum' in config.keys() else 0.0
            rho = config['rho'] if 'rho' in config.keys() else 0.0
            optimizer = RMSprop(learning_rate=lr, momentum=momentum, rho=rho)
    
    # Define loss multipliers
    lambda_coord = config['lambda_coord'] if 'lambda_coord' in config.keys() else 5
    lambda_noobj = config['lambda_noobj'] if 'lambda_noobj' in config.keys() else 0.5

    # Create model
    model = YOLO_v2(config)
    model.build((None, model.img_shape[1], model.img_shape[0], 3))
    model.summary()

    # Load weights
    if weights_file:
        model.load_weights(str(weights_file))
        print('Weights loaded from: {}'.format(str(weights_file)))

    # Compile model
    model.compile(loss=yolo_loss(lambda_coord=lambda_coord, lambda_noobj=lambda_noobj), 
                  optimizer=optimizer)

    return model

def main():
    if not args.config:
        config_path = Path(__file__).resolve().parent.joinpath('config.yaml')
    else:
        config_path = Path(args.config).resolve()
        if not config_path.exists():
            raise ValueError('Specified config file not found: {}'.format(args.config))

    config = load_config(config_path)
    
    weights_file = None
    if args.weights_file:
        weights_file = Path(args.weights_file).resolve()
        if (not weights_file.exists()) or (weights_file.suffix != '.h5'):
            weights_file = None 
            print('Weights file not found or not compatible type (.h5)!\n{}'.format(
                   str(weights_file)))           

    model = get_model(config, weights_file)

if __name__ == '__main__':
    main()