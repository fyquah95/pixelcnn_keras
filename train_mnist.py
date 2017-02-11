import os
import sys
import argparse
import configparser
from datetime import datetime
import pytz
from distutils.util import strtobool
import numpy as np

from core.layers import PixelCNN
from core.utils import Utils

from keras.utils.visualize_util import plot


def train(argv=None):
    ''' train conditional Gated PixelCNN model 
    Usage:
    	python train_mnist.py -c sample_train.cfg        : training example using configfile
    	python train_mnist.py --option1 hoge ...         : train with command-line options
        python train_mnist.py -c test.cfg --opt1 hoge... : overwrite config options with command-line options
    '''

    ### parsing arguments from command-line or config-file ###
    if argv is None:
        argv = sys.argv

    conf_parser = argparse.ArgumentParser(
        description=__doc__, # printed with -h/--help
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
        )
    conf_parser.add_argument("-c", "--conf_file",
                        help="Specify config file", metavar="FILE_PATH")
    args, remaining_argv = conf_parser.parse_known_args()

    defaults = {}

    if args.conf_file:
        config = configparser.SafeConfigParser()
        config.read([args.conf_file])
        defaults.update(dict(config.items("General")))

    parser = argparse.ArgumentParser(
        parents=[conf_parser]
        )
    parser.set_defaults(**defaults)
    parser.add_argument("--nb_epoch", help="Number of epochs [Required]", type=int, metavar="INT")
    parser.add_argument("--batch_size", help="Minibatch size", type=int, metavar="INT")
    parser.add_argument("--conditional", help="model the conditional distribution p(x|h) (default:False)", type=str, metavar="BOOL")
    parser.add_argument("--nb_pixelcnn_layers", help="Number of PixelCNN Layers (exept last two ReLu layers)", metavar="INT")
    parser.add_argument("--nb_filters", help="Number of filters for each layer", metavar="INT")
    parser.add_argument("--filter_size_1st", help="Filter size for the first layer. (default: (7,7))", metavar="INT,INT")
    parser.add_argument("--filter_size", help="Filter size for the subsequent layers. (default: (3,3))", metavar="INT,INT")
    parser.add_argument("--optimizer", help="SGD optimizer (default: adadelta)", type=str, metavar="OPT_NAME")
    parser.add_argument("--es_patience", help="Patience parameter for EarlyStopping", type=int, metavar="INT")
    parser.add_argument("--save_root", help="Root directory which trained files are saved (default: /tmp/pixelcnn)", type=str, metavar="DIR_PATH")
    parser.add_argument("--timezone", help="Trained files are saved in save_root/YYYYMMDDHHMMSS/ (default: Asia/Tokyo)", type=str, metavar="REGION_NAME")
    parser.add_argument("--save_best_only", help="The latest best model will not be overwritten (default: False)", type=str, metavar="BOOL")
    parser.add_argument("--plot_model", help="If True, plot a Keras model (using graphviz)", type=str, metavar="BOOL")

    args = parser.parse_args(remaining_argv)

    conditional = strtobool(args.conditional) if args.conditional else False

    ### mnist image size ###
    input_size = (28, 28)
    nb_classes = 10

    utils = Utils()

    ### load mnist dataset ###
    ### if conditional == True, X_train = [X_train, h_train] ###
    (X_train, Y_train), (X_validation, Y_validation) = utils.load_mnist_dataset(conditional)

    ### build PixelCNN model ###
    model_params = {}
    model_params['input_size'] = input_size
    model_params['nb_channels'] = 1
    model_params['conditional'] = conditional
    if conditional:
        model_params['latent_dim'] = nb_classes
    if args.nb_pixelcnn_layers:
        model_params['nb_pixelcnn_layers'] = int(args.nb_pixelcnn_layers)
    if args.nb_filters:
        model_params['nb_filters'] = int(args.nb_filters)
    if args.filter_size_1st:
        model_params['filter_size_1st'] = tuple(map(int, args.filter_size_1st.split(',')))
    if args.filter_size:
        model_params['filter_size'] = tuple(map(int, args.filter_size.split(',')))
    if args.optimizer:
        model_params['optimizer'] = args.optimizer
    if args.es_patience:
        model_params['es_patience'] = int(args.patience)
    if args.save_best_only:
        model_params['save_best_only'] = strtobool(args.save_best_only)

    save_root = args.save_root if args.save_root else '/tmp/pixelcnn_mnist'
    timezone = args.timezone if args.timezone else 'Asia/Tokyo'
    current_datetime = datetime.now(pytz.timezone(timezone)).strftime('%Y%m%d_%H%M%S')
    save_root = os.path.join(save_root, current_datetime)
    model_params['save_root'] = save_root

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    try:
        nb_epoch = int(args.nb_epoch)
        batch_size = int(args.batch_size)
    except:
        sys.exit("Error: {--nb_epoch, --batch_size} must be specified.")


    pixelcnn = PixelCNN(**model_params)
    pixelcnn.build_model()
    pixelcnn.model.summary()

    pixelcnn.print_train_parameters(save_root)
    pixelcnn.export_train_parameters(save_root)
    with open(os.path.join(save_root, 'parameters.txt'), 'a') as txt_file:
        txt_file.write('########## other options ##########\n')
        txt_file.write('nb_epoch\t: %s\n' % nb_epoch)
        txt_file.write('batch_size\t: %s\n' % batch_size)
        txt_file.write('\n')
    plot_model = strtobool(args.plot_model) if args.plot_model else True
    if plot_model:
        plot(pixelcnn.model, to_file=os.path.join(save_root, 'pixelcnn_mnist_model.png'))


    train_params = {}
    train_params['x'] = X_train
    train_params['y'] = Y_train
    train_params['validation_data'] = (X_validation, Y_validation)
    train_params['nb_epoch'] = nb_epoch
    train_params['batch_size'] = batch_size
    train_params['shuffle'] = True

    pixelcnn.fit(**train_params)

    return (0)


if __name__ == '__main__':
    sys.exit(train())
