import os
import argparse
import glob
import warnings
from pprint import pprint
import shutil
import torch

from lib.utils import read_model_params, save_model_params, ensure_dir, add_suffix_to_path
# from lib.paintings_dataset import PaintingsDataset
from lib.loaders import ImageDataSet
from lib.lenet import LeNet
from lib.logger import Logger
from lib.trainer import Trainer

from lib import cropped
import utils

C = utils.get_config('./config.ini')

models_folder = 'models'
model_params_fname = 'model_params.json'
train_images_folder = os.path.join('Downloads', 'art_images', 'train', 'resized')

DATA_DIRECTORY = os.path.join('Downloads', 'art_images')
train_labels = os.path.join(DATA_DIRECTORY, 'mnist_m_train_labels.txt')


def train(identifier):
    models = glob.glob(os.path.join(models_folder, str(identifier) + '_created'))
    images_train = glob.glob(train_images_folder)

    for model_folder in models:
        new_model_folder_name = model_folder.replace('_created', '_training')
        shutil.move(model_folder, new_model_folder_name)

        model_params_path = os.path.join(new_model_folder_name, model_params_fname)
        print('train.py: training model', model_params_path, 'with hyperparams')

        # load model params.
        model_params = read_model_params(model_params_path)

        # print model and training parameters.
        pprint(model_params)

        # cuda flag
        using_cuda = model_params['cuda'] and torch.cuda.is_available()
        if using_cuda is True:
            print('train.py: Using ' + str(torch.cuda.get_device_name(0)))

        # Load primary training data
        # num_samples = 10 ** 5
        # streamer = ImageStreamer(images_train, )
        # dat_train = ImageDataSet()
        # dat_train = PaintingsDataset(model_params['data_train'], num_samples)
        data_train = cropped.Data(root=C.paths.root, info_csv=C.paths.train_csv, train=True)
        loader_train = torch.utils.data.DataLoader(data_train, batch_size=model_params['batch_size'], shuffle=True, num_workers=1)

        # Load secondary training data - used to evaluate training loss after every epoch
        # num_samples = 10 ** 4
        # dat_train2 = PaintingsDataset(model_params['data_train'], num_samples)
        data_train_eval = cropped.Data(root=C.paths.root, info_csv=C.paths.train_csv, train=True)
        loader_train_eval = torch.utils.data.DataLoader(data_train_eval, batch_size=model_params['batch_size'], shuffle=False, num_workers=1)

        # Load validation data - used to evaluate validation loss after every epoch
        # num_samples = 10 ** 4
        # dat_val = PaintingsDataset(model_params['data_val'], num_samples)
        data_val = cropped.Data(root=C.paths.root, info_csv=C.paths.train_csv, train=True)
        loader_val = torch.utils.data.DataLoader(data_val, batch_size=model_params['batch_size'], shuffle=False, num_workers=1)

        # create model
        model = LeNet(model_params['input_size'],
                      model_params['output_size'],

                      model_params['batch_norm'],

                      model_params['use_pooling'],
                      model_params['pooling_method'],

                      model_params['conv1_kernel_size'],
                      model_params['conv1_num_kernels'],
                      model_params['conv1_stride'],
                      model_params['conv1_dropout'],

                      model_params['pool1_kernel_size'],
                      model_params['pool1_stride'],

                      model_params['conv2_kernel_size'],
                      model_params['conv2_num_kernels'],
                      model_params['conv2_stride'],
                      model_params['conv2_dropout'],

                      model_params['pool2_kernel_size'],
                      model_params['pool2_stride'],

                      model_params['fcs_hidden_size'],
                      model_params['fcs_num_hidden_layers'],
                      model_params['fcs_dropout'])

        if using_cuda is True:
            model.cuda()

        # save initial weights
        if model_params['save_initial'] and model_params['save_dir']:
            suffix = '_initial'
            path = add_suffix_to_path(model_params['save_dir'], suffix)
            print('Saving model weights in : ' + path)
            ensure_dir(path)
            torch.save(model.state_dict(), os.path.join(path, 'model.dat'))
            save_model_params(os.path.join(path, 'model_params.json'), model_params)

        # loss
        if model_params['cost_function'] == 'cross_entropy':
            loss = torch.nn.NLLLoss()

        if using_cuda is True:
            loss.cuda()

        # optimizer
        if model_params['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=model_params['learning_rate'], weight_decay=model_params['weight_decay'])
        elif model_params['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=model_params['learning_rate'], momentum=model_params['momentum'], weight_decay=model_params['weight_decay'])
        else:
            raise ValueError('model_params[\'optimizer\'] must be either Adam or SGD. Got ' + model_params['optimizer'])

        logger = Logger()

        trainer = Trainer(model=model,
                          loss=loss,
                          optimizer=optimizer,
                          patience=model_params['patience'],
                          loader_train=loader_train,
                          loader_train_eval=loader_train_eval,
                          loader_val=loader_val,
                          cuda=using_cuda,
                          model_path = new_model_folder_name,
                          logger=logger)

        # run training
        trainer.train()

        os.rename(new_model_folder_name, new_model_folder_name.replace('_training', '_trained'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('identifier', help='Option to load model params from a file. Values in this file take precedence.')
    args = parser.parse_args()

    identifier = args.identifier
    train(identifier)


if __name__ == '__main__':
    main()
