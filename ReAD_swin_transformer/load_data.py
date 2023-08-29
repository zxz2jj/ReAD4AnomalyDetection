from datasets import Dataset, DatasetDict
import os
from PIL import Image
import glob
import pandas as pd
import numpy as np
import re


def load_data(dataset):
    if dataset == 'omniglot':
        data = load_omniglot()
    elif dataset == 'tiny_imagenet':
        data = load_tiny_imagenet()
    elif dataset == 'isun':
        data = load_isun()
    elif dataset == 'lsun':
        data = load_lsun()
    elif dataset == 'gtsrb':
        data = load_gtsrb()
    elif dataset == 'uniform_noise':
        data = load_uniform_noise()
    elif dataset == 'gaussian_noise':
        data = load_gaussian_noise()
    else:
        data = None

    return data


def load_omniglot():
    data_dir = './data/public/omniglot/images_evaluation'
    data = []
    for alphabet in os.listdir(data_dir):
        alphabet_path = os.path.join(data_dir, alphabet)
        for character in os.listdir(alphabet_path):
            character_path = os.path.join(alphabet_path, character)
            for image in os.listdir(character_path):
                img_path = os.path.join(character_path, image)
                print(f'\r{img_path}', end='')
                img = Image.open(img_path)
                data.append(img.copy())
                img.close()

    return Dataset.from_dict({'image': data})


def load_uniform_noise():
    data_dir = './data/public/uniform_noise'
    data = []
    for image in os.listdir(data_dir):
        img_path = os.path.join(data_dir, image)
        print(f'\r{img_path}', end='')
        img = Image.open(img_path)
        data.append(img.copy())
        img.close()

    return Dataset.from_dict({'image': data})


def load_gaussian_noise():
    data_dir = './data/public/gaussian_noise'
    data = []
    for image in os.listdir(data_dir):
        img_path = os.path.join(data_dir, image)
        print(f'\r{img_path}', end='')
        img = Image.open(img_path)
        data.append(img.copy())
        img.close()

    return Dataset.from_dict({'image': data})


def load_tiny_imagenet():
    data_dir = './data/public/tiny_imagenet/test/images/'
    data = []
    for image in os.listdir(data_dir):
        img_path = os.path.join(data_dir, image)
        print(f'\r{img_path}', end='')
        img = Image.open(img_path)
        data.append(img.copy())
        img.close()

    return Dataset.from_dict({'image': data})


def load_isun():
    data_dir = './data/public/isun'
    data = []
    for image in os.listdir(data_dir):
        img_path = os.path.join(data_dir, image)
        print(f'\r{img_path}', end='')
        img = Image.open(img_path)
        data.append(img.copy())
        img.close()

    return Dataset.from_dict({'image': data})


def load_lsun():
    data_dir = './data/public/lsun'
    data = []
    for image in os.listdir(data_dir):
        img_path = os.path.join(data_dir, image)
        print(f'\r{img_path}', end='')
        img = Image.open(img_path)
        data.append(img.copy())
        img.close()

    return Dataset.from_dict({'image': data})


def load_gtsrb():

    train_dir = './data/public/GTSRB/Final_Training/Images/'
    x_train = []
    y_train = []
    print('\nLoad GTSRB train dataset...')
    all_img_paths = glob.glob(os.path.join(train_dir, '*/*.ppm'))

    for img_path in all_img_paths:
        img_path = img_path.replace('\\', '/')
        print('\r', img_path, end='')
        img = Image.open(img_path)
        label = int(img_path.split('/')[-2])
        x_train.append(img.copy())
        y_train.append(label)
        img.close()
    print(f'\nTraining Dataset Length:{x_train.__len__()}')
    train_data = Dataset.from_dict({'image': x_train, 'label': y_train})

    test = pd.read_csv('./data/public/GTSRB/Final_Test/Images/GT-final_test.csv', sep=';')
    x_test = []
    y_test = []
    print('\nLoad GTSRB test dataset...')
    for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
        img_path = os.path.join('./data/public/GTSRB/Final_Test/Images/', file_name)
        print('\r', img_path, end='')
        img = Image.open(img_path)
        x_test.append(img.copy())
        y_test.append(class_id)
        img.close()

    print(f'\nTesting Dataset Length:{x_test.__len__()}')
    test_data = Dataset.from_dict({'image': x_test, 'label': y_test})

    data = DatasetDict({'train': train_data, 'test': test_data})

    return data


if __name__ == '__main__':

    d = load_gtsrb()

