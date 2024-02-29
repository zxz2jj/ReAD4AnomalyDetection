from datasets import load_dataset, Dataset, DatasetDict
import os
from PIL import Image
import glob
import pandas as pd
from sklearn.utils import shuffle


def load_ood_data(ood_dataset):
    if ood_dataset == 'MNIST':
        data = load_dataset('./data/mnist/mnist/', cache_dir='./dataset/')
        data = Dataset.from_dict({'image': data['test']['image']})
    elif ood_dataset == 'FMNIST':
        data = load_dataset('./data/fashion_mnist/fashion_mnist/', cache_dir='./dataset/')
        data = Dataset.from_dict({'image': data['test']['image']})
    elif ood_dataset == 'Omniglot':
        data = load_omniglot()
    elif ood_dataset == 'TinyImageNet':
        data = load_tiny_imagenet()
    elif ood_dataset == 'iSUN':
        data = load_isun()
    elif ood_dataset == 'LSUN':
        data = load_lsun()
    elif ood_dataset == 'UniformNoise':
        data = load_uniform_noise()
    elif ood_dataset == 'GuassianNoise':
        data = load_gaussian_noise()
    else:
        print(f'No OOD dataset named {ood_dataset}!')
        data = None

    return data


def load_omniglot():
    data_dir = './data/public/Omniglot/images_evaluation'
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
    print('\n')

    return Dataset.from_dict({'image': data})


def load_uniform_noise():
    data_dir = './data/public/UniformNoise'
    data = []
    for image in os.listdir(data_dir):
        img_path = os.path.join(data_dir, image)
        print(f'\r{img_path}', end='')
        img = Image.open(img_path)
        data.append(img.copy())
        img.close()
    print('\n')

    return Dataset.from_dict({'image': data})


def load_gaussian_noise():
    data_dir = './data/public/GuassianNoise'
    data = []
    for image in os.listdir(data_dir):
        img_path = os.path.join(data_dir, image)
        print(f'\r{img_path}', end='')
        img = Image.open(img_path)
        data.append(img.copy())
        img.close()
    print('\n')

    return Dataset.from_dict({'image': data})


def load_tiny_imagenet():
    data_dir = './data/public/TinyImageNet/tiny-imagenet-200/test/images/'
    data = []
    for image in os.listdir(data_dir):
        img_path = os.path.join(data_dir, image)
        print(f'\r{img_path}', end='')
        img = Image.open(img_path)
        data.append(img.copy())
        img.close()
    print('\n')

    return Dataset.from_dict({'image': data})


def load_isun():
    data_dir = './data/public/iSUN'
    data = []
    for image in os.listdir(data_dir):
        img_path = os.path.join(data_dir, image)
        print(f'\r{img_path}', end='')
        img = Image.open(img_path)
        data.append(img.copy())
        img.close()
    print('\n')

    return Dataset.from_dict({'image': data})


def load_lsun():
    data_dir = './data/public/LSUN'
    data = []
    for image in os.listdir(data_dir):
        img_path = os.path.join(data_dir, image)
        print(f'\r{img_path}', end='')
        img = Image.open(img_path)
        data.append(img.copy())
        img.close()
    print('\n')

    return Dataset.from_dict({'image': data})


def load_gtsrb():
    if os.path.exists('./data/gtsrb/gtsrb/dataset_dict.json'):
        data = DatasetDict().load_from_disk('./data/gtsrb/gtsrb/')
        return data

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
    x_train, y_train = shuffle(x_train, y_train)
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
    x_test, y_test = shuffle(x_test, y_test)
    test_data = Dataset.from_dict({'image': x_test, 'label': y_test})

    data = DatasetDict({'train': train_data, 'test': test_data})
    data.save_to_disk('./data/gtsrb/gtsrb/')

    return data


if __name__ == '__main__':
    # dataset = load_gtsrb()
    dataset = load_uniform_noise()
    # dataset = load_dataset('./data/svhn/svhn/', 'cropped_digits', cache_dir='./dataset/')
    # del dataset['extra']
    # print('a')