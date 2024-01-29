import os
import tensorflow
from load_data import load_mnist, load_fmnist, load_cifar10, load_gtsrb, load_svhn, load_cifar100
from model_lenet import LeNetModel
from model_vgg import VGGModel
from model_resnet18 import ResNet18Model
from global_config import num_of_labels


def train_model(id_dataset, model_save_path):
    if id_dataset == 'mnist':
        x_train, y_train, x_test, y_test = load_mnist()
        model = LeNetModel(train_class_number=num_of_labels[id_dataset], train_data=x_train, train_label=y_train,
                           test_data=x_test, test_label=y_test, model_save_path=model_save_path)
    elif id_dataset == 'fmnist':
        x_train, y_train, x_test, y_test = load_fmnist()
        model = LeNetModel(train_class_number=num_of_labels[id_dataset], train_data=x_train, train_label=y_train,
                           test_data=x_test, test_label=y_test, model_save_path=model_save_path)
    elif id_dataset == 'cifar10':
        x_train, y_train, x_test, y_test = load_cifar10()
        model = VGGModel(train_class_number=num_of_labels[id_dataset], train_data=x_train, train_label=y_train,
                         test_data=x_test, test_label=y_test, model_save_path=model_save_path)
    elif id_dataset == 'gtsrb':
        x_train, y_train, x_test, y_test = load_gtsrb()
        model = VGGModel(train_class_number=num_of_labels[id_dataset], train_data=x_train, train_label=y_train,
                         test_data=x_test, test_label=y_test, model_save_path=model_save_path)
    elif id_dataset == 'svhn':
        x_train, y_train, x_test, y_test = load_svhn()
        model = ResNet18Model(train_class_number=num_of_labels[id_dataset], train_data=x_train, train_label=y_train,
                              test_data=x_test, test_label=y_test, model_save_path=model_save_path)
    elif id_dataset == 'cifar100':
        x_train, y_train, x_test, y_test = load_cifar100()
        model = ResNet18Model(train_class_number=num_of_labels[id_dataset], train_data=x_train, train_label=y_train,
                              test_data=x_test, test_label=y_test, model_save_path=model_save_path)
    else:
        model = None

    if os.path.exists(model_save_path+'tf_model.h5'):
        print('{} is existed!'.format(model_save_path+'tf_model.h5'))
        model.show_model()
    else:
        model.train(is_used_data_augmentation=True)


if __name__ == "__main__":

    # print(tensorflow.config.list_physical_devices('GPU'))
    # dataset = 'mnist'
    # save_path = './models/lenet_mnist/'

    # dataset = 'fmnist'
    # save_path = './models/lenet_fmnist/'

    # dataset = 'cifar10'
    # save_path = './models/vgg19_cifar10/'

    # dataset = 'gtsrb'
    # save_path = './models/vgg19_gtsrb/'

    dataset = 'cifar100'
    save_path = './models/resnet18_cifar100/'

    # dataset = 'svhn'
    # save_path = './models/resnet18_svhn/'

    train_model(id_dataset=dataset, model_save_path=save_path)
