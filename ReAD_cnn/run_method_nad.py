import pickle
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from train_models import train_model
from load_data import load_mnist, load_fmnist, load_cifar10, load_cifar100, load_gtsrb, load_svhn, load_clean_adv_data, load_ood_data
from ReAD import classify_id_pictures, get_neural_value, statistic_of_neural_value, encode_abstraction, \
    concatenate_data_between_layers, k_means, statistic_distance, auroc, sk_auc
from ReAD import t_sne_visualization
from global_config import num_of_labels, selective_rate, cnn_config


if __name__ == "__main__":

    print('\n********************** Evaluate OOD Detection ****************************')
    # id_dataset = 'mnist'
    # model_path = './models/lenet_mnist/'
    # detector_path = './data/mnist/detector/'
    # x_train, y_train, x_test, y_test = load_mnist()

    # id_dataset = 'fmnist'
    # model_path = './models/lenet_fmnist/'
    # detector_path = './data/fmnist/detector/'
    # x_train, y_train, x_test, y_test = load_fmnist()
    #
    # id_dataset = 'svhn'
    # model_path = './models/resnet18_svhn/'
    # detector_path = './data/svhn/detector/'
    # x_train, y_train, x_test, y_test = load_svhn()
    #
    id_dataset = 'cifar100'
    model_path = './models/resnet18_cifar100/'
    detector_path = './data/cifar100/detector/'
    x_train, y_train, x_test, y_test = load_cifar100()
    #
    # id_dataset = 'cifar10'
    # model_path = './models/vgg19_cifar10/'
    # detector_path = './data/cifar10/detector/'
    # x_train, y_train, x_test, y_test = load_cifar10()
    #
    # id_dataset = 'gtsrb'
    # model_path = './models/vgg19_gtsrb/'
    # detector_path = './data/gtsrb/detector/'
    # x_train, y_train, x_test, y_test = load_gtsrb()

    print('\n********************** Train Detector ****************************')
    train_picture_classified = classify_id_pictures(id_dataset=id_dataset, dataset=x_train,
                                                    labels=tf.argmax(y_train, axis=1), model_path=model_path)

    print('\nGet neural value of train dataset:')
    train_picture_neural_value = get_neural_value(id_dataset=id_dataset, model_path=model_path,
                                                  pictures_classified=train_picture_classified)

    train_NAD_concatenated = \
        concatenate_data_between_layers(id_dataset=id_dataset, data_dict=train_picture_neural_value)

    k_means_centers = k_means(data=train_NAD_concatenated,
                              category_number=num_of_labels[id_dataset])

    # ********************** Evaluate OOD Detection **************************** #
    test_picture_classified = classify_id_pictures(id_dataset=id_dataset, dataset=x_test,
                                                   labels=tf.argmax(y_test, axis=1), model_path=model_path)
    print('\nGet neural value of test dataset:')
    test_picture_neural_value = get_neural_value(id_dataset=id_dataset, model_path=model_path,
                                                 pictures_classified=test_picture_classified)
    test_NAD_concatenated = concatenate_data_between_layers(id_dataset=id_dataset, data_dict=test_picture_neural_value)
    print('\nCalculate distance between abstractions and cluster centers ...')
    test_distance = statistic_distance(id_dataset=id_dataset, abstractions=test_NAD_concatenated,
                                       cluster_centers=k_means_centers)

    OOD_dataset = cnn_config[id_dataset]['ood_settings']
    for ood in OOD_dataset:
        print('\n************Evaluating*************')
        print(f'In-Distribution Data: {id_dataset}, Out-of-Distribution Data: {ood}.')
        ood_data, number_of_ood = load_ood_data(ood_dataset=ood, id_model_path=model_path,
                                                num_of_categories=num_of_labels[id_dataset])

        print('\nGet neural value of ood dataset...')
        ood_picture_neural_value = get_neural_value(id_dataset=id_dataset, model_path=model_path,
                                                    pictures_classified=ood_data)
        ood_NAD_concatenated = concatenate_data_between_layers(id_dataset=id_dataset, data_dict=ood_picture_neural_value)
        print('\nCalculate distance between abstractions and cluster centers ...')
        ood_distance = statistic_distance(id_dataset=id_dataset, abstractions=ood_NAD_concatenated,
                                          cluster_centers=k_means_centers)
        auc = sk_auc(id_dataset, test_distance, ood_distance)
        print('*************************************\n')

    print('\n********************** Evaluate Adversarial Detection ****************************')
    # clean_dataset = 'fmnist'
    # model_path = './models/lenet_fmnist/'
    # detector_path = './data/fmnist/detector/'
    # x_train, y_train, x_test, y_test = load_fmnist()

    # clean_dataset = 'svhn'
    # model_path = './models/resnet18_svhn/'
    # detector_path = './data/svhn/detector/'
    # x_train, y_train, x_test, y_test = load_svhn()

    clean_dataset = 'cifar10'
    model_path = './models/vgg19_cifar10/'
    detector_path = './data/cifar10/detector/'
    x_train, y_train, x_test, y_test = load_cifar10()

    print('\n********************** Train Detector ****************************')
    train_picture_classified = classify_id_pictures(id_dataset=clean_dataset, dataset=x_train,
                                                    labels=tf.argmax(y_train, axis=1), model_path=model_path)

    print('\nGet neural value of train dataset:')
    train_picture_neural_value = get_neural_value(id_dataset=clean_dataset, model_path=model_path,
                                                  pictures_classified=train_picture_classified)
    train_NAD_concatenated = \
        concatenate_data_between_layers(id_dataset=clean_dataset, data_dict=train_picture_neural_value)

    print('\nK-Means Clustering of Combination Abstraction on train data:')
    k_means_centers = k_means(data=train_NAD_concatenated,
                              category_number=num_of_labels[clean_dataset])

    adversarial_attacks = cnn_config[clean_dataset]['adversarial_settings']
    for attack in adversarial_attacks:
        print(f'\nATTACK: {attack}')
        clean_data_classified, adv_data_classified, num_of_test = \
            load_clean_adv_data(id_dataset=clean_dataset, attack=attack, num_of_categories=num_of_labels[clean_dataset])

        print('\nGet neural value:')
        print('CLEAN dataset ...')
        clean_data_neural_value = get_neural_value(id_dataset=clean_dataset, model_path=model_path,
                                                   pictures_classified=clean_data_classified)
        print('ADVERSARIAL dataset ...')
        adv_data_neural_value = get_neural_value(id_dataset=clean_dataset, model_path=model_path,
                                                 pictures_classified=adv_data_classified)
        clean_NAD_concatenated = \
            concatenate_data_between_layers(id_dataset=clean_dataset, data_dict=clean_data_neural_value)
        adv_NAD_concatenated = \
            concatenate_data_between_layers(id_dataset=clean_dataset, data_dict=adv_data_neural_value)

        print('\nCalculate distance between abstractions and cluster centers ...')
        print('CLEAN dataset ...')
        clean_distance = statistic_distance(id_dataset=clean_dataset, abstractions=clean_NAD_concatenated,
                                            cluster_centers=k_means_centers)
        print('ADVERSARIAL dataset ...')
        adv_distance = statistic_distance(id_dataset=clean_dataset, abstractions=adv_NAD_concatenated,
                                          cluster_centers=k_means_centers)

        sk_auc(clean_dataset, clean_distance, adv_distance)
        print('*************************************\n')

