import pickle
import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from train_models import train_model
from load_data import load_fmnist, load_cifar10, load_svhn, load_clean_adv_data
from ReAD import classify_id_pictures, get_neural_value, statistic_of_neural_value, encode_abstraction, \
    concatenate_data_between_layers, k_means, statistic_distance, auroc, sk_auc
from ReAD import t_sne_visualization
from global_config import num_of_labels, selective_rate, cnn_config


if __name__ == "__main__":

    # clean_dataset = 'fmnist'
    # model_path = './models/lenet_fmnist/'
    # detector_path = './data/fmnist/detector/'
    # x_train, y_train, x_test, y_test = load_fmnist()

    # clean_dataset = 'cifar10'
    # model_path = './models/vgg19_cifar10/'
    # detector_path = './data/cifar10/detector/'
    # x_train, y_train, x_test, y_test = load_cifar10()

    clean_dataset = 'svhn'
    model_path = './models/resnet18_svhn/'
    detector_path = './data/svhn/detector/'
    x_train, y_train, x_test, y_test = load_svhn()

    # train models. If model is existed, it will show details of the model
    show_model = False
    if show_model:
        train_model(id_dataset=clean_dataset, model_save_path=model_path)

    train_nv_statistic = None
    k_means_centers = None
    distance_train_statistic = None
    if os.path.exists(detector_path + 'train_nv_statistic.pkl') and \
            os.path.exists(detector_path + 'k_means_centers.pkl') and \
            os.path.exists(detector_path + 'distance_of_ReAD_for_train_data.pkl'):
        print("\nDetector is existed!")
        file1 = open(detector_path + 'train_nv_statistic.pkl', 'rb')
        file2 = open(detector_path + 'k_means_centers.pkl', 'rb')
        file3 = open(detector_path + 'distance_of_ReAD_for_train_data.pkl', 'rb')
        train_nv_statistic = pickle.load(file1)
        k_means_centers = pickle.load(file2)
        distance_train_statistic = pickle.load(file3)

    else:
        # ********************** Train Detector **************************** #
        print('\n********************** Train Detector ****************************')
        train_picture_classified = classify_id_pictures(id_dataset=clean_dataset, dataset=x_train,
                                                        labels=tf.argmax(y_train, axis=1), model_path=model_path)

        print('\nGet neural value of train dataset:')
        train_picture_neural_value = get_neural_value(id_dataset=clean_dataset, model_path=model_path,
                                                      pictures_classified=train_picture_classified)

        print('\nStatistic of train data neural value:')
        train_nv_statistic = statistic_of_neural_value(id_dataset=clean_dataset,
                                                       neural_value=train_picture_neural_value)
        print('finished!')

        file1 = open(detector_path + 'train_nv_statistic.pkl', 'wb')
        pickle.dump(train_nv_statistic, file1)
        file1.close()

        print('\nEncoding combination abstraction of train dataset:')
        train_ReAD_abstractions = encode_abstraction(id_dataset=clean_dataset, neural_value=train_picture_neural_value,
                                                     train_dataset_statistic=train_nv_statistic)
        train_ReAD_concatenated = \
            concatenate_data_between_layers(id_dataset=clean_dataset, data_dict=train_ReAD_abstractions)

        # visualization will need some minutes. If you want to show the visualization, please run the following codes.
        show_visualization = False
        if show_visualization:
            print('\nShow t-SNE visualization results on ReAD.')
            t_sne_visualization(data=train_ReAD_concatenated,
                                category_number=num_of_labels[clean_dataset])

        print('\nK-Means Clustering of Combination Abstraction on train data:')
        k_means_centers = k_means(data=train_ReAD_concatenated,
                                  category_number=num_of_labels[clean_dataset])
        file2 = open(detector_path + 'k_means_centers.pkl', 'wb')
        pickle.dump(k_means_centers, file2)
        file2.close()

        print('\nCalculate distance between abstractions and cluster centers ...')
        distance_train_statistic = statistic_distance(id_dataset=clean_dataset, abstractions=train_ReAD_concatenated,
                                                      cluster_centers=k_means_centers)
        file3 = open(detector_path + 'distance_of_ReAD_for_train_data.pkl', 'wb')
        pickle.dump(distance_train_statistic, file3)
        file3.close()

    # ********************** Evaluate adversarial Detection **************************** #
    print('\n********************** Evaluate Adversarial Detection ****************************')
    adversarial_attacks = ['ba']
    # adversarial_attacks = cnn_config[clean_dataset]['adversarial_settings']
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
        print(f'\nEncoding: (selective rate - {selective_rate}):')
        print("CLEAN dataset ...")
        clean_data_ReAD = encode_abstraction(id_dataset=clean_dataset, neural_value=clean_data_neural_value,
                                             train_dataset_statistic=train_nv_statistic)
        print('ADVERSARIAL dataset ...')
        adv_data_ReAD = encode_abstraction(id_dataset=clean_dataset, neural_value=adv_data_neural_value,
                                           train_dataset_statistic=train_nv_statistic)
        clean_ReAD_concatenated = \
            concatenate_data_between_layers(id_dataset=clean_dataset, data_dict=clean_data_ReAD)
        adv_ReAD_concatenated = \
            concatenate_data_between_layers(id_dataset=clean_dataset, data_dict=adv_data_ReAD)

        print('\nCalculate distance between abstractions and cluster centers ...')
        print('CLEAN dataset ...')
        clean_distance = statistic_distance(id_dataset=clean_dataset, abstractions=clean_ReAD_concatenated,
                                            cluster_centers=k_means_centers)
        print('ADVERSARIAL dataset ...')
        adv_distance = statistic_distance(id_dataset=clean_dataset, abstractions=adv_ReAD_concatenated,
                                          cluster_centers=k_means_centers)

        # auc = auroc(distance_train_statistic, clean_distance, adv_distance, num_of_labels[clean_dataset])
        auc = sk_auc(clean_distance, adv_distance, num_of_labels[clean_dataset])

        print('\nPerformance of Detector:')
        print('AUROC: {:.6f}'.format(auc))
        print('*************************************\n')

        # distance_list = []
        # for i in range(10):
        #     distance_list.append(clean_distance[i]['correct_pictures'])
        #     distance_list.append(adv_distance[i]['wrong_pictures'])
        # data = {'T0': distance_list[0]}
        # data = pd.DataFrame(data, columns=['T0'])
        # data['F0'] = pd.Series(distance_list[1])
        # data['T1'] = pd.Series(distance_list[2])
        # data['F1'] = pd.Series(distance_list[3])
        # data['T2'] = pd.Series(distance_list[4])
        # data['F2'] = pd.Series(distance_list[5])
        # data['T3'] = pd.Series(distance_list[6])
        # data['F3'] = pd.Series(distance_list[7])
        # data['T4'] = pd.Series(distance_list[8])
        # data['F4'] = pd.Series(distance_list[9])
        # data['T5'] = pd.Series(distance_list[10])
        # data['F5'] = pd.Series(distance_list[11])
        # data['T6'] = pd.Series(distance_list[12])
        # data['F6'] = pd.Series(distance_list[13])
        # data['T7'] = pd.Series(distance_list[14])
        # data['F7'] = pd.Series(distance_list[15])
        # data['T8'] = pd.Series(distance_list[16])
        # data['F8'] = pd.Series(distance_list[17])
        # data['T9'] = pd.Series(distance_list[18])
        # data['F9'] = pd.Series(distance_list[19])
        # data = pd.DataFrame(data)
        # sns.boxplot(data=data)
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        # plt.show()

