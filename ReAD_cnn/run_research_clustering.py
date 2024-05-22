import pickle
import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus, estimate_bandwidth
from sklearn_extra.cluster import KMedoids
from sklearn import metrics
import tensorflow as tf
import time
import os

from train_models import train_model
from load_data import load_mnist, load_fmnist, load_cifar10, load_cifar100, load_gtsrb, load_svhn, load_clean_adv_data, load_ood_data
from ReAD import classify_id_pictures, get_neural_value, statistic_of_neural_value, encode_abstraction, \
    concatenate_data_between_layers, statistic_distance, auroc, sk_auc
from ReAD import t_sne_visualization
from global_config import num_of_labels, selective_rate, cnn_config


def k_means(data, category_number):

    combination_abstraction = []
    number_of_categories = []
    for category in range(category_number):
        combination_abstraction.append(data[category]['correct_pictures'])
        number_of_categories.append(data[category]['correct_pictures'].shape[0])
    combination_abstraction = np.concatenate(combination_abstraction, axis=0)

    estimator = KMeans(n_clusters=category_number, init='random')
    estimator.fit(combination_abstraction)
    y_predict = estimator.predict(combination_abstraction)
    prediction_each_category = []
    y = [0 for _ in range(int(combination_abstraction.shape[0]))]
    end = 0
    for i in range(category_number):
        start = end
        end += int(number_of_categories[i])
        cluster = max(set(y_predict[start:end]), key=list(y_predict[start:end]).count)
        prediction_each_category.append(cluster)
        for j in range(start, end):
            y[j] = cluster
    centers = estimator.cluster_centers_
    centers_resorted = []
    for kind in range(category_number):
        centers_resorted.append(list(centers[prediction_each_category[kind]]))
    homo_score = metrics.homogeneity_score(y, y_predict)
    comp_score = metrics.completeness_score(y, y_predict)
    v_measure = metrics.v_measure_score(y, y_predict)
    print("K-Means Score(homo_score, comp_score, v_measure):", homo_score, comp_score, v_measure)

    return centers_resorted, {'homo_score': homo_score, 'comp_score': comp_score, 'v_measure': v_measure}


def k_means_plusplus(data, category_number):

    combination_abstraction = []
    number_of_categories = []
    for category in range(category_number):
        combination_abstraction.append(data[category]['correct_pictures'])
        number_of_categories.append(data[category]['correct_pictures'].shape[0])
    combination_abstraction = np.concatenate(combination_abstraction, axis=0)

    estimator = KMeans(n_clusters=category_number, init='k-means++')
    estimator.fit(combination_abstraction)
    y_predict = estimator.predict(combination_abstraction)
    prediction_each_category = []
    y = [0 for _ in range(int(combination_abstraction.shape[0]))]
    end = 0
    for i in range(category_number):
        start = end
        end += int(number_of_categories[i])
        cluster = max(set(y_predict[start:end]), key=list(y_predict[start:end]).count)
        prediction_each_category.append(cluster)
        for j in range(start, end):
            y[j] = cluster
    centers = estimator.cluster_centers_
    centers_resorted = []
    for kind in range(category_number):
        centers_resorted.append(list(centers[prediction_each_category[kind]]))
    homo_score = metrics.homogeneity_score(y, y_predict)
    comp_score = metrics.completeness_score(y, y_predict)
    v_measure = metrics.v_measure_score(y, y_predict)
    print("K-Means Score(homo_score, comp_score, v_measure):", homo_score, comp_score, v_measure)

    return centers_resorted, {'homo_score': homo_score, 'comp_score': comp_score, 'v_measure': v_measure}


def k_medoids(data, category_number):

    combination_abstraction = []
    number_of_categories = []
    for category in range(category_number):
        combination_abstraction.append(data[category]['correct_pictures'])
        number_of_categories.append(data[category]['correct_pictures'].shape[0])
    combination_abstraction = np.concatenate(combination_abstraction, axis=0)

    estimator = KMedoids(n_clusters=category_number, method='pam', init='k-medoids++')
    estimator.fit(combination_abstraction)
    y_predict = estimator.predict(combination_abstraction)
    prediction_each_category = []
    y = [0 for _ in range(int(combination_abstraction.shape[0]))]
    end = 0
    for i in range(category_number):
        start = end
        end += int(number_of_categories[i])
        cluster = max(set(y_predict[start:end]), key=list(y_predict[start:end]).count)
        prediction_each_category.append(cluster)
        for j in range(start, end):
            y[j] = cluster
    centers = estimator.cluster_centers_
    centers_resorted = []
    for kind in range(category_number):
        centers_resorted.append(list(centers[prediction_each_category[kind]]))
    homo_score = metrics.homogeneity_score(y, y_predict)
    comp_score = metrics.completeness_score(y, y_predict)
    v_measure = metrics.v_measure_score(y, y_predict)
    print("K-Means Score(homo_score, comp_score, v_measure):", homo_score, comp_score, v_measure)

    return centers_resorted, {'homo_score': homo_score, 'comp_score': comp_score, 'v_measure': v_measure}


if __name__ == "__main__":

    # id_dataset = 'mnist'
    # model_path = './models/lenet_mnist/'
    # detector_path = './data/mnist/detector/'
    # x_train, y_train, x_test, y_test = load_mnist()

    # id_dataset = 'fmnist'
    # model_path = './models/lenet_fmnist/'
    # detector_path = './data/fmnist/detector/'
    # x_train, y_train, x_test, y_test = load_fmnist()

    # id_dataset = 'svhn'
    # model_path = './models/resnet18_svhn/'
    # detector_path = './data/svhn/detector/'
    # x_train, y_train, x_test, y_test = load_svhn()

    id_dataset = 'cifar10'
    model_path = './models/vgg19_cifar10/'
    detector_path = './data/cifar10/detector/'
    x_train, y_train, x_test, y_test = load_cifar10()

    # id_dataset = 'cifar100'
    # model_path = './models/resnet18_cifar100/'
    # detector_path = './data/cifar100/detector/'
    # x_train, y_train, x_test, y_test = load_cifar100()

    # id_dataset = 'gtsrb'
    # model_path = './models/vgg19_gtsrb/'
    # detector_path = './data/gtsrb/detector/'
    # x_train, y_train, x_test, y_test = load_gtsrb()

    if not os.path.exists(f'./data/{id_dataset}/cluster_research/train_ReAD_ID_{id_dataset}.pkl'):
        print('\n********************** Train Detector ****************************')

        train_picture_classified = classify_id_pictures(id_dataset=id_dataset, dataset=x_train,
                                                        labels=tf.argmax(y_train, axis=1), model_path=model_path)

        print('\nGet neural value of train dataset:')
        train_picture_neural_value = get_neural_value(id_dataset=id_dataset, model_path=model_path,
                                                      pictures_classified=train_picture_classified)

        print('\nStatistic of train data neural value:')
        train_nv_statistic = statistic_of_neural_value(id_dataset=id_dataset,
                                                       neural_value=train_picture_neural_value)
        print('finished!')

        print('\nEncoding ReAD abstractions of train dataset:')
        train_ReAD_abstractions = encode_abstraction(id_dataset=id_dataset, neural_value=train_picture_neural_value,
                                                     train_dataset_statistic=train_nv_statistic)
        train_ReAD_concatenated = \
            concatenate_data_between_layers(id_dataset=id_dataset, data_dict=train_ReAD_abstractions)
        file = open(f'./data/{id_dataset}/cluster_research/train_ReAD_ID_{id_dataset}.pkl', 'wb')
        pickle.dump(train_ReAD_concatenated, file)
        file.close()
    else:
        file1 = open(f'./data/{id_dataset}/cluster_research/train_ReAD_ID_{id_dataset}.pkl', 'rb')
        train_ReAD_concatenated = pickle.load(file1)
        file2 = open(detector_path + 'train_nv_statistic.pkl', 'rb')
        train_nv_statistic = pickle.load(file2)

    file3 = open(f'./data/{id_dataset}/cluster_research/result.txt', 'w')
    clustering_start_time = time.time()
    print('\nClustering of Combination Abstraction on train data:')
    cluster_centers, metrics_dict = k_means(data=train_ReAD_concatenated,
                                            category_number=num_of_labels[id_dataset])
    # cluster_centers, metrics_dict = k_means_plusplus(data=train_ReAD_concatenated,
    #                                                  category_number=num_of_labels[id_dataset])
    # cluster_centers, metrics_dict = k_medoids(data=train_ReAD_concatenated,
    #                                           category_number=num_of_labels[id_dataset])

    clustering_end_time = time.time()
    total_time = clustering_end_time - clustering_start_time
    print(f"\nClustering Process: total time-{total_time}s")
    print(f"\nClustering Process: total time-{total_time}s \n {metrics_dict}", file=file3)

    if not os.path.exists(f'./data/{id_dataset}/cluster_research/test_ReAD_ID_{id_dataset}.pkl'):
        print('\n********************** Evaluate OOD Detection ****************************')
        test_picture_classified = classify_id_pictures(id_dataset=id_dataset, dataset=x_test,
                                                       labels=tf.argmax(y_test, axis=1), model_path=model_path)
        print('\nGet neural value of test dataset:')
        test_picture_neural_value = get_neural_value(id_dataset=id_dataset, model_path=model_path,
                                                     pictures_classified=test_picture_classified)
        print('\nEncoding ReAD abstraction of test dataset:')
        print("selective rate: {}".format(selective_rate))
        test_ReAD_abstractions = encode_abstraction(id_dataset=id_dataset, neural_value=test_picture_neural_value,
                                                    train_dataset_statistic=train_nv_statistic)
        test_ReAD_concatenated = concatenate_data_between_layers(id_dataset=id_dataset, data_dict=test_ReAD_abstractions)

        file = open(f'./data/{id_dataset}/cluster_research/test_ReAD_ID_{id_dataset}.pkl', 'wb')
        pickle.dump(test_ReAD_concatenated, file)
        file.close()
    else:
        file = open(f'./data/{id_dataset}/cluster_research/test_ReAD_ID_{id_dataset}.pkl', 'rb')
        test_ReAD_concatenated = pickle.load(file)

    test_distance = statistic_distance(id_dataset=id_dataset, abstractions=test_ReAD_concatenated,
                                       cluster_centers=cluster_centers)

    print('\nCalculate distance between abstractions and cluster centers ...')
    OOD_dataset = cnn_config[id_dataset]['ood_settings']

    for ood in OOD_dataset:
        print('\n************Evaluating*************')
        print(f'In-Distribution Data: {id_dataset}, Out-of-Distribution Data: {ood}.')
        print(f'In-Distribution Data: {id_dataset}, Out-of-Distribution Data: {ood}.', file=file3)
        if not os.path.exists(f'./data/{id_dataset}/cluster_research/test_ReAD_OOD_{ood}.pkl'):
            ood_data, number_of_ood = load_ood_data(ood_dataset=ood, id_model_path=model_path,
                                                    num_of_categories=num_of_labels[id_dataset])

            print('\nGet neural value of ood dataset...')
            ood_picture_neural_value = get_neural_value(id_dataset=id_dataset, model_path=model_path,
                                                        pictures_classified=ood_data)
            print('\nEncoding ReAD abstraction of ood dataset...')
            ood_ReAD_abstractions = encode_abstraction(id_dataset=id_dataset, neural_value=ood_picture_neural_value,
                                                       train_dataset_statistic=train_nv_statistic)
            ood_ReAD_concatenated = concatenate_data_between_layers(id_dataset=id_dataset, data_dict=ood_ReAD_abstractions)
            file = open(f'./data/{id_dataset}/cluster_research/test_ReAD_OOD_{ood}.pkl', 'wb')
            pickle.dump(ood_ReAD_concatenated, file)
            file.close()
        else:
            file = open(f'./data/{id_dataset}/cluster_research/test_ReAD_OOD_{ood}.pkl', 'rb')
            ood_ReAD_concatenated = pickle.load(file)
        print('\nCalculate distance between abstractions and cluster centers ...')
        ood_distance = statistic_distance(id_dataset=id_dataset, abstractions=ood_ReAD_concatenated,
                                          cluster_centers=cluster_centers)
        auc = sk_auc(id_dataset, test_distance, ood_distance)
        print(auc, file=file3)
        print('*************************************\n')
        print('*************************************\n', file=file3)

    adversarial_attacks = cnn_config[id_dataset]['adversarial_settings']
    for attack in adversarial_attacks:
        print(f'\nATTACK: {attack}')
        print(f'\nATTACK: {attack}', file=file3)
        if (not os.path.exists(f'./data/{id_dataset}/cluster_research/test_ReAD_Clean_{attack}.pkl') and
                not os.path.exists(f'./data/{id_dataset}/cluster_research/test_ReAD_Adv_{attack}.pkl')):
            clean_data_classified, adv_data_classified, num_of_test = \
                load_clean_adv_data(id_dataset=id_dataset, attack=attack,
                                    num_of_categories=num_of_labels[id_dataset])

            print('\nGet neural value:')
            print('CLEAN dataset ...')
            clean_data_neural_value = get_neural_value(id_dataset=id_dataset, model_path=model_path,
                                                       pictures_classified=clean_data_classified)
            print('ADVERSARIAL dataset ...')
            adv_data_neural_value = get_neural_value(id_dataset=id_dataset, model_path=model_path,
                                                     pictures_classified=adv_data_classified)

            print(f'\nEncoding: (selective rate - {selective_rate}):')
            print("CLEAN dataset ...")
            clean_data_ReAD = encode_abstraction(id_dataset=id_dataset, neural_value=clean_data_neural_value,
                                                 train_dataset_statistic=train_nv_statistic)
            print('ADVERSARIAL dataset ...')
            adv_data_ReAD = encode_abstraction(id_dataset=id_dataset, neural_value=adv_data_neural_value,
                                               train_dataset_statistic=train_nv_statistic)
            clean_ReAD_concatenated = \
                concatenate_data_between_layers(id_dataset=id_dataset, data_dict=clean_data_ReAD)
            adv_ReAD_concatenated = \
                concatenate_data_between_layers(id_dataset=id_dataset, data_dict=adv_data_ReAD)
            file1 = open(f'./data/{id_dataset}/cluster_research/test_ReAD_Clean_{attack}.pkl', 'wb')
            pickle.dump(clean_ReAD_concatenated, file1)
            file1.close()
            file2 = open(f'./data/{id_dataset}/cluster_research/test_ReAD_Adv_{attack}.pkl', 'wb')
            pickle.dump(adv_ReAD_concatenated, file2)
            file2.close()
        else:
            file1 = open(f'./data/{id_dataset}/cluster_research/test_ReAD_Clean_{attack}.pkl', 'rb')
            clean_ReAD_concatenated = pickle.load(file1)
            file2 = open(f'./data/{id_dataset}/cluster_research/test_ReAD_Adv_{attack}.pkl', 'rb')
            adv_ReAD_concatenated = pickle.load(file2)

        print('\nCalculate distance between abstractions and cluster centers ...')
        print('CLEAN dataset ...')
        clean_distance = statistic_distance(id_dataset=id_dataset, abstractions=clean_ReAD_concatenated,
                                            cluster_centers=cluster_centers)
        print('ADVERSARIAL dataset ...')
        adv_distance = statistic_distance(id_dataset=id_dataset, abstractions=adv_ReAD_concatenated,
                                          cluster_centers=cluster_centers)

        auc = sk_auc(id_dataset, clean_distance, adv_distance)

        print(auc, file=file3)
        print('*************************************\n')
        print('*************************************\n', file=file3)

    file3.close()

