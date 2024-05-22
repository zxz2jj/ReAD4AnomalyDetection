import pickle
import os
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from datasets import load_dataset
from sklearn import metrics
from sklearn.metrics import auc, roc_curve

from load_data import load_gtsrb, load_ood_data
from ReAD import (get_neural_value, statistic_of_neural_value, encode_abstraction, concatenate_data_between_layers,
                  statistic_distance, sk_auc)
from global_config import num_of_labels, swin_config


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
    # id_dataset = 'fashion_mnist'
    # id_dataset = 'cifar10'
    # id_dataset = 'svhn'
    # id_dataset = 'gtsrb'
    id_dataset = 'cifar100'

    if id_dataset == 'mnist':
        row_names = ('image', 'label')
        finetuned_checkpoint = "./models/swin-finetuned-mnist/best"
        dataset = load_dataset("./data/mnist/mnist/", cache_dir='./dataset/')
        detector_path = './data/mnist/detector/'

    elif id_dataset == 'fashion_mnist':
        row_names = ('image', 'label')
        finetuned_checkpoint = "./models/swin-finetuned-fashion_mnist/best"
        dataset = load_dataset("./data/fashion_mnist/fashion_mnist/", cache_dir='./dataset/')
        detector_path = './data/fashion_mnist/detector/'

    elif id_dataset == 'cifar10':
        row_names = ('img', 'label')
        finetuned_checkpoint = "./models/swin-finetuned-cifar10/best"
        dataset = load_dataset("./data/cifar10/cifar10/", cache_dir='./dataset/')
        detector_path = './data/cifar10/detector/'

    elif id_dataset == 'svhn':
        row_names = ('image', 'label')
        finetuned_checkpoint = "./models/swin-finetuned-svhn/best"
        dataset = load_dataset('./data/svhn/svhn/', 'cropped_digits', cache_dir='./dataset/')
        del dataset['extra']
        detector_path = './data/svhn/detector/'

    elif id_dataset == 'gtsrb':
        row_names = ('image', 'label')
        finetuned_checkpoint = "./models/swin-finetuned-gtsrb/best"
        dataset = load_gtsrb()
        detector_path = './data/gtsrb/detector/'

    elif id_dataset == 'cifar100':
        row_names = ('img', 'fine_label')
        finetuned_checkpoint = "./models/swin-finetuned-cifar100/best"
        dataset = load_dataset("./data/cifar100/cifar100/", cache_dir='./dataset/')
        detector_path = './data/cifar100/detector/'

    else:
        print('dataset is not exist.')
        exit()

    train_data = dataset['train']
    test_data = dataset['test']
    if row_names[0] != 'image':
        train_data = train_data.rename_column(row_names[0], 'image')
        test_data = test_data.rename_column(row_names[0], 'image')
    if row_names[1] != 'label':
        train_data = train_data.rename_column(row_names[1], 'label')
        test_data = test_data.rename_column(row_names[1], 'label')

    if not os.path.exists(f'./data/{id_dataset}/cluster_research/train_ReAD_ID_{id_dataset}.pkl'):
        print('\n********************** Train Detector ****************************')
        print('\nGet neural value of train dataset:')
        train_picture_neural_value = get_neural_value(id_dataset=id_dataset, dataset=train_data,
                                                      checkpoint=finetuned_checkpoint, is_ood=False)
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

    print('\nClustering of Combination Abstraction on train data:')
    clustering_start_time = time.time()
    # cluster_centers, metrics_dict = k_means(data=train_ReAD_concatenated, category_number=num_of_labels[id_dataset])
    # cluster_centers, metrics_dict = k_means_plusplus(data=train_ReAD_concatenated, category_number=num_of_labels[id_dataset])
    cluster_centers, metrics_dict = k_medoids(data=train_ReAD_concatenated, category_number=num_of_labels[id_dataset])
    clustering_end_time = time.time()
    total_time = clustering_end_time - clustering_start_time
    print(f"\nClustering Process: total time-{total_time}s")
    print(f"\nClustering Process: total time-{total_time}s \n {metrics_dict}", file=file3)

    if not os.path.exists(f'./data/{id_dataset}/cluster_research/test_ReAD_ID_{id_dataset}.pkl'):
        print('\n********************** Evaluate OOD Detection ****************************')
        print('\nGet neural value of test dataset:')
        test_picture_neural_value = get_neural_value(id_dataset=id_dataset, dataset=test_data,
                                                     checkpoint=finetuned_checkpoint, is_ood=False)
        print('\nEncoding ReAD abstraction of test dataset:')
        test_ReAD_abstractions = encode_abstraction(id_dataset=id_dataset, neural_value=test_picture_neural_value,
                                                    train_dataset_statistic=train_nv_statistic)
        test_ReAD_concatenated = concatenate_data_between_layers(id_dataset=id_dataset, data_dict=test_ReAD_abstractions)

        file = open(f'./data/{id_dataset}/cluster_research/test_ReAD_ID_{id_dataset}.pkl', 'wb')
        pickle.dump(test_ReAD_concatenated, file)
        file.close()
    else:
        file = open(f'./data/{id_dataset}/cluster_research/test_ReAD_ID_{id_dataset}.pkl', 'rb')
        test_ReAD_concatenated = pickle.load(file)

    print('\nCalculate distance between abstractions and cluster centers ...')
    test_distance = statistic_distance(id_dataset=id_dataset, abstractions=test_ReAD_concatenated,
                                       cluster_centers=cluster_centers)

    OOD_dataset = swin_config[id_dataset]['ood_settings']
    for ood in OOD_dataset:
        print('\n************Evaluating*************')
        print(f'In-Distribution Data: {id_dataset}, Out-of-Distribution Data: {ood}.')
        print(f'In-Distribution Data: {id_dataset}, Out-of-Distribution Data: {ood}.', file=file3)
        if not os.path.exists(f'./data/{id_dataset}/cluster_research/test_ReAD_OOD_{ood}.pkl'):
            ood_data = load_ood_data(ood_dataset=ood)
            print('\nGet neural value of ood dataset...')
            ood_picture_neural_value = get_neural_value(id_dataset=id_dataset, dataset=ood_data,
                                                        checkpoint=finetuned_checkpoint,
                                                        is_ood=True)
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

    file3.close()


