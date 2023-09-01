import matplotlib
matplotlib.get_backend()
import matplotlib.pyplot as plt
import pickle
import os
from datasets import load_dataset, Dataset
import tensorflow as tf
from load_data import load_gtsrb, load_ood_data
from ReAD import statistic_of_neural_value, get_neural_value
from ReAD import encode_abstraction, concatenate_data_between_layers
from ReAD import k_means, statistic_distance
from ReAD import sk_auc
from global_config import num_of_labels, swin_config
import global_config


if __name__ == "__main__":

    if os.path.exists(f'./data/mnist/selection_ratio_ood.pkl') and \
            os.path.exists(f'./data/fashion_mnist/selection_ratio_ood.pkl') and \
            os.path.exists(f'./data/cifar10/selection_ratio_ood.pkl') and \
            os.path.exists(f'./data/gtsrb/selection_ratio_ood.pkl'):
        print('Research is finished!')

        file1 = open(f'./data/mnist/selection_ratio_ood.pkl', 'rb')
        file2 = open(f'./data/fashion_mnist/selection_ratio_ood.pkl', 'rb')
        file3 = open(f'./data/cifar10/selection_ratio_ood.pkl', 'rb')
        file4 = open(f'./data/gtsrb/selection_ratio_ood.pkl', 'rb')
        mnist_performance = pickle.load(file1)
        fmnist_performance = pickle.load(file2)
        cifar10_performance = pickle.load(file3)
        gtsrb_performance = pickle.load(file4)
        ood = ['FMNIST', 'MNIST','Omniglot', 'TinyImageNet', 'LSUN', 'iSUN', 'UniformNoise', 'GuassianNoise']

        x = [0.01*a for a in range(1, 101)]
        for o in ood:
            if mnist_performance[o]:
                plt.plot(x, mnist_performance[o], label=f'MNIST vs {o}')
            if fmnist_performance[o]:
                plt.plot(x, fmnist_performance[o], label=f'FMNIST vs {o}')
            if cifar10_performance[o]:
                plt.plot(x, cifar10_performance[o], label=f'Cifar10 vs {o}')
            if gtsrb_performance[o]:
                plt.plot(x, gtsrb_performance[o], label=f'GTSRB vs {o}')
        plt.legend(loc='lower right')
        plt.tick_params(labelsize=15)
        plt.ylim(0.7, 1)
        ax = plt.gca()
        y_major_locator = plt.MultipleLocator(0.02)
        ax.yaxis.set_major_locator(y_major_locator)
        ax.vlines(0.2, 0.7, 1.0, linestyles='--', colors='darkgray')
        plt.show()
        exit()

    rate = [x for x in range(1, 101)]

    id_dataset = 'mnist'
    # id_dataset = 'fashion_mnist'
    # id_dataset = 'cifar10'
    # id_dataset = 'gtsrb'


    if id_dataset == 'mnist':
        row_names = ('image', 'label')
        finetuned_checkpoint = "./models/swin-finetuned-mnist/best"
        dataset = load_dataset("./data/mnist/")

    elif id_dataset == 'fashion_mnist':
        row_names = ('image', 'label')
        finetuned_checkpoint = "./models/swin-finetuned-fashion_mnist/best"
        dataset = load_dataset("./data/fashion_mnist/")

    elif id_dataset == 'cifar10':
        row_names = ('img', 'label')
        finetuned_checkpoint = "./models/swin-finetuned-cifar10/best"
        dataset = load_dataset("./data/cifar10/")

    elif id_dataset == 'gtsrb':
        row_names = ('image', 'label')
        finetuned_checkpoint = "./models/swin-finetuned-gtsrb/best"
        dataset = load_gtsrb()

    else:
        print('dataset is not exist.')
        exit()

    if os.path.exists(f'./data/{id_dataset}/selection_ratio_ood.pkl'):
        print(f'Research on Dataset {id_dataset} is done!')
        exit()
    ood_performance = {'FMNIST': [], 'MNIST': [], 'Omniglot': [], 'TinyImageNet': [], 'LSUN': [], 'iSUN': [],
                       'UniformNoise_28': [], 'GuassianNoise_28': [], 'UniformNoise_32': [], 'GuassianNoise_32': [],
                       'UniformNoise_48': [], 'GuassianNoise_48': []}
    detector_path = f'./data/{id_dataset}/detector/selection_ratio_research/'
    if not os.path.exists(detector_path):
        if not os.path.exists(f'./data/{id_dataset}/detector/'):
            os.mkdir(f'./data/{id_dataset}/detector/')
        os.mkdir(detector_path)
    num_of_category = num_of_labels[id_dataset]

    rename_train_data = Dataset.from_dict({'image': dataset['train'][row_names[0]],
                                           'label': dataset['train'][row_names[1]]})
    rename_test_data = Dataset.from_dict({'image': dataset['test'][row_names[0]],
                                          'label': dataset['test'][row_names[1]]})

    print('\nGet neural value of train dataset:')
    train_picture_neural_value = get_neural_value(id_dataset=id_dataset, dataset=rename_train_data,
                                                  checkpoint=finetuned_checkpoint, is_ood=False)
    print('\nStatistic of train data neural value:')
    train_nv_statistic = statistic_of_neural_value(id_dataset=id_dataset,
                                                   neural_value=train_picture_neural_value)
    print('finished!')
    file1 = open(detector_path + f'train_nv_statistic.pkl', 'wb')
    pickle.dump(train_nv_statistic, file1)
    file1.close()

    neural_value_dict = dict()
    print('\nGet neural value of test dataset:')
    test_picture_neural_value = get_neural_value(id_dataset=id_dataset, dataset=rename_test_data,
                                                 checkpoint=finetuned_checkpoint, is_ood=False)
    neural_value_dict['test'] = test_picture_neural_value
    OOD_dataset = swin_config[id_dataset]['ood_settings']
    for ood in OOD_dataset:
        print(f'In-Distribution Data: {id_dataset}, Out-of-Distribution Data: {ood}.')
        ood_data = load_ood_data(ood_dataset=ood)
        print('\nGet neural value of ood dataset...')
        ood_picture_neural_value = get_neural_value(id_dataset=id_dataset, dataset=ood_data,
                                                    checkpoint=finetuned_checkpoint,
                                                    is_ood=True)
        neural_value_dict[ood] = ood_picture_neural_value

    for r in rate:
        print(f'-----------------------------------------------------------------------{r*0.01}-----------------------')
        global_config.selective_rate = r*0.01
        k_means_centers = None
        if os.path.exists(detector_path + f'train_nv_statistic.pkl') and \
                os.path.exists(detector_path + f'k_means_centers_{r}.pkl'):
            print("\nDetector is existed!")
            file1 = open(detector_path + f'train_nv_statistic.pkl', 'rb')
            file2 = open(detector_path + f'k_means_centers_{r}.pkl', 'rb')
            train_nv_statistic = pickle.load(file1)
            k_means_centers = pickle.load(file2)

        else:
            # ********************** Train Detector **************************** #
            print('\n********************** Train Detector ****************************')

            print('\nEncoding combination abstraction of train dataset:')
            train_ReAD_abstractions = encode_abstraction(id_dataset=id_dataset,
                                                         neural_value=train_picture_neural_value,
                                                         train_dataset_statistic=train_nv_statistic)
            train_ReAD_concatenated = \
                concatenate_data_between_layers(id_dataset=id_dataset, data_dict=train_ReAD_abstractions)

            print('\nK-Means Clustering of Combination Abstraction on train data:')
            k_means_centers = k_means(data=train_ReAD_concatenated,
                                      category_number=num_of_labels[id_dataset])
            file2 = open(detector_path + f'k_means_centers_{r}.pkl', 'wb')
            pickle.dump(k_means_centers, file2)
            file2.close()

        # ********************** Evaluate Detector **************************** #
        print('\n********************** Evaluate OOD Detection ****************************')
        test_picture_neural_value = neural_value_dict['test']
        print('\nEncoding ReAD abstraction of test dataset:')
        test_ReAD_abstractions = encode_abstraction(id_dataset=id_dataset, neural_value=test_picture_neural_value,
                                                    train_dataset_statistic=train_nv_statistic)
        test_ReAD_concatenated = concatenate_data_between_layers(id_dataset=id_dataset,
                                                                 data_dict=test_ReAD_abstractions)
        print('\nCalculate distance between abstractions and cluster centers ...')
        test_distance = statistic_distance(id_dataset=id_dataset, abstractions=test_ReAD_concatenated,
                                           cluster_centers=k_means_centers)

        OOD_dataset = swin_config[id_dataset]['ood_settings']
        for ood in OOD_dataset:
            print('\n************Evaluating*************')
            print(f'In-Distribution Data: {id_dataset}, Out-of-Distribution Data: {ood}.')
            ood_picture_neural_value = neural_value_dict[ood]
            print('\nEncoding ReAD abstraction of ood dataset...')
            ood_ReAD_abstractions = encode_abstraction(id_dataset=id_dataset, neural_value=ood_picture_neural_value,
                                                       train_dataset_statistic=train_nv_statistic)
            ood_ReAD_concatenated = concatenate_data_between_layers(id_dataset=id_dataset,
                                                                    data_dict=ood_ReAD_abstractions)
            print('\nCalculate distance between abstractions and cluster centers ...')
            ood_distance = statistic_distance(id_dataset=id_dataset, abstractions=ood_ReAD_concatenated,
                                              cluster_centers=k_means_centers)

            auc = sk_auc(test_distance, ood_distance, num_of_labels[id_dataset])

            print('\nPerformance of Detector:')
            print('AUROC: {:.6f}'.format(auc))
            print('*************************************\n')

            ood_performance[ood].append(auc)

    ood_performance_file = open(f'./data/{id_dataset}/selection_ratio_ood.pkl', 'wb')
    pickle.dump(ood_performance, ood_performance_file)
    ood_performance_file.close()






