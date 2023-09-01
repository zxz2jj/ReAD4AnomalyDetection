import matplotlib
matplotlib.get_backend()
import matplotlib.pyplot as plt
import pickle
import os
import tensorflow as tf
from load_data import load_mnist, load_fmnist, load_cifar10, load_svhn, load_gtsrb, \
    load_clean_adv_data, load_ood_data
from ReAD import statistic_of_neural_value, classify_id_pictures, get_neural_value
from ReAD import encode_abstraction, concatenate_data_between_layers
from ReAD import k_means, statistic_distance
from ReAD import sk_auc
from global_config import num_of_labels, cnn_config
import global_config


if __name__ == "__main__":
    # conduct selection ratio research on adversarial/ood detection tasks
    for_adversarial_detection = True
    for_ood_detection = False

    # for_adversarial_detection = False
    # for_ood_detection = True

    if for_adversarial_detection is True:
        if os.path.exists(f'./data/fmnist/selection_ratio_adversarial.pkl') and \
                os.path.exists(f'./data/cifar10/selection_ratio_adversarial.pkl') and \
                os.path.exists(f'./data/cifar10/selection_ratio_adversarial.pkl'):
            print('Research is finished!')

            file1 = open(f'./data/fmnist/selection_ratio_adversarial.pkl', 'rb')
            file2 = open(f'./data/cifar10/selection_ratio_adversarial.pkl', 'rb')
            file3 = open(f'./data/svhn/selection_ratio_adversarial.pkl', 'rb')
            fmnist_performance = pickle.load(file1)
            cifar10_performance = pickle.load(file2)
            svhn_performance = pickle.load(file3)
            attacks = {'ba': 'BA', 'fgsm': 'FGSM', 'jsma': 'JSMA', 'deepfool': 'DeepFool', 'newtonfool': 'NewtonFool',
                       'pixelattack': 'PA', 'hopskipjumpattack_l2': 'HSJA', 'squareattack_linf': 'SA', 'wassersteinattack': 'WA'}

            x = [0.01*a for a in range(1, 101)]
            for attack in attacks.keys():
                if fmnist_performance[attack]:
                    plt.plot(x, fmnist_performance[attack], label=f'FMNIST vs {attacks[attack]}')
                if cifar10_performance[attack]:
                    plt.plot(x, cifar10_performance[attack], label=f'Cifar10 vs {attacks[attack]}')
                if svhn_performance[attack]:
                    plt.plot(x, svhn_performance[attack], label=f'SVHN vs {attacks[attack]}')
            plt.legend(loc='lower right')
            plt.tick_params(labelsize=15)
            plt.ylim(0.7, 1)
            ax = plt.gca()
            y_major_locator = plt.MultipleLocator(0.05)
            ax.yaxis.set_major_locator(y_major_locator)
            ax.vlines(0.2, 0.7, 1.0, linestyles='--', colors='darkgray')
            plt.show()
            exit()

        rate = [x for x in range(1, 101)]

        clean_dataset = 'fmnist'
        adversarial_attacks = ['ba', 'hopskipjumpattack_l2', 'jsma', 'squareattack_linf', 'wassersteinattack']
        model_path = './models/lenet_fmnist/'
        x_train, y_train, x_test, y_test = load_fmnist()

        # clean_dataset = 'cifar10'
        # adversarial_attacks = ['ba', 'fgsm', 'jsma', 'newtonfool', 'pixelattack', 'wassersteinattack']
        # model_path = './model/vgg19_cifar10/'
        # x_train, y_train, x_test, y_test = load_cifar10()

        # clean_dataset = 'svhn'
        # adversarial_attacks = ['deepfool', 'fgsm', 'jsma', 'pixelattack', 'wassersteinattack']
        # model_path = './model/resnet18_svhn/'
        # x_train, y_train, x_test, y_test = load_svhn()

        if os.path.exists(f'./data/{clean_dataset}/selection_ratio_adversarial.pkl'):
            print(f'Research on Dataset {clean_dataset} is done!')
            exit()
        adversarial_performance = {'ba': [], 'fgsm': [], 'jsma': [], 'deepfool': [], 'newtonfool': [], 'pixelattack': [],
                                   'hopskipjumpattack_l2': [], 'squareattack_linf': [], 'wassersteinattack': []}
        detector_path = './data/{}/detector/selection_ratio_research/'.format(clean_dataset)
        if not os.path.exists(detector_path):
            os.mkdir(detector_path)
        num_of_category = num_of_labels[clean_dataset]

        train_picture_classified = classify_id_pictures(id_dataset=clean_dataset, dataset=x_train,
                                                        labels=tf.argmax(y_train, axis=1), model_path=model_path)

        print('\nGet neural value of train dataset:')
        train_picture_neural_value = get_neural_value(id_dataset=clean_dataset, model_path=model_path,
                                                      pictures_classified=train_picture_classified)

        print('\nStatistic of train data neural value:')
        train_nv_statistic = statistic_of_neural_value(id_dataset=clean_dataset,
                                                       neural_value=train_picture_neural_value)
        print('finished!')
        file1 = open(detector_path + f'train_nv_statistic.pkl', 'wb')
        pickle.dump(train_nv_statistic, file1)
        file1.close()

        neural_value_dict = {}
        for attack in adversarial_attacks:
            nv_of_each_attack = dict()
            print(f'\nATTACK: {attack}')
            clean_data_classified, adv_data_classified, num_of_test = \
                load_clean_adv_data(id_dataset=clean_dataset, attack=attack,
                                    num_of_categories=num_of_labels[clean_dataset])

            print('\nGet neural value:')
            print('CLEAN dataset ...')
            clean_data_neural_value = get_neural_value(id_dataset=clean_dataset, model_path=model_path,
                                                       pictures_classified=clean_data_classified)
            print('\nADVERSARIAL dataset ...')
            adv_data_neural_value = get_neural_value(id_dataset=clean_dataset, model_path=model_path,
                                                     pictures_classified=adv_data_classified)
            nv_of_each_attack['clean'] = clean_data_neural_value
            nv_of_each_attack['adversarial'] = adv_data_neural_value
            neural_value_dict[attack] = nv_of_each_attack

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
                train_ReAD_abstractions = encode_abstraction(id_dataset=clean_dataset,
                                                             neural_value=train_picture_neural_value,
                                                             train_dataset_statistic=train_nv_statistic)
                train_ReAD_concatenated = \
                    concatenate_data_between_layers(id_dataset=clean_dataset, data_dict=train_ReAD_abstractions)

                print('\nK-Means Clustering of Combination Abstraction on train data:')
                k_means_centers = k_means(data=train_ReAD_concatenated,
                                          category_number=num_of_labels[clean_dataset])
                file2 = open(detector_path + f'k_means_centers_{r}.pkl', 'wb')
                pickle.dump(k_means_centers, file2)
                file2.close()

            # ********************** Evaluate Detector **************************** #
            print('\n********************** Evaluate Detector ****************************')
            for attack in adversarial_attacks:
                print(f'\nATTACK: {attack}')
                clean_data_neural_value = neural_value_dict[attack]['clean']
                adv_data_neural_value = neural_value_dict[attack]['adversarial']
                print(f'\nEncoding: (selective rate - {global_config.selective_rate}):')
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

                auc = sk_auc(clean_distance, adv_distance, num_of_labels[clean_dataset])

                print('\nPerformance of Detector:')
                print('AUROC: {:.6f}'.format(auc))
                print('*************************************\n')

                adversarial_performance[attack].append(auc)

        adversarial_performance_file = open(f'./data/{clean_dataset}/selection_ratio_adversarial.pkl', 'wb')
        pickle.dump(adversarial_performance, adversarial_performance_file)
        adversarial_performance_file.close()

    # ###############################################################################################################
    if for_ood_detection is True:
        if os.path.exists(f'./data/mnist/selection_ratio_ood.pkl') and \
                os.path.exists(f'./data/fmnist/selection_ratio_ood.pkl') and \
                os.path.exists(f'./data/cifar10/selection_ratio_ood.pkl') and \
                os.path.exists(f'./data/gtsrb/selection_ratio_ood.pkl'):
            print('Research is finished!')

            file1 = open(f'./data/mnist/selection_ratio_ood.pkl', 'rb')
            file2 = open(f'./data/fmnist/selection_ratio_ood.pkl', 'rb')
            file3 = open(f'./data/cifar10/selection_ratio_ood.pkl', 'rb')
            file4 = open(f'./data/gtsrb/selection_ratio_ood.pkl', 'rb')
            mnist_performance = pickle.load(file1)
            fmnist_performance = pickle.load(file2)
            cifar10_performance = pickle.load(file3)
            gtsrb_performance = pickle.load(file4)
            ood = ['FMNIST', 'MNIST','Omniglot', 'TinyImageNet', 'LSUN', 'iSUN', 'UniformNoise_28', 'GuassianNoise_28',
                   'UniformNoise_32', 'GuassianNoise_32', 'UniformNoise_48', 'GuassianNoise_48']

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
        model_path = './models/lenet_mnist/'
        x_train, y_train, x_test, y_test = load_mnist()

        # id_dataset = 'fmnist'
        # model_path = './models/lenet_fmnist/'
        # x_train, y_train, x_test, y_test = load_fmnist()

        # id_dataset = 'cifar10'
        # model_path = './models/vgg19_cifar10/'
        # x_train, y_train, x_test, y_test = load_cifar10()

        # id_dataset = 'gtsrb'
        # model_path = './models/vgg19_gtsrb/'
        # x_train, y_train, x_test, y_test = load_gtsrb()

        if os.path.exists(f'./data/{id_dataset}/selection_ratio_ood.pkl'):
            print(f'Research on Dataset {id_dataset} is done!')
            exit()
        ood_performance = {'FMNIST': [], 'MNIST': [], 'Omniglot': [], 'TinyImageNet': [], 'LSUN': [], 'iSUN': [],
                           'UniformNoise_28': [], 'GuassianNoise_28': [], 'UniformNoise_32': [], 'GuassianNoise_32': [],
                           'UniformNoise_48': [], 'GuassianNoise_48': []}
        detector_path = './data/{}/detector/selection_ratio_research/'.format(id_dataset)
        if not os.path.exists(detector_path):
            os.mkdir(detector_path)
        num_of_category = num_of_labels[id_dataset]

        train_picture_classified = classify_id_pictures(id_dataset=id_dataset, dataset=x_train,
                                                        labels=tf.argmax(y_train, axis=1), model_path=model_path)

        print('\nGet neural value of train dataset:')
        train_picture_neural_value = get_neural_value(id_dataset=id_dataset, model_path=model_path,
                                                      pictures_classified=train_picture_classified)

        print('\nStatistic of train data neural value:')
        train_nv_statistic = statistic_of_neural_value(id_dataset=id_dataset,
                                                       neural_value=train_picture_neural_value)
        print('finished!')
        file1 = open(detector_path + f'train_nv_statistic.pkl', 'wb')
        pickle.dump(train_nv_statistic, file1)
        file1.close()

        neural_value_dict = dict()
        test_picture_classified = classify_id_pictures(id_dataset=id_dataset, dataset=x_test,
                                                       labels=tf.argmax(y_test, axis=1), model_path=model_path)
        print('\nGet neural value of test dataset:')
        test_picture_neural_value = get_neural_value(id_dataset=id_dataset, model_path=model_path,
                                                     pictures_classified=test_picture_classified)
        neural_value_dict['test'] = test_picture_neural_value
        OOD_dataset = cnn_config[id_dataset]['ood_settings']
        for ood in OOD_dataset:
            print(f'In-Distribution Data: {id_dataset}, Out-of-Distribution Data: {ood}.')
            ood_data, number_of_ood = load_ood_data(ood_dataset=ood, id_model_path=model_path,
                                                    num_of_categories=num_of_labels[id_dataset])

            print('\nGet neural value of ood dataset...')
            ood_picture_neural_value = get_neural_value(id_dataset=id_dataset, model_path=model_path,
                                                        pictures_classified=ood_data)
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

            OOD_dataset = cnn_config[id_dataset]['ood_settings']
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






