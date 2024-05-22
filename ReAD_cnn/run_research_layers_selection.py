import matplotlib
matplotlib.get_backend()
import os
import pickle
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from load_data import load_mnist, load_fmnist, load_cifar10, load_svhn, load_clean_adv_data, load_ood_data
from ReAD import classify_id_pictures, k_means, auroc, sk_auc
from global_config import num_of_labels, selective_rate


def get_neural_value(id_dataset, layers, tf_model_path, pictures_classified):

    model = tf.keras.models.load_model(tf_model_path+'tf_model.h5')
    get_layers_output = tf.keras.backend.function(inputs=model.layers[0].input,
                                                  outputs=[model.layers[lay].output for lay in layers])
    correct_layers_output = {}
    wrong_layers_output = {}
    for category in range(num_of_labels[id_dataset]):
        print('\rcategory: {} / {}'.format(category + 1, num_of_labels[id_dataset]), end='')
        correct_pictures = pictures_classified[category]['correct_pictures']
        wrong_pictures = pictures_classified[category]['wrong_pictures']
        if correct_pictures.shape[0] != 0:
            correct_pictures_layers_output = list()
            batch_size = 128
            for b in range(0, len(correct_pictures), batch_size):
                batch = correct_pictures[b:b + batch_size]
                batch_output = get_layers_output([batch])
                correct_pictures_layers_output.append(batch_output)
            batch_output_concatenation = []
            for k in range(len(layers)):
                layer_output = []
                for layer_batch_output in correct_pictures_layers_output:
                    layer_output.append(layer_batch_output[k])
                batch_output_concatenation.append(np.concatenate(layer_output))
            correct_layers_output[category] = batch_output_concatenation
        else:
            correct_layers_output[category] = None
        if wrong_pictures.shape[0] != 0:
            wrong_pictures_layer_output = list()
            batch_size = 128
            for b in range(0, len(wrong_pictures), batch_size):
                batch = wrong_pictures[b:b + batch_size]
                batch_output = get_layers_output([batch])
                wrong_pictures_layer_output.append(batch_output)
            batch_output_concatenation = []
            for k in range(len(layers)):
                layer_output = []
                for layer_batch_output in wrong_pictures_layer_output:
                    layer_output.append(layer_batch_output[k])
                batch_output_concatenation.append(np.concatenate(layer_output))
            wrong_layers_output[category] = batch_output_concatenation
        else:
            wrong_layers_output[category] = None

    neural_value_dict = {}
    for k in range(len(layers)):
        neural_value_layer_dict = {}
        for category in range(num_of_labels[id_dataset]):
            neural_value_category = {}
            if correct_layers_output[category] is not None:
                if len(correct_layers_output[category][k].shape) > 2:
                    pictures_number = correct_layers_output[category][k].shape[0]
                    neural_value_category['correct_pictures'] = \
                        np.average(correct_layers_output[category][k], axis=-1).reshape([pictures_number, -1])
                else:
                    neural_value_category['correct_pictures'] = correct_layers_output[category][k]
                neural_value_category['correct_prediction'] = pictures_classified[category]['correct_prediction']
            else:
                neural_value_category['correct_pictures'] = np.array([])
                neural_value_category['correct_prediction'] = np.array([])

            if wrong_layers_output[category] is not None:
                if len(wrong_layers_output[category][k].shape) > 2:
                    pictures_number = wrong_layers_output[category][k].shape[0]
                    neural_value_category['wrong_pictures'] = \
                        np.average(wrong_layers_output[category][k], axis=-1).reshape([pictures_number, -1])
                else:
                    neural_value_category['wrong_pictures'] = wrong_layers_output[category][k]
                neural_value_category['wrong_prediction'] = pictures_classified[category]['wrong_prediction']
            else:
                neural_value_category['wrong_pictures'] = np.array([])
                neural_value_category['wrong_prediction'] = np.array([])

            neural_value_layer_dict[category] = neural_value_category
        neural_value_dict[layers[k]] = neural_value_layer_dict

    return neural_value_dict


def statistic_of_neural_value(id_dataset, layers, length_of_layers, neural_value):
    """
    计算某神经元在非第i类上输出均值
    non_class_l_average[l][k]表示第k个神经元在非第l类上的输出均值
    :param
    :return: non_class_l_average[] all_class_average
    """

    neuron_number_of_each_layer = length_of_layers
    number_of_classes = num_of_labels[id_dataset]

    neural_value_statistic = {}
    for i in range(len(layers)):
        neural_value_layer = neural_value[layers[i]]
        number_of_neurons = neuron_number_of_each_layer[i]
        not_class_c_average = [[] for _ in range(number_of_classes)]
        class_c_sum = [[] for _ in range(number_of_classes)]
        number_of_examples = []
        all_class_average = []

        for c in range(number_of_classes):
            correct_neural_value = neural_value_layer[c]['correct_pictures']
            number_of_examples.append(correct_neural_value.shape[0])
            train_correct_neural_value_transpose = np.transpose(correct_neural_value)
            for k in range(number_of_neurons):
                class_c_sum[c].append(np.sum(train_correct_neural_value_transpose[k]))

        for k in range(number_of_neurons):    # for each neuron
            output_sum = 0.0
            for c in range(number_of_classes):
                output_sum += class_c_sum[c][k]
            all_class_average.append(output_sum / np.sum(number_of_examples))
            for c in range(number_of_classes):
                not_class_c_average[c].append((output_sum - class_c_sum[c][k]) /
                                              (np.sum(number_of_examples) - number_of_examples[c]))

        neural_value_statistic_category = {'not_class_l_average': not_class_c_average,
                                           'all_class_average': all_class_average}
        neural_value_statistic[layers[i]] = neural_value_statistic_category

    return neural_value_statistic


def encode_by_selective(image_neural_value, label, encode_rate, number_of_neuron, not_class_l_average, all_class_average):
    """
    根据某selective值对某一图片的神经元进行编码
    :param image_neural_value: 图片的神经元输出
    :param label: 图片的预测标签CC
    :param number_of_neuron: 神经元个数
    :param encode_rate: 编码率
    :param not_class_l_average: 训练集非l类的图片的神经元输出均值
    :param all_class_average: 训练集所有类的图片的神经元输出均值
    :return: combination_code
    """

    selective = [0.0 for _ in range(number_of_neuron)]
    for r in range(number_of_neuron):
        if all_class_average[r] == 0:
            selective[r] = 0.0
        else:
            selective[r] = (image_neural_value[r] - not_class_l_average[label][r]) / all_class_average[r]

    dict_sel = {}
    for index in range(len(selective)):
        dict_sel[index] = selective[index]
    sort_by_sel = sorted(dict_sel.items(), key=lambda x: x[1])

    combination_code = [0 for _ in range(number_of_neuron)]
    for k in range(0, math.ceil(number_of_neuron * encode_rate / 2), 1):
        combination_code[sort_by_sel[k][0]] = -1
    for k in range(-1, -math.ceil(number_of_neuron * encode_rate / 2) - 1, -1):
        combination_code[sort_by_sel[k][0]] = 1

    return combination_code


def encode_abstraction(id_dataset, layers, length_of_layers, neural_value, train_dataset_statistic):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # --------------------------selective  = [ output - average(-l) ] / average(all)------------------------# #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    neuron_number_list = length_of_layers
    encode_layer_number = len(layers)
    print(f'selective rate: {selective_rate}')
    abstractions_dict = {}

    for k in range(encode_layer_number):
        print("\nEncoding in layer:{}.".format(layers[k]))
        non_class_l_avg = train_dataset_statistic[layers[k]]['not_class_l_average']
        all_class_avg = train_dataset_statistic[layers[k]]['all_class_average']

        combination_abstraction_each_category = {}
        for category in range(num_of_labels[id_dataset]):
            print('\rcategory: {} / {}'.format(category+1, num_of_labels[id_dataset]), end='')
            neural_value_category = neural_value[layers[k]][category]
            correct_abstraction_list = []
            wrong_abstraction_list = []
            if neural_value_category['correct_pictures'].shape[0] != 0:
                for image, label in \
                        zip(neural_value_category['correct_pictures'], neural_value_category['correct_prediction']):
                    combination = encode_by_selective(image, label, selective_rate, neuron_number_list[k], non_class_l_avg, all_class_avg)
                    correct_abstraction_list.append(combination)
            else:
                correct_abstraction_list = np.array([])

            if neural_value_category['wrong_pictures'].shape[0] != 0:
                for image, label in \
                        zip(neural_value_category['wrong_pictures'],  neural_value_category['wrong_prediction']):
                    combination = encode_by_selective(image, label, selective_rate, neuron_number_list[k], non_class_l_avg, all_class_avg)
                    wrong_abstraction_list.append(combination)
            else:
                wrong_abstraction_list = np.array([])

            combination_abstraction = {'correct_pictures': np.array(correct_abstraction_list),
                                       'correct_prediction': neural_value_category['correct_prediction'],
                                       'wrong_pictures': np.array(wrong_abstraction_list),
                                       'wrong_prediction': neural_value_category['wrong_prediction']}
            combination_abstraction_each_category[category] = combination_abstraction

        abstractions_dict[layers[k]] = combination_abstraction_each_category
    print('\n')

    return abstractions_dict


def concatenate_data_between_layers(id_dataset, layers, data_dict):
    abstraction_concatenated_dict = {}
    for category in range(num_of_labels[id_dataset]):
        correct_data = []
        if data_dict[layers[0]][category]['correct_pictures'].shape[0] != 0:
            for k in range(len(layers)):
                correct_data.append(data_dict[layers[k]][category]['correct_pictures'])
            correct_data = np.concatenate(correct_data, axis=1)
        else:
            correct_data = np.array([])

        wrong_data = []
        if np.array(data_dict[layers[0]][category]['wrong_pictures']).shape[0] != 0:
            for k in range(len(layers)):
                wrong_data.append(data_dict[layers[k]][category]['wrong_pictures'])
            wrong_data = np.concatenate(wrong_data, axis=1)
        else:
            wrong_data = np.array([])

        concatenate_data = {'correct_pictures': correct_data,
                            'correct_prediction': data_dict[layers[0]][category]['correct_prediction'],
                            'wrong_pictures': wrong_data,
                            'wrong_prediction': data_dict[layers[0]][category]['wrong_prediction']}

        abstraction_concatenated_dict[category] = concatenate_data

    return abstraction_concatenated_dict


def statistic_distance(id_dataset, length_of_layers, abstractions, cluster_centers):
    neuron_number_list = length_of_layers
    euclidean_distance = {}
    for category in range(num_of_labels[id_dataset]):
        abstraction_category = abstractions[category]
        distance_category = {}
        if abstraction_category['correct_pictures'] is not None:
            correct_distance = []
            abstraction_length = sum(neuron_number_list)
            if abstraction_category['correct_pictures'].shape[0] == 0:
                distance_category['correct_pictures'] = correct_distance
                distance_category['correct_prediction'] = abstraction_category['correct_prediction']
            else:
                for abstraction, prediction in zip(abstraction_category['correct_pictures'],
                                                   abstraction_category['correct_prediction']):
                    distance = 0.0
                    for k in range(abstraction_length):
                        distance += pow(abstraction[k] - cluster_centers[prediction][k], 2)
                    correct_distance.append(distance ** 0.5)
                distance_category['correct_pictures'] = correct_distance
                distance_category['correct_prediction'] = abstraction_category['correct_prediction']

        else:
            distance_category['correct_pictures'] = []
            distance_category['correct_prediction'] = []

        if abstraction_category['wrong_pictures'] is not None:
            wrong_distance = []
            abstraction_length = sum(neuron_number_list)
            if abstraction_category['wrong_pictures'].shape[0] == 0:
                distance_category['wrong_pictures'] = wrong_distance
                distance_category['wrong_prediction'] = abstraction_category['wrong_prediction']
            else:
                for abstraction, prediction in zip(abstraction_category['wrong_pictures'],
                                                   abstraction_category['wrong_prediction']):
                    distance = 0.0
                    for k in range(abstraction_length):
                        distance += pow(abstraction[k] - cluster_centers[prediction][k], 2)
                    wrong_distance.append(distance ** 0.5)
                distance_category['wrong_pictures'] = wrong_distance
                distance_category['wrong_prediction'] = abstraction_category['wrong_prediction']

        else:
            distance_category['wrong_pictures'] = []
            distance_category['wrong_prediction'] = []

        euclidean_distance[category] = distance_category

    return euclidean_distance


if __name__ == "__main__":
    # if research has been done, analyze ...
    if os.path.exists(f'./data/mnist/single_layer_selection_ood.pkl') and \
            os.path.exists(f'./data/mnist/multi_layers_selection_ood.pkl') and \
            os.path.exists(f'./data/fmnist/single_layer_selection_ood.pkl') and \
            os.path.exists(f'./data/fmnist/multi_layers_selection_ood.pkl') and \
            os.path.exists(f'./data/cifar10/single_layer_selection_ood.pkl') and \
            os.path.exists(f'./data/cifar10/multi_layers_selection_ood.pkl') and \
            os.path.exists(f'./data/fmnist/single_layer_selection_adversarial.pkl') and \
            os.path.exists(f'./data/fmnist/multi_layers_selection_adversarial.pkl') and \
            os.path.exists(f'./data/cifar10/single_layer_selection_adversarial.pkl') and \
            os.path.exists(f'./data/cifar10/multi_layers_selection_adversarial.pkl'):

        # # Single Layer
        print('Single Layer Selection Analyzing...')
        x = [a for a in range(1, 8)]
        x_cifar = [3, 4]
        single_ood_file1 = open(f'./data/mnist/single_layer_selection_ood.pkl', 'rb')
        single_layer_ood_performance_1 = pickle.load(single_ood_file1)
        single_ood_file2 = open(f'./data/fmnist/single_layer_selection_ood.pkl', 'rb')
        single_layer_ood_performance_2 = pickle.load(single_ood_file2)
        single_ood_file3 = open(f'./data/cifar10/single_layer_selection_ood.pkl', 'rb')
        single_layer_ood_performance_3 = pickle.load(single_ood_file3)
        single_adv_file1 = open(f'./data/fmnist/single_layer_selection_adversarial.pkl', 'rb')
        single_layer_adv_performance_1 = pickle.load(single_adv_file1)
        single_adv_file2 = open(f'./data/cifar10/single_layer_selection_adversarial.pkl', 'rb')
        single_layer_adv_performance_2 = pickle.load(single_adv_file2)

        MNISTvsFMNIST = single_layer_ood_performance_1['FMNIST']
        MNISTvsOmniglot = single_layer_ood_performance_1['Omniglot']
        FMNISTvsMNIST = single_layer_ood_performance_2['MNIST']
        FMNISTvsOmniglot = single_layer_ood_performance_2['Omniglot']
        Cifar10vsTinyImageNet = single_layer_ood_performance_3['TinyImageNet']
        Cifar10vsiSUN = single_layer_ood_performance_3['iSUN']
        Cifar10vsLSUN = single_layer_ood_performance_3['LSUN']
        FMNISTvsBA = single_layer_adv_performance_1['ba']
        FMNISTvsHSJA = single_layer_adv_performance_1['hopskipjumpattack_l2']
        FMNISTvsJSMA = single_layer_adv_performance_1['jsma']
        FMNISTvsSA = single_layer_adv_performance_1['squareattack_linf']
        FMNISTvsWA = single_layer_adv_performance_1['wassersteinattack']
        Cifar10vsBA = single_layer_adv_performance_2['ba']
        Cifar10vsCW = single_layer_adv_performance_2['cw_l2']
        Cifar10vsDeepFool = single_layer_adv_performance_2['deepfool']
        Cifar10vsEAD = single_layer_adv_performance_2['ead']
        Cifar10vsFGSM = single_layer_adv_performance_2['fgsm']
        Cifar10vsHSJA = single_layer_adv_performance_2['hopskipjumpattack_l2']
        Cifar10vsJSMA = single_layer_adv_performance_2['jsma']
        Cifar10vsNewtonFool = single_layer_adv_performance_2['newtonfool']
        Cifar10vsPixelAttack = single_layer_adv_performance_2['pixelattack']
        Cifar10vsSA = single_layer_adv_performance_2['squareattack_linf']
        Cifar10vsWA = single_layer_adv_performance_2['wassersteinattack']
        layer = ['Conv1', 'Conv2', 'Flatten', 'FC1', 'FC2', 'FC3', 'FC4']
        plt.ylim(0.5, 1)
        ax = plt.gca()
        y_major_locator = plt.MultipleLocator(0.05)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.xticks(x, layer, rotation=-45)

        plt.plot(x, MNISTvsFMNIST, label='MNIST vs. FMNIST (OOD)', marker='.')
        plt.plot(x, MNISTvsOmniglot, label='MNIST vs. Omniglot (OOD)', marker='.')
        plt.plot(x, FMNISTvsMNIST, label='FMNIST vs. MNIST (OOD)', marker='.')
        plt.plot(x, FMNISTvsOmniglot, label='FMNIST vs. Omniglot (OOD)', marker='.')
        plt.plot(x_cifar, Cifar10vsTinyImageNet, label='Cifar10 vs TinyImageNet (OOD)', marker='.')
        plt.plot(x_cifar, Cifar10vsiSUN, label='Cifar10 vs iSUN (OOD)', marker='.')
        plt.plot(x_cifar, Cifar10vsLSUN, label='Cifar10 vs. LSUN (OOD)', marker='.')
        plt.plot(x, FMNISTvsBA, label='FMNIST vs. BA (Adversarial)', marker='.')
        plt.plot(x, FMNISTvsHSJA, label='FMNIST vs. HJSA (Adversarial)', marker='.')
        plt.plot(x, FMNISTvsJSMA, label='FMNIST vs. JSMA (Adversarial)', marker='.')
        plt.plot(x, FMNISTvsSA, label='FMNIST vs. SA (Adversarial)', marker='.')
        plt.plot(x, FMNISTvsWA, label='FMNIST vs. WA (Adversarial)', marker='.')
        plt.plot(x_cifar, Cifar10vsBA, label='Cifar10 vs. BA (Adversarial)', marker='.')
        plt.plot(x_cifar, Cifar10vsCW, label='Cifar10 vs. CW-L2 (Adversarial)', marker='.')
        plt.plot(x_cifar, Cifar10vsDeepFool, label='Cifar10 vs. DeepFool (Adversarial)', marker='.')
        plt.plot(x_cifar, Cifar10vsEAD, label='Cifar10 vs. EAD (Adversarial)', marker='.')
        plt.plot(x_cifar, Cifar10vsFGSM, label='Cifar10 vs. FGSM (Adversarial)', marker='.')
        plt.plot(x_cifar, Cifar10vsHSJA, label='Cifar10 vs. HSJA-L2 (Adversarial)', marker='.')
        plt.plot(x_cifar, Cifar10vsJSMA, label='Cifar10 vs. JSMA (Adversarial)', marker='.')
        plt.plot(x_cifar, Cifar10vsNewtonFool, label='Cifar10 vs. NewtonFool (Adversarial)', marker='.')
        plt.plot(x_cifar, Cifar10vsPixelAttack, label='Cifar10 vs. PixelAttack (Adversarial)', marker='.')
        plt.plot(x_cifar, Cifar10vsSA, label='Cifar10 vs. SA (Adversarial)', marker='.')
        plt.plot(x_cifar, Cifar10vsWA, label='Cifar10 vs. WA (Adversarial)', marker='.')
        plt.legend(loc='lower right', prop={'size': 13})
        plt.tick_params(labelsize=15)
        plt.show()

        print('Layers Aggregation Selection Analyzing...')
        x = [a for a in range(1, 8)]
        x_cifar = [4, 5]
        multi_ood_file1 = open(f'./data/mnist/multi_layers_selection_ood.pkl', 'rb')
        multi_layers_ood_performance_1 = pickle.load(multi_ood_file1)
        multi_ood_file2 = open(f'./data/fmnist/multi_layers_selection_ood.pkl', 'rb')
        multi_layers_ood_performance_2 = pickle.load(multi_ood_file2)
        multi_ood_file3 = open(f'./data/cifar10/multi_layers_selection_ood.pkl', 'rb')
        multi_layers_ood_performance_3 = pickle.load(multi_ood_file3)
        multi_adv_file1 = open(f'./data/fmnist/multi_layers_selection_adversarial.pkl', 'rb')
        multi_layers_adv_performance_1 = pickle.load(multi_adv_file1)
        multi_adv_file2 = open(f'./data/cifar10/multi_layers_selection_adversarial.pkl', 'rb')
        multi_layers_adv_performance_2 = pickle.load(multi_adv_file2)

        MNISTvsFMNIST = multi_layers_ood_performance_1['FMNIST']
        MNISTvsOmniglot = multi_layers_ood_performance_1['Omniglot']
        FMNISTvsMNIST = multi_layers_ood_performance_2['MNIST']
        FMNISTvsOmniglot = multi_layers_ood_performance_2['Omniglot']
        Cifar10vsTinyImageNet = multi_layers_ood_performance_3['TinyImageNet']
        Cifar10vsiSUN = multi_layers_ood_performance_3['iSUN']
        Cifar10vsLSUN = multi_layers_ood_performance_3['LSUN']
        FMNISTvsBA = multi_layers_adv_performance_1['ba']
        FMNISTvsHSJA = multi_layers_adv_performance_1['hopskipjumpattack_l2']
        FMNISTvsJSMA = multi_layers_adv_performance_1['jsma']
        FMNISTvsSA = multi_layers_adv_performance_1['squareattack_linf']
        FMNISTvsWA = multi_layers_adv_performance_1['wassersteinattack']
        Cifar10vsBA = multi_layers_adv_performance_2['ba']
        Cifar10vsCW = multi_layers_adv_performance_2['cw_l2']
        Cifar10vsDeepFool = multi_layers_adv_performance_2['deepfool']
        Cifar10vsEAD = multi_layers_adv_performance_2['ead']
        Cifar10vsFGSM = multi_layers_adv_performance_2['fgsm']
        Cifar10vsHSJA = multi_layers_adv_performance_2['hopskipjumpattack_l2']
        Cifar10vsJSMA = multi_layers_adv_performance_2['jsma']
        Cifar10vsNewtonFool = multi_layers_adv_performance_2['newtonfool']
        Cifar10vsPixelAttack = multi_layers_adv_performance_2['pixelattack']
        Cifar10vsSA = multi_layers_adv_performance_2['squareattack_linf']
        Cifar10vsWA = multi_layers_adv_performance_2['wassersteinattack']
        layer = ['FC4', 'FC4-FC3', 'FC4-FC2', 'FC4-FC1', 'FC4-Flatten', 'FC4-Conv2', 'FC4-Conv1']
        plt.ylim(0.60, 1)
        ax = plt.gca()
        y_major_locator = plt.MultipleLocator(0.05)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.xticks(x, layer, rotation=-45)

        plt.plot(x, MNISTvsFMNIST, label='MNIST vs. FMNIST (OOD)', marker='.')
        plt.plot(x, MNISTvsOmniglot, label='MNIST vs. Omniglot (OOD)', marker='.')
        plt.plot(x, FMNISTvsMNIST, label='FMNIST vs. MNIST (OOD)', marker='.')
        plt.plot(x, FMNISTvsOmniglot, label='FMNIST vs. Omniglot (OOD)', marker='.')
        plt.plot(x_cifar, Cifar10vsTinyImageNet, label='Cifar10 vs TinyImageNet (OOD)', marker='.')
        plt.plot(x_cifar, Cifar10vsiSUN, label='Cifar10 vs iSUN (OOD)', marker='.')
        plt.plot(x_cifar, Cifar10vsLSUN, label='Cifar10 vs. LSUN (OOD)', marker='.')
        plt.plot(x, FMNISTvsBA, label='FMNIST vs. BA (Adversarial)', marker='.')
        plt.plot(x, FMNISTvsHSJA, label='FMNIST vs. HJSA (Adversarial)', marker='.')
        plt.plot(x, FMNISTvsJSMA, label='FMNIST vs. JSMA (Adversarial)', marker='.')
        plt.plot(x, FMNISTvsSA, label='FMNIST vs. SA (Adversarial)', marker='.')
        plt.plot(x, FMNISTvsWA, label='FMNIST vs. WA (Adversarial)', marker='.')
        plt.plot(x_cifar, Cifar10vsBA, label='Cifar10 vs. BA (Adversarial)', marker='.')
        plt.plot(x_cifar, Cifar10vsCW, label='Cifar10 vs. CW-L2 (Adversarial)', marker='.')
        plt.plot(x_cifar, Cifar10vsDeepFool, label='Cifar10 vs. DeepFool (Adversarial)', marker='.')
        plt.plot(x_cifar, Cifar10vsEAD, label='Cifar10 vs. EAD (Adversarial)', marker='.')
        plt.plot(x_cifar, Cifar10vsFGSM, label='Cifar10 vs. FGSM (Adversarial)', marker='.')
        plt.plot(x_cifar, Cifar10vsHSJA, label='Cifar10 vs. HSJA-L2 (Adversarial)', marker='.')
        plt.plot(x_cifar, Cifar10vsJSMA, label='Cifar10 vs. JSMA (Adversarial)', marker='.')
        plt.plot(x_cifar, Cifar10vsNewtonFool, label='Cifar10 vs. NewtonFool (Adversarial)', marker='.')
        plt.plot(x_cifar, Cifar10vsPixelAttack, label='Cifar10 vs. PixelAttack (Adversarial)', marker='.')
        plt.plot(x_cifar, Cifar10vsSA, label='Cifar10 vs. SA (Adversarial)', marker='.')
        plt.plot(x_cifar, Cifar10vsWA, label='Cifar10 vs. WA (Adversarial)', marker='.')
        plt.legend(loc='lower right', prop={'size': 13})
        plt.tick_params(labelsize=15)
        plt.show()

        exit()

    run_single_layer_selection_ood = False
    if run_single_layer_selection_ood:
        # ********************************* single layer selection - ood ******************************************** #
        id_dataset_name = 'mnist'
        # id_dataset_name = 'fmnist'
        # id_dataset_name = 'cifar10'

        if os.path.exists(f'./data/{id_dataset_name}/single_layer_selection_ood.pkl'):
            print('Single Layer Selection (OOD) Research has finished! The result was saved in:\n '
                  f'./data/{id_dataset_name}/single_layer_selection_ood.pkl')
        else:
            if id_dataset_name == 'mnist':
                model_path = './models/lenet_mnist/'
                research_layers = [-15, -14, -11, -10, -8, -6, -4]
                neurons_of_each_layer = [784, 576, 2880, 320, 160, 80, 40]
                x_train, y_train, x_test, y_test = load_mnist()
                OOD_dataset = ['FMNIST', 'Omniglot']
                fmnist_data, _ = load_ood_data(ood_dataset='FMNIST', id_model_path=model_path,
                                               num_of_categories=num_of_labels[id_dataset_name])
                omniglot_data, _ = load_ood_data(ood_dataset='Omniglot', id_model_path=model_path,
                                                 num_of_categories=num_of_labels[id_dataset_name])
                ood_data_dict = {'FMNIST': fmnist_data, 'Omniglot': omniglot_data}
                single_layer_performance = {'FMNIST': [], 'Omniglot': []}
            elif id_dataset_name == 'fmnist':
                model_path = './models/lenet_fmnist/'
                research_layers = [-15, -14, -11, -10, -8, -6, -4]
                neurons_of_each_layer = [784, 576, 2880, 320, 160, 80, 40]
                x_train, y_train, x_test, y_test = load_fmnist()
                OOD_dataset = ['MNIST', 'Omniglot']
                mnist_data, _ = load_ood_data(ood_dataset='MNIST', id_model_path=model_path,
                                              num_of_categories=num_of_labels[id_dataset_name])
                omniglot_data, _ = load_ood_data(ood_dataset='Omniglot', id_model_path=model_path,
                                                 num_of_categories=num_of_labels[id_dataset_name])
                ood_data_dict = {'MNIST': mnist_data, 'Omniglot': omniglot_data}
                single_layer_performance = {'MNIST': [], 'Omniglot': []}
            elif id_dataset_name == 'cifar10':
                model_path = './models/vgg19_cifar10/'
                research_layers = [-7, -5]
                neurons_of_each_layer = [512, 512]
                x_train, y_train, x_test, y_test = load_cifar10()
                OOD_dataset = ['TinyImageNet', 'LSUN', 'iSUN']
                TinyImageNet_ood_data, _ = load_ood_data(ood_dataset='TinyImageNet', id_model_path=model_path,
                                                                     num_of_categories=num_of_labels[id_dataset_name])
                LSUN_ood_data, _ = load_ood_data(ood_dataset='LSUN', id_model_path=model_path,
                                                 num_of_categories=num_of_labels[id_dataset_name])
                iSUN_ood_data, _ = load_ood_data(ood_dataset='iSUN', id_model_path=model_path,
                                                 num_of_categories=num_of_labels[id_dataset_name])
                ood_data_dict = {'TinyImageNet': TinyImageNet_ood_data,
                                 'LSUN': LSUN_ood_data,
                                 'iSUN': iSUN_ood_data}
                single_layer_performance = {'TinyImageNet': [], 'LSUN': [], 'iSUN': []}
            else:
                print('Dataset is not existed！')
                exit()

            for i in range(len(research_layers)):
                single_research_layers = [research_layers[i]]
                single_neurons_of_layers = [neurons_of_each_layer[i]]
                current_research_layers = single_research_layers
                current_neurons_of_layers = single_neurons_of_layers

                # ********************** Train Detector **************************** #
                print('\n********************** Train Detector ****************************')
                print("Research Layers:", current_research_layers)

                train_picture_classified = classify_id_pictures(id_dataset=id_dataset_name, dataset=x_train,
                                                                labels=tf.argmax(y_train, axis=1), model_path=model_path)

                print('\nGet neural value of train dataset...')
                train_picture_neural_value = get_neural_value(id_dataset=id_dataset_name,
                                                              layers=current_research_layers,
                                                              tf_model_path=model_path,
                                                              pictures_classified=train_picture_classified)

                print('\nStatistic of train data neural value...')
                train_nv_statistic = statistic_of_neural_value(id_dataset=id_dataset_name,
                                                               layers=current_research_layers,
                                                               length_of_layers=current_neurons_of_layers,
                                                               neural_value=train_picture_neural_value)
                print('finished!')

                print('\nEncoding ReAD abstraction of train dataset:')
                train_ReAD_abstractions = encode_abstraction(id_dataset=id_dataset_name,
                                                             layers=current_research_layers,
                                                             length_of_layers=current_neurons_of_layers,
                                                             neural_value=train_picture_neural_value,
                                                             train_dataset_statistic=train_nv_statistic)
                train_ReAD_concatenated = concatenate_data_between_layers(id_dataset=id_dataset_name,
                                                                          layers=current_research_layers,
                                                                          data_dict=train_ReAD_abstractions)

                print('\nK-Means Clustering of Combination Abstraction on train data:')
                k_means_centers = k_means(data=train_ReAD_concatenated,
                                          category_number=num_of_labels[id_dataset_name])

                # ********************** Evaluate OOD Detection **************************** #
                print('\n********************** Evaluate OOD Detection ****************************')
                test_picture_classified = classify_id_pictures(id_dataset=id_dataset_name, dataset=x_test,
                                                               labels=tf.argmax(y_test, axis=1), model_path=model_path)
                print('\nGet neural value of test dataset:')
                test_picture_neural_value = get_neural_value(id_dataset=id_dataset_name,
                                                             layers=current_research_layers,
                                                             tf_model_path=model_path,
                                                             pictures_classified=test_picture_classified)
                print('\nEncoding ReAD abstraction of test dataset:')
                test_ReAD_abstractions = encode_abstraction(id_dataset=id_dataset_name,
                                                            layers=current_research_layers,
                                                            length_of_layers=current_neurons_of_layers,
                                                            neural_value=test_picture_neural_value,
                                                            train_dataset_statistic=train_nv_statistic)
                test_ReAD_concatenated = concatenate_data_between_layers(id_dataset=id_dataset_name,
                                                                         layers=current_research_layers,
                                                                         data_dict=test_ReAD_abstractions)
                print('\nCalculate distance between abstractions and cluster centers ...')
                test_distance = statistic_distance(id_dataset=id_dataset_name,
                                                   length_of_layers=current_neurons_of_layers,
                                                   abstractions=test_ReAD_concatenated,
                                                   cluster_centers=k_means_centers)

                for ood in OOD_dataset:
                    print('\n************Evaluating*************')
                    print(f'In-Distribution Data: {id_dataset_name}, Out-of-Distribution Data: {ood}.')

                    ood_data = ood_data_dict[ood]

                    print('\nGet neural value of ood dataset...')
                    ood_picture_neural_value = get_neural_value(id_dataset=id_dataset_name,
                                                                layers=current_research_layers,
                                                                tf_model_path=model_path,
                                                                pictures_classified=ood_data)

                    print('\nEncoding ReAD abstraction of ood dataset...')
                    ood_ReAD_abstractions = encode_abstraction(id_dataset=id_dataset_name,
                                                               layers=current_research_layers,
                                                               length_of_layers=current_neurons_of_layers,
                                                               neural_value=ood_picture_neural_value,
                                                               train_dataset_statistic=train_nv_statistic)
                    ood_ReAD_concatenated = concatenate_data_between_layers(id_dataset=id_dataset_name,
                                                                            layers=current_research_layers,
                                                                            data_dict=ood_ReAD_abstractions)
                    print('\nCalculate distance between abstractions and cluster centers ...')
                    ood_distance = statistic_distance(id_dataset=id_dataset_name,
                                                      length_of_layers=current_neurons_of_layers,
                                                      abstractions=ood_ReAD_concatenated,
                                                      cluster_centers=k_means_centers)
                    performance = sk_auc(id_dataset_name, test_distance, ood_distance)
                    single_layer_performance[ood].append(performance[-1]['avg']['AUROC'])
                    print('\nPerformance of Detector:')
                    print('AUROC: {:.6f}'.format(performance[-1]['avg']['AUROC']))
                    print('*************************************\n')
            single_performance_file = open(f'./data/{id_dataset_name}/single_layer_selection_ood.pkl', 'wb')
            pickle.dump(single_layer_performance, single_performance_file)
            single_performance_file.close()
            print('Single Layer Selection Research has finished! The result was saved in:\n '
                  f'./data/{id_dataset_name}/single_layer_selection_ood.pkl')

    run_single_layer_selection_adversarial = False
    if run_single_layer_selection_adversarial:
        # *********************************** single layer selection - adversarial ************************** #
        # clean_dataset_name = 'fmnist'
        clean_dataset_name = 'cifar10'
        # clean_dataset_name = 'svhn'

        if os.path.exists(f'./data/{clean_dataset_name}/single_layer_selection_adversarial.pkl'):
            print('Single Layer Selection (Adversarial) Research has finished! The result was saved in:\n '
                  f'./data/{clean_dataset_name}/single_layer_selection_adversarial.pkl')
        else:
            if clean_dataset_name == 'fmnist':
                model_path = './models/lenet_fmnist/'
                research_layers = [-15, -14, -11, -10, -8, -6, -4]
                neurons_of_each_layer = [784, 576, 2880, 320, 160, 80, 40]
                x_train, y_train, x_test, y_test = load_fmnist()
                adversarial_attacks = ['ba', 'hopskipjumpattack_l2', 'jsma', 'squareattack_linf', 'wassersteinattack']
                single_layer_performance = {'ba': [], 'hopskipjumpattack_l2': [], 'jsma': [],
                                            'squareattack_linf': [], 'wassersteinattack': []}
            elif clean_dataset_name == 'cifar10':
                model_path = './models/vgg19_cifar10/'
                research_layers = [-7, -5]
                neurons_of_each_layer = [512, 512]
                x_train, y_train, x_test, y_test = load_cifar10()
                adversarial_attacks = ['ba', 'cw_l2', 'deepfool', 'ead', 'fgsm', 'hopskipjumpattack_l2',
                                       'jsma', 'newtonfool', 'pixelattack', 'squareattack_linf', 'wassersteinattack']
                single_layer_performance = {'ba': [], 'cw_l2': [], 'deepfool': [], 'ead': [], 'fgsm': [],
                                            'hopskipjumpattack_l2': [], 'jsma': [], 'newtonfool': [], 'pixelattack': [],
                                            'squareattack_linf': [], 'wassersteinattack': []}

            else:
                print('Dataset is not existed！')
                exit()

            for i in range(len(research_layers)):
                single_research_layers = [research_layers[i]]
                single_neurons_of_layers = [neurons_of_each_layer[i]]
                current_research_layers = single_research_layers
                current_neurons_of_layers = single_neurons_of_layers

                # ********************** Train Detector **************************** #
                print('\n********************** Train Detector ****************************')
                print("Research Layers:", current_research_layers)

                train_picture_classified = classify_id_pictures(id_dataset=clean_dataset_name, dataset=x_train,
                                                                labels=tf.argmax(y_train, axis=1),
                                                                model_path=model_path)

                print('\nGet neural value of train dataset...')
                train_picture_neural_value = get_neural_value(id_dataset=clean_dataset_name,
                                                              layers=current_research_layers,
                                                              tf_model_path=model_path,
                                                              pictures_classified=train_picture_classified)

                print('\nStatistic of train data neural value...')
                train_nv_statistic = statistic_of_neural_value(id_dataset=clean_dataset_name,
                                                               layers=current_research_layers,
                                                               length_of_layers=current_neurons_of_layers,
                                                               neural_value=train_picture_neural_value)

                print('\nEncoding ReAD abstraction of train dataset:')
                train_ReAD_abstractions = encode_abstraction(id_dataset=clean_dataset_name,
                                                             layers=current_research_layers,
                                                             length_of_layers=current_neurons_of_layers,
                                                             neural_value=train_picture_neural_value,
                                                             train_dataset_statistic=train_nv_statistic)
                train_ReAD_concatenated = concatenate_data_between_layers(id_dataset=clean_dataset_name,
                                                                          layers=current_research_layers,
                                                                          data_dict=train_ReAD_abstractions)

                print('\nK-Means Clustering of Combination Abstraction on train data:')
                k_means_centers = k_means(data=train_ReAD_concatenated,
                                          category_number=num_of_labels[clean_dataset_name])

                print('\n********************** Evaluate Adversarial Detection ****************************')
                for attack in adversarial_attacks:
                    print(f'\nATTACK: {attack}')
                    clean_data_classified, adv_data_classified, num_of_test = \
                        load_clean_adv_data(id_dataset=clean_dataset_name, attack=attack,
                                            num_of_categories=num_of_labels[clean_dataset_name])

                    print('\nGet neural value:')
                    print('CLEAN dataset ...')
                    clean_data_neural_value = get_neural_value(id_dataset=clean_dataset_name,
                                                               layers=current_research_layers,
                                                               tf_model_path=model_path,
                                                               pictures_classified=clean_data_classified)
                    print('\nADVERSARIAL dataset ...')
                    adv_data_neural_value = get_neural_value(id_dataset=clean_dataset_name,
                                                             layers=current_research_layers,
                                                             tf_model_path=model_path,
                                                             pictures_classified=adv_data_classified)
                    print(f'\nEncoding: (selective rate - {selective_rate}):')
                    print("CLEAN dataset ...")
                    clean_data_ReAD = encode_abstraction(id_dataset=clean_dataset_name,
                                                         layers=current_research_layers,
                                                         length_of_layers=current_neurons_of_layers,
                                                         neural_value=clean_data_neural_value,
                                                         train_dataset_statistic=train_nv_statistic,)
                    print('ADVERSARIAL dataset ...')
                    adv_data_ReAD = encode_abstraction(id_dataset=clean_dataset_name,
                                                       layers=current_research_layers,
                                                       length_of_layers=current_neurons_of_layers,
                                                       neural_value=adv_data_neural_value,
                                                       train_dataset_statistic=train_nv_statistic)
                    clean_ReAD_concatenated = \
                        concatenate_data_between_layers(id_dataset=clean_dataset_name,
                                                        layers=current_research_layers,
                                                        data_dict=clean_data_ReAD)
                    adv_ReAD_concatenated = \
                        concatenate_data_between_layers(id_dataset=clean_dataset_name,
                                                        layers=current_research_layers,
                                                        data_dict=adv_data_ReAD)

                    print('\nCalculate distance between abstractions and cluster centers ...')
                    print('CLEAN dataset ...')
                    clean_distance = statistic_distance(id_dataset=clean_dataset_name,
                                                        length_of_layers=current_neurons_of_layers,
                                                        abstractions=clean_ReAD_concatenated,
                                                        cluster_centers=k_means_centers)
                    print('ADVERSARIAL dataset ...')
                    adv_distance = statistic_distance(id_dataset=clean_dataset_name,
                                                      length_of_layers=current_neurons_of_layers,
                                                      abstractions=adv_ReAD_concatenated,
                                                      cluster_centers=k_means_centers)

                    performance = sk_auc(clean_dataset_name, clean_distance, adv_distance)
                    single_layer_performance[attack].append(performance[-1]['avg']['AUROC'])
                    print('\nPerformance of Detector:')
                    print('AUROC: {:.6f}'.format(performance[-1]['avg']['AUROC']))
                    print('*************************************\n')
            single_performance_file = open(f'./data/{clean_dataset_name}/single_layer_selection_adversarial.pkl', 'wb')
            pickle.dump(single_layer_performance, single_performance_file)
            single_performance_file.close()
            print('Single Layer Selection Research has finished! The result was saved in:\n '
                  f'./data/{clean_dataset_name}/single_layer_selection_adversarial.pkl')

    run_layers_aggregation_ood = False
    if run_layers_aggregation_ood:
        # *********************************** layers aggregation - ood ************************** #
        id_dataset_name = 'mnist'
        # id_dataset_name = 'fmnist'
        # id_dataset_name = 'cifar10'

        if os.path.exists(f'./data/{id_dataset_name}/multi_layers_selection_ood.pkl'):
            print('Layers Aggregation Selection Research has finished! The result was saved in:\n '
                  f'./data/{id_dataset_name}/multi_layers_selection_ood.pkl')
        else:
            if id_dataset_name == 'mnist':
                model_path = './models/lenet_mnist/'
                research_layers = [-15, -14, -11, -10, -8, -6, -4]
                neurons_of_each_layer = [784, 576, 2880, 320, 160, 80, 40]
                x_train, y_train, x_test, y_test = load_mnist()
                OOD_dataset = ['FMNIST', 'Omniglot']
                fmnist_data, _ = load_ood_data(ood_dataset='FMNIST', id_model_path=model_path,
                                               num_of_categories=num_of_labels[id_dataset_name])
                omniglot_data, _ = load_ood_data(ood_dataset='Omniglot', id_model_path=model_path,
                                                 num_of_categories=num_of_labels[id_dataset_name])
                ood_data_dict = {'FMNIST': fmnist_data, 'Omniglot': omniglot_data}
                multi_layers_performance = {'FMNIST': [], 'Omniglot': []}
            elif id_dataset_name == 'fmnist':
                model_path = './models/lenet_fmnist/'
                research_layers = [-15, -14, -11, -10, -8, -6, -4]
                neurons_of_each_layer = [784, 576, 2880, 320, 160, 80, 40]
                x_train, y_train, x_test, y_test = load_fmnist()
                OOD_dataset = ['MNIST', 'Omniglot']
                mnist_data, _ = load_ood_data(ood_dataset='MNIST', id_model_path=model_path,
                                              num_of_categories=num_of_labels[id_dataset_name])
                omniglot_data, _ = load_ood_data(ood_dataset='Omniglot', id_model_path=model_path,
                                                 num_of_categories=num_of_labels[id_dataset_name])
                ood_data_dict = {'MNIST': mnist_data, 'Omniglot': omniglot_data}
                multi_layers_performance = {'MNIST': [], 'Omniglot': []}
            elif id_dataset_name == 'cifar10':
                model_path = './models/vgg19_cifar10/'
                research_layers = [-7, -5]
                neurons_of_each_layer = [512, 512]
                x_train, y_train, x_test, y_test = load_cifar10()
                OOD_dataset = ['TinyImageNet', 'LSUN', 'iSUN']
                TinyImageNet_ood_data, _ = load_ood_data(ood_dataset='TinyImageNet', id_model_path=model_path,
                                                                     num_of_categories=num_of_labels[id_dataset_name])
                LSUN_ood_data, _ = load_ood_data(ood_dataset='LSUN', id_model_path=model_path,
                                                 num_of_categories=num_of_labels[id_dataset_name])
                iSUN_ood_data, _ = load_ood_data(ood_dataset='iSUN', id_model_path=model_path,
                                                 num_of_categories=num_of_labels[id_dataset_name])
                ood_data_dict = {'TinyImageNet': TinyImageNet_ood_data,
                                 'LSUN': LSUN_ood_data,
                                 'iSUN': iSUN_ood_data}
                multi_layers_performance = {'TinyImageNet': [], 'LSUN': [], 'iSUN': []}
            else:
                print('Dataset is not existed！')
                exit()

            for i in range(len(research_layers)):
                multi_research_layers = research_layers[len(research_layers)-i-1: len(research_layers)]
                multi_neurons_of_layers = neurons_of_each_layer[len(research_layers)-i-1: len(research_layers)]
                current_research_layers = multi_research_layers
                current_neurons_of_layers = multi_neurons_of_layers

                # ********************** Train Detector **************************** #
                print('\n********************** Train Detector ****************************')
                print("Research Layers:", current_research_layers)

                train_picture_classified = classify_id_pictures(id_dataset=id_dataset_name, dataset=x_train,
                                                                labels=tf.argmax(y_train, axis=1), model_path=model_path)

                print('\nGet neural value of train dataset...')
                train_picture_neural_value = get_neural_value(id_dataset=id_dataset_name,
                                                              layers=current_research_layers,
                                                              tf_model_path=model_path,
                                                              pictures_classified=train_picture_classified)

                print('\nStatistic of train data neural value...')
                train_nv_statistic = statistic_of_neural_value(id_dataset=id_dataset_name,
                                                               layers=current_research_layers,
                                                               length_of_layers=current_neurons_of_layers,
                                                               neural_value=train_picture_neural_value)
                print('finished!')

                print('\nEncoding ReAD abstraction of train dataset:')
                train_ReAD_abstractions = encode_abstraction(id_dataset=id_dataset_name,
                                                             layers=current_research_layers,
                                                             length_of_layers=current_neurons_of_layers,
                                                             neural_value=train_picture_neural_value,
                                                             train_dataset_statistic=train_nv_statistic)
                train_ReAD_concatenated = concatenate_data_between_layers(id_dataset=id_dataset_name,
                                                                          layers=current_research_layers,
                                                                          data_dict=train_ReAD_abstractions)

                print('\nK-Means Clustering of Combination Abstraction on train data:')
                k_means_centers = k_means(data=train_ReAD_concatenated,
                                          category_number=num_of_labels[id_dataset_name])

                # ********************** Evaluate OOD Detection **************************** #
                print('\n********************** Evaluate OOD Detection ****************************')
                test_picture_classified = classify_id_pictures(id_dataset=id_dataset_name, dataset=x_test,
                                                               labels=tf.argmax(y_test, axis=1), model_path=model_path)
                print('\nGet neural value of test dataset:')
                test_picture_neural_value = get_neural_value(id_dataset=id_dataset_name,
                                                             layers=current_research_layers,
                                                             tf_model_path=model_path,
                                                             pictures_classified=test_picture_classified)
                print('\nEncoding ReAD abstraction of test dataset:')
                test_ReAD_abstractions = encode_abstraction(id_dataset=id_dataset_name,
                                                            layers=current_research_layers,
                                                            length_of_layers=current_neurons_of_layers,
                                                            neural_value=test_picture_neural_value,
                                                            train_dataset_statistic=train_nv_statistic)
                test_ReAD_concatenated = concatenate_data_between_layers(id_dataset=id_dataset_name,
                                                                         layers=current_research_layers,
                                                                         data_dict=test_ReAD_abstractions)
                print('\nCalculate distance between abstractions and cluster centers ...')
                test_distance = statistic_distance(id_dataset=id_dataset_name,
                                                   length_of_layers=current_neurons_of_layers,
                                                   abstractions=test_ReAD_concatenated,
                                                   cluster_centers=k_means_centers)

                for ood in OOD_dataset:
                    print('\n************Evaluating*************')
                    print(f'In-Distribution Data: {id_dataset_name}, Out-of-Distribution Data: {ood}.')

                    ood_data = ood_data_dict[ood]

                    print('\nGet neural value of ood dataset...')
                    ood_picture_neural_value = get_neural_value(id_dataset=id_dataset_name,
                                                                layers=current_research_layers,
                                                                tf_model_path=model_path,
                                                                pictures_classified=ood_data)

                    print('\nEncoding ReAD abstraction of ood dataset...')
                    ood_ReAD_abstractions = encode_abstraction(id_dataset=id_dataset_name,
                                                               layers=current_research_layers,
                                                               length_of_layers=current_neurons_of_layers,
                                                               neural_value=ood_picture_neural_value,
                                                               train_dataset_statistic=train_nv_statistic)
                    ood_ReAD_concatenated = concatenate_data_between_layers(id_dataset=id_dataset_name,
                                                                            layers=current_research_layers,
                                                                            data_dict=ood_ReAD_abstractions)
                    print('\nCalculate distance between abstractions and cluster centers ...')
                    ood_distance = statistic_distance(id_dataset=id_dataset_name,
                                                      length_of_layers=current_neurons_of_layers,
                                                      abstractions=ood_ReAD_concatenated,
                                                      cluster_centers=k_means_centers)

                    performance = sk_auc(test_distance, ood_distance, num_of_labels[id_dataset_name])
                    multi_layers_performance[ood].append(performance[-1]['avg']['AUROC'])
                    print('\nPerformance of Detector:')
                    print('AUROC: {:.6f}'.format(performance[-1]['avg']['AUROC']))
                    print('*************************************\n')
            multi_performance_file = open(f'./data/{id_dataset_name}/multi_layers_selection_ood.pkl', 'wb')
            pickle.dump(multi_layers_performance, multi_performance_file)
            multi_performance_file.close()
            print('Layers Aggregation Selection (OOD) Research has finished! The result was saved in:\n '
                  f'./data/{id_dataset_name}/multi_layers_selection_ood.pkl')

    run_layers_aggregation_adversarial = False
    if run_layers_aggregation_adversarial:
        print('Run Layers Aggregation - Adversarial')
        # *********************************** layers_aggregation - adversarial ************************** #
        # clean_dataset_name = 'fmnist'
        clean_dataset_name = 'cifar10'
        # clean_dataset_name = 'svhn'

        if os.path.exists(f'./data/{clean_dataset_name}/multi_layers_selection_adversarial.pkl'):
            print('Layers Aggregation Selection (Adversarial) Research has finished! The result was saved in:\n '
                  f'./data/{clean_dataset_name}/multi_layers_selection_adversarial.pkl')
        else:
            if clean_dataset_name == 'fmnist':
                model_path = './models/lenet_fmnist/'
                research_layers = [-15, -14, -11, -10, -8, -6, -4]
                neurons_of_each_layer = [784, 576, 2880, 320, 160, 80, 40]
                x_train, y_train, x_test, y_test = load_fmnist()
                adversarial_attacks = ['ba', 'hopskipjumpattack_l2', 'jsma', 'squareattack_linf', 'wassersteinattack']
                multi_layers_performance = {'ba': [], 'hopskipjumpattack_l2': [], 'jsma': [],
                                            'squareattack_linf': [], 'wassersteinattack': []}
            elif clean_dataset_name == 'cifar10':
                model_path = './models/vgg19_cifar10/'
                research_layers = [-7, -5]
                neurons_of_each_layer = [512, 512]
                x_train, y_train, x_test, y_test = load_cifar10()
                adversarial_attacks = ['ba', 'cw_l2', 'deepfool', 'ead', 'fgsm', 'hopskipjumpattack_l2',
                                       'jsma', 'newtonfool', 'pixelattack', 'squareattack_linf', 'wassersteinattack']
                multi_layers_performance = {'ba': [], 'cw_l2': [], 'deepfool': [], 'ead': [], 'fgsm': [],
                                            'hopskipjumpattack_l2': [], 'jsma': [], 'newtonfool': [], 'pixelattack': [],
                                            'squareattack_linf': [], 'wassersteinattack': []}
            else:
                print('Dataset is not existed！')
                exit()

            for i in range(len(research_layers)):
                multi_research_layers = research_layers[len(research_layers)-i-1: len(research_layers)]
                multi_neurons_of_layers = neurons_of_each_layer[len(research_layers)-i-1: len(research_layers)]
                current_research_layers = multi_research_layers
                current_neurons_of_layers = multi_neurons_of_layers

                # ********************** Train Detector **************************** #
                print('\n********************** Train Detector ****************************')
                print("Research Layers:", current_research_layers)

                train_picture_classified = classify_id_pictures(id_dataset=clean_dataset_name, dataset=x_train,
                                                                labels=tf.argmax(y_train, axis=1),
                                                                model_path=model_path)

                print('\nGet neural value of train dataset...')
                train_picture_neural_value = get_neural_value(id_dataset=clean_dataset_name,
                                                              layers=current_research_layers,
                                                              tf_model_path=model_path,
                                                              pictures_classified=train_picture_classified)

                print('\nStatistic of train data neural value...')
                train_nv_statistic = statistic_of_neural_value(id_dataset=clean_dataset_name,
                                                               layers=current_research_layers,
                                                               length_of_layers=current_neurons_of_layers,
                                                               neural_value=train_picture_neural_value)

                print('\nEncoding ReAD abstraction of train dataset:')
                train_ReAD_abstractions = encode_abstraction(id_dataset=clean_dataset_name,
                                                             layers=current_research_layers,
                                                             length_of_layers=current_neurons_of_layers,
                                                             neural_value=train_picture_neural_value,
                                                             train_dataset_statistic=train_nv_statistic)
                train_ReAD_concatenated = concatenate_data_between_layers(id_dataset=clean_dataset_name,
                                                                          layers=current_research_layers,
                                                                          data_dict=train_ReAD_abstractions)

                print('\nK-Means Clustering of Combination Abstraction on train data:')
                k_means_centers = k_means(data=train_ReAD_concatenated,
                                          category_number=num_of_labels[clean_dataset_name])

                print('\n********************** Evaluate Adversarial Detection ****************************')
                for attack in adversarial_attacks:
                    print(f'\nATTACK: {attack}')
                    clean_data_classified, adv_data_classified, num_of_test = \
                        load_clean_adv_data(id_dataset=clean_dataset_name, attack=attack,
                                            num_of_categories=num_of_labels[clean_dataset_name])

                    print('\nGet neural value:')
                    print('CLEAN dataset ...')
                    clean_data_neural_value = get_neural_value(id_dataset=clean_dataset_name,
                                                               layers=current_research_layers,
                                                               tf_model_path=model_path,
                                                               pictures_classified=clean_data_classified)
                    print('\nADVERSARIAL dataset ...')
                    adv_data_neural_value = get_neural_value(id_dataset=clean_dataset_name,
                                                             layers=current_research_layers,
                                                             tf_model_path=model_path,
                                                             pictures_classified=adv_data_classified)
                    print(f'\nEncoding: (selective rate - {selective_rate}):')
                    print("CLEAN dataset ...")
                    clean_data_ReAD = encode_abstraction(id_dataset=clean_dataset_name,
                                                         layers=current_research_layers,
                                                         length_of_layers=current_neurons_of_layers,
                                                         neural_value=clean_data_neural_value,
                                                         train_dataset_statistic=train_nv_statistic,)
                    print('ADVERSARIAL dataset ...')
                    adv_data_ReAD = encode_abstraction(id_dataset=clean_dataset_name,
                                                       layers=current_research_layers,
                                                       length_of_layers=current_neurons_of_layers,
                                                       neural_value=adv_data_neural_value,
                                                       train_dataset_statistic=train_nv_statistic)
                    clean_ReAD_concatenated = \
                        concatenate_data_between_layers(id_dataset=clean_dataset_name,
                                                        layers=current_research_layers,
                                                        data_dict=clean_data_ReAD)
                    adv_ReAD_concatenated = \
                        concatenate_data_between_layers(id_dataset=clean_dataset_name,
                                                        layers=current_research_layers,
                                                        data_dict=adv_data_ReAD)

                    print('\nCalculate distance between abstractions and cluster centers ...')
                    print('CLEAN dataset ...')
                    clean_distance = statistic_distance(id_dataset=clean_dataset_name,
                                                        length_of_layers=current_neurons_of_layers,
                                                        abstractions=clean_ReAD_concatenated,
                                                        cluster_centers=k_means_centers)
                    print('ADVERSARIAL dataset ...')
                    adv_distance = statistic_distance(id_dataset=clean_dataset_name,
                                                      length_of_layers=current_neurons_of_layers,
                                                      abstractions=adv_ReAD_concatenated,
                                                      cluster_centers=k_means_centers)

                    performance = sk_auc(clean_dataset_name, clean_distance, adv_distance)
                    multi_layers_performance[attack].append(performance[-1]['avg']['AUROC'])
                    print('\nPerformance of Detector:')
                    print('AUROC: {:.6f}'.format(performance[-1]['avg']['AUROC']))
                    print('*************************************\n')
            multi_performance_file = open(f'./data/{clean_dataset_name}/multi_layers_selection_adversarial.pkl', 'wb')
            pickle.dump(multi_layers_performance, multi_performance_file)
            multi_performance_file.close()
            print('Layers Aggregation Selection (Adversarial)  Research has finished! The result was saved in:\n '
                  f'./data/{clean_dataset_name}/multi_layers_selection_adversarial.pkl')

