import matplotlib
matplotlib.get_backend()
import matplotlib.pyplot as plt
import os.path
import tensorflow as tf
import math
import pickle
import numpy as np


from ReAD import classify_id_pictures
from ReAD import get_neural_value, statistic_of_neural_value
from load_data import load_mnist, load_cifar10
from global_config import num_of_labels, selective_rate


def encode_by_selective(image_neural_value, label, encode_rate, number_of_neuron, non_class_l_average, topk,
                        all_class_average=None):

    selective = [0.0 for _ in range(number_of_neuron)]
    if all_class_average is not None:
        for r in range(number_of_neuron):
            if all_class_average[r] == 0:
                selective[r] = 0
            else:
                selective[r] = (image_neural_value[r] - non_class_l_average[label][r]) / all_class_average[r]
    else:
        for r in range(number_of_neuron):
            selective[r] = image_neural_value[r] - non_class_l_average[label][r]

    dict_sel = {}
    for index in range(len(selective)):
        dict_sel[index] = selective[index]
    sort_by_sel = sorted(dict_sel.items(), key=lambda x: x[1])

    combination_code = [0 for _ in range(number_of_neuron)]
    only_active_code = [0 for _ in range(number_of_neuron)]
    only_deactive_code = [0 for _ in range(number_of_neuron)]
    for k in range(0, math.ceil(number_of_neuron * encode_rate / 2), 1):
        combination_code[sort_by_sel[k][0]] = -1
    for k in range(-1, -math.ceil(number_of_neuron * encode_rate / 2) - 1, -1):
        combination_code[sort_by_sel[k][0]] = 1

    for k in range(-1, -math.ceil(number_of_neuron * encode_rate) - 1, -1):
        only_active_code[sort_by_sel[k][0]] = 1

    for k in range(0, math.ceil(number_of_neuron * encode_rate), 1):
        only_deactive_code[sort_by_sel[k][0]] = -1

    top_k_ReA_index = []
    top_k_ReD_index = []
    top_k_ReA_index.extend([sort_by_sel[-k][0] for k in range(1, topk+1, 1)])
    top_k_ReD_index.extend([sort_by_sel[k][0] for k in range(topk)])

    dict_neuron_value = {}
    for index in range(len(selective)):
        dict_neuron_value[index] = image_neural_value[index]
    sort_by_neuron_value = sorted(dict_neuron_value.items(), key=lambda x: x[1])
    top_k_NA_index = []
    top_k_ND_index = []
    top_k_NA_index.extend([sort_by_neuron_value[-k][0] for k in range(1, topk+1, 1)])
    top_k_ND_index.extend([sort_by_neuron_value[k][0] for k in range(topk)])

    return combination_code, only_active_code, only_deactive_code, \
        top_k_ReA_index, top_k_ReD_index, top_k_NA_index, top_k_ND_index, selective


def encode_abstraction_for_statistic(id_dataset, neural_value, train_neural_value_statistic, statistic_layers):

    categories = num_of_labels[id_dataset]
    encode_layer_number = statistic_layers['fully_connected_layers_number']
    layers = statistic_layers['fully_connected_layers']
    neuron_number_list = statistic_layers['neuron_number_of_fully_connected_layers']

    combination_abstraction_dict = {}
    for k in range(encode_layer_number):
        print("\nEncoding in fully connected layer:{}.".format(layers[k]))

        non_class_l_avg = train_neural_value_statistic[layers[k]]['not_class_l_average']
        all_class_avg = train_neural_value_statistic[layers[k]]['all_class_average']

        statistic_selectivity = np.zeros([neuron_number_list[k], 10, 2])

        topk_neurons = {0: {'ReA': [], 'ReD': [], 'NVA': [], 'NVD': []},
                        1: {'ReA': [], 'ReD': [], 'NVA': [], 'NVD': []},
                        2: {'ReA': [], 'ReD': [], 'NVA': [], 'NVD': []},
                        3: {'ReA': [], 'ReD': [], 'NVA': [], 'NVD': []},
                        4: {'ReA': [], 'ReD': [], 'NVA': [], 'NVD': []},
                        5: {'ReA': [], 'ReD': [], 'NVA': [], 'NVD': []},
                        6: {'ReA': [], 'ReD': [], 'NVA': [], 'NVD': []},
                        7: {'ReA': [], 'ReD': [], 'NVA': [], 'NVD': []},
                        8: {'ReA': [], 'ReD': [], 'NVA': [], 'NVD': []},
                        9: {'ReA': [], 'ReD': [], 'NVA': [], 'NVD': []},
                        }
        for top_i in range(1, 2):
            print(f'----------------------------top:{top_i}----------------------------------------------')
            for category in range(categories):
                print('category: {} / {}'.format(category+1, categories))
                neural_value_category = neural_value[layers[k]][category]

                statistic_top_k_ReA = [0 for _ in range(neuron_number_list[k])]
                statistic_top_k_ReD = [0 for _ in range(neuron_number_list[k])]
                statistic_top_k_NVA = [0 for _ in range(neuron_number_list[k])]
                statistic_top_k_NVD = [0 for _ in range(neuron_number_list[k])]

                if neural_value_category['correct_pictures'] is not None:
                    if neural_value_category['correct_pictures'].shape[0] == 0:
                        pass
                    else:
                        for image, label in \
                                zip(neural_value_category['correct_pictures'], neural_value_category['correct_prediction']):
                            combination_code, only_active_code, only_deactive_code, \
                                topk_ReA_index, topk_ReD_index, topk_NVA_index, topk_NVD_index, selectivity = \
                                encode_by_selective(image, label, selective_rate, neuron_number_list[k],
                                                    non_class_l_avg, top_i, all_class_avg)

                            for index in topk_ReA_index:
                                statistic_top_k_ReA[index] += 1
                            for index in topk_ReD_index:
                                statistic_top_k_ReD[index] += 1
                            for index in topk_NVA_index:
                                statistic_top_k_NVA[index] += 1
                            for index in topk_NVD_index:
                                statistic_top_k_NVD[index] += 1
                            for i in range(len(selectivity)):
                                if selectivity[i] > 0:
                                    statistic_selectivity[i][category][0] += 1
                                else:
                                    statistic_selectivity[i][category][1] += 1

                    # print(f'ReA: index:{statistic_top_k_ReA.index(max(statistic_top_k_ReA))},'
                    #       f'percentage: {max(statistic_top_k_ReA)/sum(statistic_top_k_ReA)},'
                    #       f'count: {len(statistic_top_k_ReA)-statistic_top_k_ReA.count(0)}')
                    # print(f'ReD: index:{statistic_top_k_ReD.index(max(statistic_top_k_ReD))},'
                    #       f'percentage: {max(statistic_top_k_ReD)/sum(statistic_top_k_ReD)},'
                    #       f'count: {len(statistic_top_k_ReD)-statistic_top_k_ReD.count(0)}')
                    # print(f'NVA: index:{statistic_top_k_NVA.index(max(statistic_top_k_NVA))},'
                    #       f'percentage: {max(statistic_top_k_NVA)/sum(statistic_top_k_NVA)},'
                    #       f'count: {len(statistic_top_k_NVA)-statistic_top_k_NVA.count(0)}')
                    # print(f'NVD: index:{statistic_top_k_NVD.index(max(statistic_top_k_NVD))},'
                    #       f'percentage: {max(statistic_top_k_NVD)/sum(statistic_top_k_NVD)},'
                    #       f'count: {len(statistic_top_k_NVD)-statistic_top_k_NVD.count(0)}')

                    topk_neurons[category]['ReA'].append(len(statistic_top_k_ReA) - statistic_top_k_ReA.count(0))
                    topk_neurons[category]['ReD'].append(len(statistic_top_k_ReD) - statistic_top_k_ReD.count(0))
                    topk_neurons[category]['NVA'].append(len(statistic_top_k_NVA) - statistic_top_k_NVA.count(0))
                    topk_neurons[category]['NVD'].append(len(statistic_top_k_NVD) - statistic_top_k_NVD.count(0))

            print(f'Statistic of Selectivity: ')
            for i in range(neuron_number_list[k]):
                print(f'{i}th / {neuron_number_list[k]}: ')
                for c in range(categories):
                    print(f'class {c} : {statistic_selectivity[i][c]}')

        file = open(f'./data/{id_dataset}/layer_{layers[k]}_topk_neurons.pkl', 'wb')
        pickle.dump(topk_neurons, file)

    return combination_abstraction_dict


if __name__ == '__main__':

    # dataset = 'mnist'
    # model_path = './models/lenet_mnist/'
    # detector_path = './data/mnist/detector/'
    # x_train, y_train, x_test, y_test = load_mnist()
    # research_layers = {'fully_connected_layers_number': 1,
    #                    'fully_connected_layers': [-6],
    #                    'neuron_number_of_fully_connected_layers': [80]}

    dataset = 'cifar10'
    model_path = './models/vgg19_cifar10/'
    detector_path = './data/cifar10/detector/'
    x_train, y_train, x_test, y_test = load_cifar10()
    research_layers = {'fully_connected_layers_number': 1,
                       'fully_connected_layers': [-5],
                       'neuron_number_of_fully_connected_layers': [512]}

    # if not os.path.exists(f'./data/{dataset}/layer_-6_topk_neurons.pkl'):
    train_picture_classified = classify_id_pictures(id_dataset=dataset, dataset=x_train,
                                                    labels=tf.argmax(y_train, axis=1), model_path=model_path)
    test_picture_classified = classify_id_pictures(id_dataset=dataset, dataset=x_test,
                                                   labels=tf.argmax(y_test, axis=1), model_path=model_path)
    print('\nGet neural value of train dataset:')
    train_picture_neural_value = get_neural_value(id_dataset=dataset, model_path=model_path,
                                                  pictures_classified=train_picture_classified)
    print('\nGet neural value of train dataset:')
    test_picture_neural_value = get_neural_value(id_dataset=dataset, model_path=model_path,
                                                 pictures_classified=test_picture_classified)
    print('\nStatistic of train data neural value:')
    train_nv_statistic = statistic_of_neural_value(id_dataset=dataset,
                                                   neural_value=train_picture_neural_value)
    print('finished!')

    train_picture_combination_abstraction = encode_abstraction_for_statistic(
        id_dataset=dataset, neural_value=test_picture_neural_value,
        train_neural_value_statistic=train_nv_statistic, statistic_layers=research_layers)

    # topk_file = open(f'./data/mnist/layer_-6_topk_neurons.pkl', 'rb')
    # topk_file_content = pickle.load(topk_file)
    # topk_neurons_by_class = topk_file_content[9]
    #
    # x = [i for i in range(1, 31)]
    # plt.plot(x, topk_neurons_by_class['ReA'][:30], color='red', marker='o',
    #          linestyle='-', label='ReA')
    # plt.plot(x, topk_neurons_by_class['NVA'][:30], color='red', marker='^',
    #          linestyle='--', label='NA')
    # plt.plot(x, topk_neurons_by_class['ReD'][:30], color='green', marker='o',
    #          linestyle='-', label='ReD')
    # plt.plot(x, topk_neurons_by_class['NVD'][:30], color='green', marker='^',
    #          linestyle='--', label='ND')
    # plt.legend(prop={'size': 15})
    # plt.tick_params(labelsize=15)
    # plt.title('Class 9', size=18, x=0.5, y=-0.2)
    # plt.show()














