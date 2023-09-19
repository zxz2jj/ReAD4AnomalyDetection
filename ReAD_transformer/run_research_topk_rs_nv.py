import os.path
import numpy as np
import math
import pickle

from global_config import num_of_labels, selective_rate
from ReAD import statistic_of_neural_value
import matplotlib.pyplot as plt


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

    dict_neuronvalue = {}
    for index in range(len(selective)):
        dict_neuronvalue[index] = image_neural_value[index]
    sort_by_neuronvalue = sorted(dict_neuronvalue.items(), key=lambda x: x[1])
    top_k_NVA_index = []
    top_k_NVD_index = []
    top_k_NVA_index.extend([sort_by_neuronvalue[-k][0] for k in range(1, topk+1, 1)])
    top_k_NVD_index.extend([sort_by_neuronvalue[k][0] for k in range(topk)])

    return combination_code, only_active_code, only_deactive_code, \
        top_k_ReA_index, top_k_ReD_index, top_k_NVA_index, top_k_NVD_index, selective


def encode_ReAD_for_statistic(id_dataset, neural_value, train_neural_value_statistic, statistic_layers):

    categories = num_of_labels[id_dataset]
    encode_layer_number = statistic_layers['fully_connected_layers_number']
    layers = statistic_layers['fully_connected_layers']
    neuron_number_list = statistic_layers['neuron_number_of_fully_connected_layers']

    combination_abstraction_dict = {}
    for k in range(encode_layer_number):
        print("\nEncoding in fully connected layer:{}.".format(layers[k]))

        non_class_l_avg = train_neural_value_statistic[layers[k]]['not_class_l_average']
        all_class_avg = train_neural_value_statistic[layers[k]]['all_class_average']

        statistic_selectivity = np.zeros([80, 10, 2])

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
        for topk in range(1, 40):
            print(f'----------------------------top:{topk}----------------------------------------------')
            for category in range(categories):
                print('category: {} / {}'.format(category+1, categories))
                neural_value_category = neural_value[layers[k]][category]

                statistic_top_k_ReA = [0 for _ in range(neuron_number_list[k])]
                statistic_top_k_ReD = [0 for _ in range(neuron_number_list[k])]
                statistic_top_k_NVA = [0 for _ in range(neuron_number_list[k])]
                statistic_top_k_NVD = [0 for _ in range(neuron_number_list[k])]

                if neural_value_category['correct_examples'] is not None:
                    if neural_value_category['correct_examples'].shape[0] == 0:
                        pass
                    else:
                        for image, label in \
                                zip(neural_value_category['correct_examples'], neural_value_category['correct_predictions']):
                            combination_code, only_active_code, only_deactive_code, \
                                topk_ReA_index, topk_ReD_index, topk_NVA_index, topk_NVD_index, selectivity = \
                                encode_by_selective(image, label, selective_rate, neuron_number_list[k],
                                                    non_class_l_avg, topk, all_class_avg)

                            for index in topk_ReA_index:
                                statistic_top_k_ReA[index] += 1
                            for index in topk_ReD_index:
                                statistic_top_k_ReD[index] += 1
                            for index in topk_NVA_index:
                                statistic_top_k_NVA[index] += 1
                            for index in topk_NVD_index:
                                statistic_top_k_NVD[index] += 1
                            for s in range(len(selectivity)):
                                if selectivity[s] > 0:
                                    statistic_selectivity[s][category][0] += 1
                                else:
                                    statistic_selectivity[s][category][1] += 1

                    print(f'ReA: index:{statistic_top_k_ReA.index(max(statistic_top_k_ReA))},'
                          f'percentage: {max(statistic_top_k_ReA)/sum(statistic_top_k_ReA)},'
                          f'count: {len(statistic_top_k_ReA)-statistic_top_k_ReA.count(0)}')
                    print(f'ReD: index:{statistic_top_k_ReD.index(max(statistic_top_k_ReD))},'
                          f'percentage: {max(statistic_top_k_ReD)/sum(statistic_top_k_ReD)},'
                          f'count: {len(statistic_top_k_ReD)-statistic_top_k_ReD.count(0)}')
                    print(f'NVA: index:{statistic_top_k_NVA.index(max(statistic_top_k_NVA))},'
                          f'percentage: {max(statistic_top_k_NVA)/sum(statistic_top_k_NVA)},'
                          f'count: {len(statistic_top_k_NVA)-statistic_top_k_NVA.count(0)}')
                    print(f'NVD: index:{statistic_top_k_NVD.index(max(statistic_top_k_NVD))},'
                          f'percentage: {max(statistic_top_k_NVD)/sum(statistic_top_k_NVD)},'
                          f'count: {len(statistic_top_k_NVD)-statistic_top_k_NVD.count(0)}')

                    topk_neurons[category]['ReA'].append(len(statistic_top_k_ReA) - statistic_top_k_ReA.count(0))
                    topk_neurons[category]['ReD'].append(len(statistic_top_k_ReD) - statistic_top_k_ReD.count(0))
                    topk_neurons[category]['NVA'].append(len(statistic_top_k_NVA) - statistic_top_k_NVA.count(0))
                    topk_neurons[category]['NVD'].append(len(statistic_top_k_NVD) - statistic_top_k_NVD.count(0))

        file = open(f'./data/{id_dataset}/layer_fc_topk_neurons.pkl', 'wb')
        pickle.dump(topk_neurons, file)

    return combination_abstraction_dict


if __name__ == '__main__':

    dataset = 'imdb'
    hidden_states_path = f'./data/{dataset}/hidden_states/'
    detector_path = f'./data/{dataset}/detector/'
    num_of_category = num_of_labels[dataset]

    if not os.path.exists(f'./data/{dataset}/layer_fc_topk_neurons.pkl'):
        if os.path.exists(detector_path + 'train_nv_statistic.pkl'):
            train_nv_statistic = np.load(detector_path + 'train_nv_statistic.pkl')
        else:
            layer_neural_value = {}
            train_neural_value = {-1: layer_neural_value}
            train_fc_hidden_states = np.load(hidden_states_path+'train_fc_hidden_states.npy')
            train_predictions = np.load(hidden_states_path+'train_predictions.npy')
            for i in range(num_of_category):
                print('Class {}: {}'.format(i, sum(train_predictions == i)))
                hidden_states_category = {'correct_examples': train_fc_hidden_states[(train_predictions == i)],
                                          'correct_predictions': train_predictions[(train_predictions == i)],
                                          'wrong_examples': None,
                                          'wrong_predictions': None}
                layer_neural_value[i] = hidden_states_category

            print('\nStatistic of train data neural value:')
            train_nv_statistic = statistic_of_neural_value(dataset, train_neural_value)

        layer_hidden_states = {}
        test_hidden_states = {-1: layer_hidden_states}
        test_fc_hidden_states = np.load(hidden_states_path + 'test_fc_hidden_states.npy')
        test_predictions = np.load(hidden_states_path + 'test_predictions.npy')
        for i in range(num_of_category):
            print('Class {}: {}'.format(i, sum(test_predictions == i)))
            hidden_states_category = {'correct_examples': test_fc_hidden_states[(test_predictions == i)],
                                      'correct_predictions': test_predictions[(test_predictions == i)],
                                      'wrong_examples': None,
                                      'wrong_predictions': None}
            layer_hidden_states[i] = hidden_states_category

        research_layers = {'fully_connected_layers_number': 1,
                           'fully_connected_layers': [-2],
                           'neuron_number_of_fully_connected_layers': [768]}

        test_ReAD_abstraction = encode_ReAD_for_statistic(id_dataset=dataset, neural_value=test_hidden_states,
                                                          train_neural_value_statistic=train_nv_statistic,
                                                          statistic_layers=research_layers)
    else:
        topk_file = open(f'./data/{dataset}/layer_-1_topk_neurons.pkl', 'rb')
        topk_file_content = pickle.load(topk_file)
        topk_neurons_by_class = topk_file_content[1]

        x = [i for i in range(1, 31)]
        plt.plot(x, topk_neurons_by_class['ReA'][:30], color='red', marker='o',
                 linestyle='-', label='ReA')
        plt.plot(x, topk_neurons_by_class['NVA'][:30], color='red', marker='^',
                 linestyle='--', label='NA')
        plt.plot(x, topk_neurons_by_class['ReD'][:30], color='green', marker='o',
                 linestyle='-', label='ReD')
        plt.plot(x, topk_neurons_by_class['NVD'][:30], color='green', marker='^',
                 linestyle='--', label='ND')
        plt.legend(prop={'size': 15})
        plt.tick_params(labelsize=15)
        plt.title('Class 1', size=18, x=0.5, y=-0.2)
        plt.show()














