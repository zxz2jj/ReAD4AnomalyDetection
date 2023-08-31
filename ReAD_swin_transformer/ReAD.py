import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import auc, roc_curve
from sklearn.manifold import TSNE
from torch.utils.data import Subset

from global_config import num_of_labels, swin_config
import global_config
from model_swin_transformer import SwinForImageClassification
from train_swin_models import image_tokenizer


def get_neural_value(id_dataset, dataset, checkpoint, is_ood=False):
    model = SwinForImageClassification.from_pretrained(
        checkpoint,
        output_hidden_states=False,
        ignore_mismatched_sizes=True,
    )

    dataset = image_tokenizer(data=dataset, model_checkpoint=checkpoint, mode='test')
    dataset = Subset(dataset, range(0, dataset.shape[0])).__getitem__([_ for _ in range(dataset.shape[0])])
    # dataset = Subset(dataset, range(0, 50)).__getitem__([_ for _ in range(50)])

    layers = swin_config[id_dataset]['layers_of_getting_value']

    neural_value_dict = {}
    for layer in layers:
        neural_value_category = {}
        for category in range(num_of_labels[id_dataset]):
            neural_value_category[category] = {'correct_pictures': [], 'correct_predictions': [],
                                               'wrong_pictures': [], 'wrong_predictions': []}
        neural_value_dict[layer] = neural_value_category

    neural_value_list = list()
    predictions_list = list()
    # model.cuda()
    for i, batch in zip(range(len(dataset['pixel_values'])), dataset['pixel_values']):
        # torch.cuda.empty_cache()
        print('\rImage: {} / {}'.format(i + 1, len(dataset['pixel_values'])), end='')
        outputs = model(torch.unsqueeze(batch, 0))
        prediction = torch.argmax(outputs.logits, 1)
        neural_value_list.append(model.get_pooled_output().cpu().detach().numpy())
        predictions_list.append(prediction)

    for layer in layers:
        print('\nget neural value in layer: {}'.format(layer))
        if not is_ood:
            labels = dataset['label']
            for nv, prediction, label in zip(neural_value_list, predictions_list, labels):
                if prediction == label:
                    neural_value_dict[layer][int(prediction)]['correct_pictures'].append(nv)
                    neural_value_dict[layer][int(prediction)]['correct_predictions'].append(prediction)
                else:
                    neural_value_dict[layer][int(prediction)]['wrong_pictures'].append(nv)
                    neural_value_dict[layer][int(prediction)]['wrong_predictions'].append(prediction)
        elif is_ood:
            for nv, prediction in zip(neural_value_list, predictions_list):
                neural_value_dict[layer][int(prediction)]['wrong_pictures'].append(nv)
                neural_value_dict[layer][int(prediction)]['wrong_predictions'].append(prediction)

    result_dict = {}
    for layer in layers:
        result_dict[layer] = {}
        for category in range(num_of_labels[id_dataset]):
            result_dict[layer][category] = {}
            number_of_correct = neural_value_dict[layer][category]['correct_pictures'].__len__()
            number_of_wrong = neural_value_dict[layer][category]['wrong_pictures'].__len__()
            print(f'category {category}: {number_of_correct} correct predictions, {number_of_wrong} wrong predictions.')
            if number_of_correct != 0:
                result_dict[layer][category]['correct_pictures'] = np.concatenate(neural_value_dict[layer][category]['correct_pictures'])
                result_dict[layer][category]['correct_predictions'] = np.concatenate(neural_value_dict[layer][category]['correct_predictions'])
            else:
                result_dict[layer][category]['correct_pictures'] = np.array([])
                result_dict[layer][category]['correct_predictions'] = np.array([])
            if number_of_wrong != 0:
                result_dict[layer][category]['wrong_pictures'] = np.concatenate(neural_value_dict[layer][category]['wrong_pictures'])
                result_dict[layer][category]['wrong_predictions'] = np.concatenate(neural_value_dict[layer][category]['wrong_predictions'])
            else:
                result_dict[layer][category]['wrong_pictures'] = np.array([])
                result_dict[layer][category]['wrong_predictions'] = np.array([])

    return result_dict


def statistic_of_neural_value(id_dataset, neural_value):
    """
    计算某神经元在非第i类上输出均值
    non_class_l_average[l][k]表示第k个神经元在非第l类上的输出均值
    :param
    :return: non_class_l_average[] all_class_average
    """

    layers = swin_config[id_dataset]['layers_of_getting_value']
    neuron_number_of_each_layer = swin_config[id_dataset]['neurons_of_each_layer']
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


def encode_abstraction(id_dataset, neural_value, train_dataset_statistic):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # --------------------------selective  = [ output - average(-l) ] / average(all)------------------------# #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    layers = swin_config[id_dataset]['layers_of_getting_value']
    neuron_number_list = swin_config[id_dataset]['neurons_of_each_layer']
    encode_layer_number = len(layers)
    print(f'selective rate: {global_config.selective_rate}')
    abstractions_dict = {}
    for k in range(encode_layer_number):
        print("\nEncoding in fully connected layer:{}.".format(layers[k]))
        non_class_l_avg = train_dataset_statistic[layers[k]]['not_class_l_average']
        all_class_avg = train_dataset_statistic[layers[k]]['all_class_average']

        combination_abstraction_each_category = {}
        for category in range(num_of_labels[id_dataset]):
            print('\rcategory: {} / {}'.format(category+1, num_of_labels[id_dataset]), end='')
            neural_value_category = neural_value[layers[k]][category]
            correct_abstraction_list = []
            wrong_abstraction_list = []

            if neural_value_category['correct_pictures'].shape[0] == 0:
                correct_abstraction_list = []
            else:
                for image, label in \
                        zip(neural_value_category['correct_pictures'], neural_value_category['correct_predictions']):
                    combination = encode_by_selective(image, label, global_config.selective_rate, neuron_number_list[k], non_class_l_avg, all_class_avg)
                    correct_abstraction_list.append(combination)

            if neural_value_category['wrong_pictures'].shape[0] == 0:
                wrong_abstraction_list = []
            else:
                for image, label in \
                        zip(neural_value_category['wrong_pictures'],  neural_value_category['wrong_predictions']):
                    combination = encode_by_selective(image, label, global_config.selective_rate, neuron_number_list[k], non_class_l_avg, all_class_avg)
                    wrong_abstraction_list.append(combination)


            combination_abstraction = {'correct_pictures': correct_abstraction_list,
                                       'correct_predictions': neural_value_category['correct_predictions'],
                                       'wrong_pictures': wrong_abstraction_list,
                                       'wrong_predictions': neural_value_category['wrong_predictions']}
            combination_abstraction_each_category[category] = combination_abstraction

        abstractions_dict[layers[k]] = combination_abstraction_each_category
    print('\n')

    return abstractions_dict


def concatenate_data_between_layers(id_dataset, data_dict):
    layers = swin_config[id_dataset]['layers_of_getting_value']
    abstraction_concatenated_dict = {}
    for category in range(num_of_labels[id_dataset]):
        correct_data = []
        if data_dict[layers[0]][category]['correct_pictures']:
            for k in range(len(layers)):
                correct_data.append(data_dict[layers[k]][category]['correct_pictures'])
            correct_data = np.concatenate(correct_data, axis=1)
        else:
            correct_data = np.array([])

        wrong_data = []
        if data_dict[layers[0]][category]['wrong_pictures']:
            for k in range(len(layers)):
                wrong_data.append(data_dict[layers[k]][category]['wrong_pictures'])
            wrong_data = np.concatenate(wrong_data, axis=1)
        else:
            wrong_data = np.array([])

        concatenate_data = {'correct_pictures': correct_data,
                            'correct_predictions': data_dict[layers[0]][category]['correct_predictions'],
                            'wrong_pictures': wrong_data,
                            'wrong_predictions': data_dict[layers[0]][category]['wrong_predictions']}

        abstraction_concatenated_dict[category] = concatenate_data

    return abstraction_concatenated_dict


def k_means(data, category_number):
    """
    对组合模式求均值并计算各类指标
    :param data:
    :param category_number:
    :return:
    """
    combination_abstraction = []
    number_of_categories = []
    for category in range(category_number):
        combination_abstraction.append(data[category]['correct_pictures'])
        number_of_categories.append(data[category]['correct_pictures'].shape[0])
    combination_abstraction = np.concatenate(combination_abstraction, axis=0)

    estimator = KMeans(n_clusters=category_number)
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

    return centers_resorted


def statistic_distance(id_dataset, abstractions, cluster_centers):

    euclidean_distance = {}
    for category in range(num_of_labels[id_dataset]):
        abstraction_category = abstractions[category]
        distance_category = {}
        if abstraction_category['correct_pictures'] is not None:
            correct_distance = []
            abstraction_length = sum(swin_config[id_dataset]['neurons_of_each_layer'])
            if abstraction_category['correct_pictures'].shape[0] == 0:
                distance_category['correct_pictures'] = correct_distance
                distance_category['correct_predictions'] = abstraction_category['correct_predictions']
            else:
                for abstraction, prediction in zip(abstraction_category['correct_pictures'],
                                                   abstraction_category['correct_predictions']):
                    distance = 0.0
                    for k in range(abstraction_length):
                        distance += pow(abstraction[k] - cluster_centers[prediction][k], 2)
                    correct_distance.append(distance ** 0.5)
                distance_category['correct_pictures'] = correct_distance
                distance_category['correct_predictions'] = abstraction_category['correct_predictions']

        else:
            distance_category['correct_pictures'] = []
            distance_category['correct_predictions'] = []

        if abstraction_category['wrong_pictures'] is not None:
            wrong_distance = []
            abstraction_length = sum(swin_config[id_dataset]['neurons_of_each_layer'])
            if abstraction_category['wrong_pictures'].shape[0] == 0:
                distance_category['wrong_pictures'] = wrong_distance
                distance_category['wrong_predictions'] = abstraction_category['wrong_predictions']
            else:
                for abstraction, prediction in zip(abstraction_category['wrong_pictures'],
                                                   abstraction_category['wrong_predictions']):
                    distance = 0.0
                    for k in range(abstraction_length):
                        distance += pow(abstraction[k] - cluster_centers[prediction][k], 2)
                    wrong_distance.append(distance ** 0.5)
                distance_category['wrong_pictures'] = wrong_distance
                distance_category['wrong_predictions'] = abstraction_category['wrong_predictions']

        else:
            distance_category['wrong_pictures'] = []
            distance_category['wrong_predictions'] = []

        euclidean_distance[category] = distance_category

    return euclidean_distance


def tp_fn_tn_fp(distance_of_train_data, percentile_of_confidence_boundary, distance_of_test_data, num_of_category):
    confidence_boundary = []
    if percentile_of_confidence_boundary >= 100:
        for category in range(num_of_category):
            confidence_boundary.append(np.max(distance_of_train_data[category]['correct_pictures']) *
                                       (1 + (percentile_of_confidence_boundary-100) / 100))
    else:
        for category in range(num_of_category):
            confidence_boundary.append(
                np.percentile(distance_of_train_data[category]['correct_pictures'], percentile_of_confidence_boundary))

    tp, fn, tn, fp = 0, 0, 0, 0

    for category in range(num_of_category):

        if distance_of_test_data[category]['correct_pictures'] is not None:
            if not distance_of_test_data[category]['correct_pictures']:
                pass
            else:
                for distance, prediction in zip(distance_of_test_data[category]['correct_pictures'],
                                                distance_of_test_data[category]['correct_predictions']):
                    if distance > confidence_boundary[prediction]:
                        fp += 1
                    else:
                        tn += 1

        if distance_of_test_data[category]['wrong_pictures'] is not None:
            if not distance_of_test_data[category]['wrong_pictures']:
                pass
            else:
                for distance, prediction in zip(distance_of_test_data[category]['wrong_pictures'],
                                                distance_of_test_data[category]['wrong_predictions']):
                    if distance > confidence_boundary[prediction]:
                        tp += 1
                    else:
                        fn += 1

    return tp, fn, tn, fp


def auroc(distance_of_train_data, distance_of_normal_data, distance_of_bad_data, num_of_category):
    fpr_list = [1.0]
    tpr_list = [1.0]
    fpr, tpr = 1.0, 1.0

    for percentile in range(100):
        tp_normal, fn_normal, tn_normal, fp_normal = \
            tp_fn_tn_fp(distance_of_train_data, percentile, distance_of_normal_data, num_of_category)
        tp_bad, fn_bad, tn_bad, fp_bad = \
            tp_fn_tn_fp(distance_of_train_data, percentile, distance_of_bad_data, num_of_category)

        tp = tp_normal + tp_bad
        fn = fn_normal + fn_bad
        tn = tn_normal + tn_bad
        fp = fp_normal + fp_bad

        if fp / (fp + tn) > fpr or tp / (tp + fn) > tpr:
            pass
        else:
            fpr = fp / (fp + tn)
            tpr = tp / (tp + fn)
            fpr_list.append(fpr)
            tpr_list.append(tpr)

    for times in range(100, 200):
        tp_normal, fn_normal, tn_normal, fp_normal = \
            tp_fn_tn_fp(distance_of_train_data, times, distance_of_normal_data, num_of_category)
        tp_bad, fn_bad, tn_bad, fp_bad = \
            tp_fn_tn_fp(distance_of_train_data, times, distance_of_bad_data, num_of_category)

        tp = tp_normal + tp_bad
        fn = fn_normal + fn_bad
        tn = tn_normal + tn_bad
        fp = fp_normal + fp_bad

        if fp / (fp + tn) > fpr or tp / (tp + fn) > tpr:
            pass
        else:
            fpr = fp / (fp + tn)
            tpr = tp / (tp + fn)
            fpr_list.append(fpr)
            tpr_list.append(tpr)

    fpr_list.append(0.0)
    tpr_list.append(0.0)
    fpr_list.reverse()
    tpr_list.reverse()

    # print(len(fpr_list))
    # print(fpr_list)
    # print(tpr_list)
    # plt.plot(fpr_list, tpr_list)
    # plt.show()

    fpr_list = np.array(fpr_list)
    tpr_list = np.array(tpr_list)
    auc_of_roc = auc(fpr_list, tpr_list)

    return auc_of_roc


def sk_auc(distance_of_test_data, distance_of_bad_data, num_of_category):

    auc_sum = 0
    count = 0
    for c in range(num_of_category):
        if not distance_of_bad_data[c]['wrong_pictures']:
            print('{}/{}: AUROC: {}'.format(c, num_of_category, 'No examples'))
            continue
        y_true = [0 for _ in range(len(distance_of_test_data[c]['correct_pictures']))] + \
                 [1 for _ in range(len(distance_of_bad_data[c]['wrong_pictures']))]
        y_score = distance_of_test_data[c]['correct_pictures'] + distance_of_bad_data[c]['wrong_pictures']
        fpr, tpr, threshold = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        print('{}/{}: AUROC: {:.6f}'.format(c, num_of_category, roc_auc))
        auc_sum += roc_auc
        count += 1

    return auc_sum/count


def t_sne_visualization(data, category_number):
    colour_label = []
    combination_abstraction = []
    for category in range(category_number):
        combination_abstraction.append(data[category]['correct_pictures'])
        colour_label.extend([category for _ in range(data[category]['correct_pictures'].shape[0])])

    combination_abstraction = np.concatenate(combination_abstraction, axis=0)
    embedded = TSNE(n_components=2).fit_transform(combination_abstraction)
    x_min, x_max = np.min(embedded, 0), np.max(embedded, 0)
    embedded = embedded / (x_max - x_min)
    plt.scatter(embedded[:, 0], embedded[:, 1],
                c=(np.array(colour_label)/10.0), s=1)
    plt.axis('off')
    plt.show()
    plt.close()
