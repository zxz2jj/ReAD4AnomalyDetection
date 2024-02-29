import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import auc, roc_curve
from sklearn.manifold import TSNE

from global_config import num_of_labels, cnn_config
import global_config


def classify_id_pictures(id_dataset, dataset, labels, model_path):
    """
    divide the in-distribution dataset into correct predictions and wrong predictions.
    :return:
    """

    model = tf.keras.models.load_model(model_path+'tf_model.h5')

    print("\nclassify dataset:")
    picture_classified = {}
    for category in range(num_of_labels[id_dataset]):
        pictures_of_category = dataset[labels == category]
        prediction = np.argmax(model.predict(pictures_of_category), axis=1)
        correct_number = np.sum(prediction == category)
        wrong_number = np.sum(prediction != category)

        correct_pictures = pictures_of_category[prediction == category]
        wrong_pictures = pictures_of_category[prediction != category]
        correct_prediction = prediction[prediction == category]
        wrong_prediction = prediction[prediction != category]

        temp_dict = {'correct_pictures': correct_pictures, 'correct_prediction': correct_prediction,
                     'wrong_pictures': wrong_pictures, 'wrong_prediction': wrong_prediction}
        picture_classified[category] = temp_dict
        print('category {}: {} correct predictions, {} wrong predictions.'.format(category, correct_number,
                                                                                  wrong_number))
    return picture_classified


def get_neural_value(id_dataset, model_path, pictures_classified):

    model = tf.keras.models.load_model(model_path+'tf_model.h5')
    layers = cnn_config[id_dataset]['layers_of_getting_value']

    neural_value_dict = {}
    for layer in layers:
        print('\nget neural value in layer: {}'.format(layer))
        get_layer_output = tf.keras.backend.function(inputs=model.layers[0].input, outputs=model.layers[layer].output)
        neural_value_layer_dict = {}
        for category in range(num_of_labels[id_dataset]):
            print('\rcategory: {} / {}'.format(category+1, num_of_labels[id_dataset]), end='')
            neural_value_category = {}
            correct_pictures = pictures_classified[category]['correct_pictures']
            wrong_pictures = pictures_classified[category]['wrong_pictures']
            if correct_pictures.shape[0] != 0:
                # correct_pictures_layer_output = get_layer_output([correct_pictures])
                correct_pictures_layer_output = list()
                batch_size = 128
                for i in range(0, len(correct_pictures), batch_size):
                    batch = correct_pictures[i:i + batch_size]
                    batch_output = get_layer_output([batch])
                    correct_pictures_layer_output.append(batch_output)
                neural_value_category['correct_pictures'] = np.concatenate(correct_pictures_layer_output)
                neural_value_category['correct_prediction'] = pictures_classified[category]['correct_prediction']
            else:
                neural_value_category['correct_pictures'] = np.array([])
                neural_value_category['correct_prediction'] = np.array([])

            if wrong_pictures.shape[0] != 0:
                # wrong_pictures_layer_output = get_layer_output([wrong_pictures])
                wrong_pictures_layer_output = list()
                batch_size = 128
                for i in range(0, len(wrong_pictures), batch_size):
                    batch = wrong_pictures[i:i + batch_size]
                    batch_output = get_layer_output([batch])
                    wrong_pictures_layer_output.append(batch_output)
                neural_value_category['wrong_pictures'] = np.concatenate(wrong_pictures_layer_output)
                neural_value_category['wrong_prediction'] = pictures_classified[category]['wrong_prediction']
            else:
                neural_value_category['wrong_pictures'] = np.array([])
                neural_value_category['wrong_prediction'] = np.array([])

            neural_value_layer_dict[category] = neural_value_category
        neural_value_dict[layer] = neural_value_layer_dict

    return neural_value_dict


def statistic_of_neural_value(id_dataset, neural_value):
    """
    计算某神经元在非第i类上输出均值
    non_class_l_average[l][k]表示第k个神经元在非第l类上的输出均值
    :param
    :return: non_class_l_average[] all_class_average
    """

    layers = cnn_config[id_dataset]['layers_of_getting_value']
    neuron_number_of_each_layer = cnn_config[id_dataset]['neurons_of_each_layer']
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
            selective[r] = (image_neural_value[r] - not_class_l_average[label][r]) / abs(all_class_average[r])

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

    layers = cnn_config[id_dataset]['layers_of_getting_value']
    neuron_number_list = cnn_config[id_dataset]['neurons_of_each_layer']
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
            if neural_value_category['correct_pictures'].shape[0] != 0:
                for image, label in \
                        zip(neural_value_category['correct_pictures'], neural_value_category['correct_prediction']):
                    combination = encode_by_selective(image, label, global_config.selective_rate, neuron_number_list[k], non_class_l_avg, all_class_avg)
                    correct_abstraction_list.append(combination)
            else:
                correct_abstraction_list = np.array([])

            if neural_value_category['wrong_pictures'].shape[0] != 0:
                for image, label in \
                        zip(neural_value_category['wrong_pictures'],  neural_value_category['wrong_prediction']):
                    combination = encode_by_selective(image, label, global_config.selective_rate, neuron_number_list[k], non_class_l_avg, all_class_avg)
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


def concatenate_data_between_layers(id_dataset, data_dict):
    layers = cnn_config[id_dataset]['layers_of_getting_value']
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
            abstraction_length = sum(cnn_config[id_dataset]['neurons_of_each_layer'])
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
            abstraction_length = sum(cnn_config[id_dataset]['neurons_of_each_layer'])
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
                                                distance_of_test_data[category]['correct_prediction']):
                    if distance > confidence_boundary[prediction]:
                        fp += 1
                    else:
                        tn += 1

        if distance_of_test_data[category]['wrong_pictures'] is not None:
            if not distance_of_test_data[category]['wrong_pictures']:
                pass
            else:
                for distance, prediction in zip(distance_of_test_data[category]['wrong_pictures'],
                                                distance_of_test_data[category]['wrong_prediction']):
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


def sk_auc(id_dataset, distance_of_test_data, distance_of_bad_data):
    auc_list = []
    far95_list = []
    for c in range(global_config.num_of_labels[id_dataset]):
        if not distance_of_bad_data[c]['wrong_pictures']:
            print('{}/{}: AUROC: {}'.format(c, global_config.num_of_labels[id_dataset], 'No examples'))
            continue
        y_true = [0 for _ in range(len(distance_of_test_data[c]['correct_pictures']))] + \
                 [1 for _ in range(len(distance_of_bad_data[c]['wrong_pictures']))]
        y_score = distance_of_test_data[c]['correct_pictures'] + distance_of_bad_data[c]['wrong_pictures']
        fpr, tpr, threshold = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        auc_list.append(roc_auc)
        target_tpr = 0.95
        target_threshold_idx = np.argmax(tpr >= target_tpr)
        target_fpr = fpr[target_threshold_idx]
        far95_list.append(target_fpr)
        print('{}/{}: AUROC: {:.6f}, FAR95: {:.6f}, TPR: {}'.
              format(c, global_config.num_of_labels[id_dataset], roc_auc, target_fpr, tpr[target_threshold_idx]))
    print('avg-AUROC: {:.6f}, avg-FAR95: {:.6f}'.format(np.average(auc_list), np.average(far95_list)))
    return


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
