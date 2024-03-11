import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, roc_curve
from load_data import (load_mnist, load_fmnist, load_cifar10, load_cifar100, load_gtsrb, load_svhn,
                       load_clean_adv_data, load_ood_data)
from ReAD import classify_id_pictures
from global_config import num_of_labels, cnn_config
import global_config


def get_confidence(dataset_name, checkpoint_path, pictures_classified):

    model = tf.keras.models.load_model(checkpoint_path+'tf_model.h5')
    confidence_dict = {}
    for category in range(num_of_labels[dataset_name]):
        confidence_category = {}
        correct_pictures = pictures_classified[category]['correct_pictures']
        wrong_pictures = pictures_classified[category]['wrong_pictures']
        if correct_pictures.shape[0] != 0:
            batch_size = 128
            batch_confidence = list()
            for i in range(0, len(correct_pictures), batch_size):
                batch = correct_pictures[i:i + batch_size]
                batch_output = model([batch]).numpy()[:, category]
                batch_confidence.append(1-batch_output)
            confidence_category['correct_pictures'] = np.concatenate(batch_confidence)
            confidence_category['correct_prediction'] = pictures_classified[category]['correct_prediction']
        else:
            confidence_category['correct_pictures'] = np.array([])
            confidence_category['correct_prediction'] = np.array([])
        if wrong_pictures.shape[0] != 0:
            batch_size = 128
            batch_confidence = list()
            for i in range(0, len(wrong_pictures), batch_size):
                batch = wrong_pictures[i:i + batch_size]
                batch_output = model([batch]).numpy()[:, category]
                batch_confidence.append(1-batch_output)
            confidence_category['wrong_pictures'] = np.concatenate(batch_confidence)
            confidence_category['wrong_prediction'] = pictures_classified[category]['wrong_prediction']
        else:
            confidence_category['wrong_pictures'] = np.array([])
            confidence_category['wrong_prediction'] = np.array([])

        confidence_dict[category] = confidence_category

    return confidence_dict


def sk_auc(dataset_name, distance_of_test_data, distance_of_bad_data):
    auc_list = []
    far95_list = []
    for c in range(global_config.num_of_labels[dataset_name]):
        if distance_of_bad_data[c]['wrong_pictures'].shape[0] == 0:
            print('{}/{}: AUROC: {}'.format(c, global_config.num_of_labels[dataset_name], 'No examples'))
            continue
        y_true = [0 for _ in range(len(distance_of_test_data[c]['correct_pictures']))] + \
                 [1 for _ in range(len(distance_of_bad_data[c]['wrong_pictures']))]
        y_score = (distance_of_test_data[c]['correct_pictures'].tolist() +
                   distance_of_bad_data[c]['wrong_pictures'].tolist())
        fpr, tpr, threshold = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        auc_list.append(roc_auc)
        target_tpr = 0.95
        target_threshold_idx = np.argmax(tpr >= target_tpr)
        target_fpr = fpr[target_threshold_idx]
        far95_list.append(target_fpr)
        print('{}/{}: AUROC: {:.6f}, FAR95: {:.6f}, TPR: {:.6f}, OOD Examples: {}'.
              format(c, global_config.num_of_labels[dataset_name], roc_auc, target_fpr, tpr[target_threshold_idx],
                     len(distance_of_bad_data[c]['wrong_pictures'])))
    print('avg-AUROC: {:.6f}, avg-FAR95: {:.6f}'.format(np.average(auc_list), np.average(far95_list)))
    return


if __name__ == "__main__":

    print('\n********************** Evaluate OOD Detection ****************************')
    id_dataset = 'mnist'
    model_path = './models/lenet_mnist/'
    detector_path = './data/mnist/detector/'
    x_train, y_train, x_test, y_test = load_mnist()
    #
    # id_dataset = 'fmnist'
    # model_path = './models/lenet_fmnist/'
    # detector_path = './data/fmnist/detector/'
    # x_train, y_train, x_test, y_test = load_fmnist()
    #
    # id_dataset = 'svhn'
    # model_path = './models/resnet18_svhn/'
    # detector_path = './data/svhn/detector/'
    # x_train, y_train, x_test, y_test = load_svhn()
    #
    # id_dataset = 'cifar100'
    # model_path = './models/resnet18_cifar100/'
    # detector_path = './data/cifar100/detector/'
    # x_train, y_train, x_test, y_test = load_cifar100()
    #
    # id_dataset = 'cifar10'
    # model_path = './models/vgg19_cifar10/'
    # detector_path = './data/cifar10/detector/'
    # x_train, y_train, x_test, y_test = load_cifar10()
    #
    # id_dataset = 'gtsrb'
    # model_path = './models/vgg19_gtsrb/'
    # detector_path = './data/gtsrb/detector/'
    # x_train, y_train, x_test, y_test = load_gtsrb()

    test_picture_classified = classify_id_pictures(id_dataset=id_dataset, dataset=x_test,
                                                   labels=tf.argmax(y_test, axis=1), model_path=model_path)
    test_confidence = get_confidence(dataset_name=id_dataset, checkpoint_path=model_path,
                                     pictures_classified=test_picture_classified)

    OOD_dataset = cnn_config[id_dataset]['ood_settings']
    for ood in OOD_dataset:
        print('\n************Evaluating*************')
        print(f'In-Distribution Data: {id_dataset}, Out-of-Distribution Data: {ood}.')
        ood_data, number_of_ood = load_ood_data(ood_dataset=ood, id_model_path=model_path,
                                                num_of_categories=num_of_labels[id_dataset])

        print('\nGet neural value of ood dataset...')
        ood_confidence = get_confidence(dataset_name=id_dataset, checkpoint_path=model_path,
                                        pictures_classified=ood_data)
        sk_auc(id_dataset, test_confidence, ood_confidence)
        print('*************************************')

    print('\n********************** Evaluate Adversarial Detection ****************************')

    # clean_dataset = 'fmnist'
    # model_path = './models/lenet_fmnist/'

    # clean_dataset = 'svhn'
    # model_path = './models/resnet18_svhn/'

    clean_dataset = 'cifar10'
    model_path = './models/vgg19_cifar10/'

    # adversarial_attacks = ['ba']
    adversarial_attacks = cnn_config[clean_dataset]['adversarial_settings']
    for attack in adversarial_attacks:
        print(f'\nATTACK: {attack}')
        clean_data_classified, adv_data_classified, num_of_test = \
            load_clean_adv_data(id_dataset=clean_dataset, attack=attack, num_of_categories=num_of_labels[clean_dataset])

        print('\nGet neural value:')
        print('CLEAN dataset ...')
        clean_data_confidence = get_confidence(dataset_name=clean_dataset, checkpoint_path=model_path,
                                               pictures_classified=clean_data_classified)
        print('ADVERSARIAL dataset ...')
        adv_data_confidence = get_confidence(dataset_name=clean_dataset, checkpoint_path=model_path,
                                             pictures_classified=adv_data_classified)
        sk_auc(clean_dataset, clean_data_confidence, adv_data_confidence)

