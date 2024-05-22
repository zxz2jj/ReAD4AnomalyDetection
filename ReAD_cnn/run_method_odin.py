import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, roc_curve
from load_data import (load_mnist, load_fmnist, load_cifar10, load_cifar100, load_gtsrb, load_svhn,
                       load_clean_adv_data, load_ood_data)
from ReAD import classify_id_pictures
from global_config import num_of_labels, cnn_config
import global_config


def get_logits(id_dataset_name, checkpoint_path, pictures_classified, split='test', ood_dataset_name=None):

    model = tf.keras.models.load_model(checkpoint_path+'tf_model.h5')
    get_logits_output = tf.keras.backend.function(inputs=model.layers[0].input, outputs=model.layers[-2].output)

    logits_list = list()
    predictions_list = list()
    labels_list = list()

    if ood_dataset_name is None:
        for category in range(num_of_labels[id_dataset_name]):
            correct_pictures = pictures_classified[category]['correct_pictures']
            batch_size = 128
            batch_logits = list()
            for i in range(0, len(correct_pictures), batch_size):
                batch = correct_pictures[i:i + batch_size]
                batch_output = get_logits_output([batch])
                batch_logits.append(batch_output)
            logits_list.append(np.concatenate(batch_logits))
            predictions_list.append(pictures_classified[category]['correct_prediction'])
            labels_list.append(pictures_classified[category]['correct_prediction'])
        np.save(f'{id_dataset_name}_logits_{split}.npy', np.concatenate(logits_list))
        np.save(f'{id_dataset_name}_predictions_{split}.npy', np.concatenate(predictions_list))
        np.save(f'{id_dataset_name}_labels_{split}.npy', np.concatenate(labels_list))
    else:
        for category in range(num_of_labels[id_dataset_name]):
            if pictures_classified[category]['wrong_pictures'] is not None:
                wrong_pictures = pictures_classified[category]['wrong_pictures']
                batch_size = 128
                batch_logits = list()
                for i in range(0, len(wrong_pictures), batch_size):
                    batch = wrong_pictures[i:i + batch_size]
                    batch_output = get_logits_output([batch])
                    batch_logits.append(batch_output)
                logits_list.append(np.concatenate(batch_logits))
                predictions_list.append(pictures_classified[category]['wrong_prediction'])
        np.save(f'{id_dataset_name}-{ood_dataset_name}_logits.npy', np.concatenate(logits_list))
        np.save(f'{id_dataset_name}-{ood_dataset_name}_predictions.npy', np.concatenate(predictions_list))


if __name__ == "__main__":

    # id_dataset = 'mnist'
    # model_path = './models/lenet_mnist/'
    # detector_path = './data/mnist/detector/'
    # x_train, y_train, x_test, y_test = load_mnist()
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

    # id_dataset = 'cifar10'
    # model_path = './models/vgg19_cifar10/'
    # detector_path = './data/cifar10/detector/'
    # x_train, y_train, x_test, y_test = load_cifar10()
    #
    id_dataset = 'gtsrb'
    model_path = './models/vgg19_gtsrb/'
    detector_path = './data/gtsrb/detector/'
    x_train, y_train, x_test, y_test = load_gtsrb()
    #
    train_picture_classified = classify_id_pictures(id_dataset=id_dataset, dataset=x_train,
                                                    labels=tf.argmax(y_train, axis=1), model_path=model_path)
    get_logits(id_dataset_name=id_dataset, checkpoint_path=model_path, pictures_classified=train_picture_classified, split='train')

    test_picture_classified = classify_id_pictures(id_dataset=id_dataset, dataset=x_test,
                                                   labels=tf.argmax(y_test, axis=1), model_path=model_path)
    get_logits(id_dataset_name=id_dataset, checkpoint_path=model_path, pictures_classified=test_picture_classified)

    OOD_dataset = cnn_config[id_dataset]['ood_settings']
    for ood in OOD_dataset:
        print('\n************Evaluating*************')
        print(f'In-Distribution Data: {id_dataset}, Out-of-Distribution Data: {ood}.')
        ood_data, number_of_ood = load_ood_data(ood_dataset=ood, id_model_path=model_path,
                                                num_of_categories=num_of_labels[id_dataset])

        get_logits(id_dataset_name=id_dataset, checkpoint_path=model_path,
                   pictures_classified=ood_data, ood_dataset_name=ood)

        print('*************************************')
