from datasets import load_dataset

from load_data import load_gtsrb, load_ood_data
from ReAD import get_neural_value, statistic_of_neural_value, \
    encode_abstraction, concatenate_data_between_layers, k_means, statistic_distance, auroc, sk_auc
from global_config import num_of_labels, swin_config


# os.environ["WANDB_DISABLED"] = "true"


if __name__ == "__main__":

    # id_dataset = 'mnist'
    # id_dataset = 'fashion_mnist'
    # id_dataset = 'svhn'
    id_dataset = 'cifar100'
    # id_dataset = 'cifar10'
    # id_dataset = 'gtsrb'

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

    # ********************** Train Detector **************************** #
    print('\n********************** Train Detector ****************************')
    print('\nGet neural value of train dataset:')
    train_picture_neural_value = get_neural_value(id_dataset=id_dataset, dataset=train_data,
                                                  checkpoint=finetuned_checkpoint, is_ood=False)
    train_nad_concatenated = \
        concatenate_data_between_layers(id_dataset=id_dataset, data_dict=train_picture_neural_value)

    k_means_centers = k_means(data=train_nad_concatenated,
                              category_number=num_of_labels[id_dataset])

    # ********************** Evaluate OOD Detection **************************** #
    print('\n********************** Evaluate OOD Detection ****************************')

    print('\nGet neural value of test dataset:')
    test_picture_neural_value = get_neural_value(id_dataset=id_dataset, dataset=test_data,
                                                 checkpoint=finetuned_checkpoint, is_ood=False)
    test_nad_concatenated = concatenate_data_between_layers(id_dataset=id_dataset, data_dict=test_picture_neural_value)
    print('\nCalculate distance between abstractions and cluster centers ...')
    test_distance = statistic_distance(id_dataset=id_dataset, abstractions=test_nad_concatenated,
                                       cluster_centers=k_means_centers)

    OOD_dataset = swin_config[id_dataset]['ood_settings']
    # OOD_dataset = ['GuassianNoise']
    for ood in OOD_dataset:
        print('\n************Evaluating*************')
        print(f'In-Distribution Data: {id_dataset}, Out-of-Distribution Data: {ood}.')
        ood_data = load_ood_data(ood_dataset=ood)
        print('\nGet neural value of ood dataset...')
        ood_picture_neural_value = get_neural_value(id_dataset=id_dataset, dataset=ood_data,
                                                    checkpoint=finetuned_checkpoint,
                                                    is_ood=True)
        ood_nad_concatenated = concatenate_data_between_layers(id_dataset=id_dataset, data_dict=ood_picture_neural_value)
        print('\nCalculate distance between abstractions and cluster centers ...')
        ood_distance = statistic_distance(id_dataset=id_dataset, abstractions=ood_nad_concatenated,
                                          cluster_centers=k_means_centers)

        auc = sk_auc(id_dataset, test_distance, ood_distance)
        print('*************************************\n')


