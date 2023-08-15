
num_of_labels = {
    'mnist': 10,
    'fmnist': 10,
    'cifar10': 10,
    'svhn': 10,
}


cnn_config = {
    'mnist': {
        'layers_of_getting_value': [-10, -8, -6, -4],
        'neurons_of_each_layer': [320, 160, 80, 40],
        'ood_settings': ['FMNIST', 'Omniglot', 'UniformNoise', 'GuassianNoise']
    },
    'fmnist': {
        'layers_of_getting_value': [-10, -8, -6, -4],
        'neurons_of_each_layer': [320, 160, 80, 40]
    },
    'cifar10': {
        'layers_of_getting_value': [-5],
        'neurons_of_each_layer': [512]
    },
    'svhn': {
        'layers_of_getting_value': [-4],
        'neurons_of_each_layer': [512]
    },
}


selective_rate = 0.2
