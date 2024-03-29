num_of_labels = {
    "mnist": 10,
    "fashion_mnist": 10,
    "cifar10": 10,
    "svhn": 10,
    "cifar100": 100,
    "gtsrb": 43,
    'imagenet100': 100,
}

swin_config = {
    'mnist': {
        'layers_of_getting_value': [-2],
        'neurons_of_each_layer': [768],
        'ood_settings': ['FMNIST', 'Omniglot', 'UniformNoise', 'GuassianNoise']
    },
    'fashion_mnist': {
        'layers_of_getting_value': [-2],
        'neurons_of_each_layer': [768],
        'ood_settings': ['MNIST', 'Omniglot', 'UniformNoise', 'GuassianNoise'],
    },
    'cifar10': {
        'layers_of_getting_value': [-2],
        'neurons_of_each_layer': [768],
        'ood_settings': ['TinyImageNet', 'LSUN', 'iSUN', 'UniformNoise', 'GuassianNoise']
    },
    'svhn': {
        'layers_of_getting_value': [-2],
        'neurons_of_each_layer': [768],
        'ood_settings': ['TinyImageNet', 'LSUN', 'iSUN', 'UniformNoise', 'GuassianNoise']
    },
    'gtsrb': {
        'layers_of_getting_value': [-2],
        'neurons_of_each_layer': [768],
        'ood_settings': ['TinyImageNet', 'LSUN', 'iSUN', 'UniformNoise', 'GuassianNoise']
    },
    'cifar100': {
        'layers_of_getting_value': [-2],
        'neurons_of_each_layer': [768],
        'ood_settings': ['TinyImageNet', 'LSUN', 'iSUN', 'UniformNoise', 'GuassianNoise']
    },
}

selective_rate = 0.6

