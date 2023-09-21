num_of_labels = {
    "imdb": 2,
    "sst2": 2,
    "trec": 6,
    "newsgroup": 20
}

roberta_config = {
    'sst2': {
        'layers_of_getting_value': [-2],
        'neurons_of_each_layer': [768],
        'ood_settings': ['trec', 'newsgroup', 'mnli', 'rte', 'wmt16', 'multi30k', 'noise'],
        'adversarial_settings': ['SCPNAttacker', 'GANAttacker', 'TextFoolerAttacker',
                                 'PWWSAttacker', 'TextBuggerAttacker', 'VIPERAttacker']
    },
    'imdb': {
        'layers_of_getting_value': [-2],
        'neurons_of_each_layer': [768],
        'ood_settings': ['trec', 'newsgroup', 'mnli', 'rte', 'wmt16', 'multi30k', 'noise'],
        'adversarial_settings': ['SCPNAttacker', 'GANAttacker', 'TextFoolerAttacker',
                                 'PWWSAttacker', 'TextBuggerAttacker', 'VIPERAttacker']
    },
    'trec': {
        'layers_of_getting_value': [-2],
        'neurons_of_each_layer': [768],
        'ood_settings': ['sst2', 'imdb', 'newsgroup', 'mnli', 'rte', 'wmt16', 'multi30k', 'noise'],
        'adversarial_settings': ['SCPNAttacker', 'GANAttacker', 'TextFoolerAttacker',
                                 'PWWSAttacker', 'TextBuggerAttacker', 'VIPERAttacker']
    },
    'newsgroup': {
        'layers_of_getting_value': [-2],
        'neurons_of_each_layer': [768],
        'ood_settings': ['sst2', 'imdb', 'trec', 'mnli', 'rte', 'wmt16', 'multi30k', 'noise'],
        'adversarial_settings': ['SCPNAttacker', 'GANAttacker', 'TextFoolerAttacker',
                                 'PWWSAttacker', 'TextBuggerAttacker', 'VIPERAttacker']
    },
}

selective_rate = 0.6

