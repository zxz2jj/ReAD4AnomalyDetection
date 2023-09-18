from datasets import load_dataset, Dataset, DatasetDict
import random
import os
import pickle

task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "rte": ("sentence1", "sentence2"),
    'wmt16': ("en", None),
    'multi30k': ("text", None),
}


def load_data(task):
    if task in ['imdb']:
        dataset = load_dataset('imdb')
        return dataset
    elif task in ['sst2']:
        dataset = load_dataset('gpt3mix/sst2')
        return dataset
    elif task in ['trec']:
        data = load_dataset('trec')
        rename_train = Dataset.from_dict({'text': data['train']['text'], 'label': data['train']['label-coarse']})
        rename_test = Dataset.from_dict({'text': data['test']['text'], 'label': data['test']['label-coarse']})
        dataset = DatasetDict({'train': rename_train, 'test': rename_test})
        return dataset
    elif task in ['newsgroup']:
        dataset = load_20ng()
        return dataset
    elif task in ['noise']:
        dataset = load_noise()
        return dataset
    elif task in ['mnli', 'rte']:
        data = load_glue(task)
    elif task in ['wmt16']:
        data = load_wmt16()
    elif task in ['multi30k']:
        data = load_multi30k()
    else:
        print(f'No dataset named {task}')
        exit()

    sentence1_key, sentence2_key = task_to_keys[task]

    def preprocess_function(examples):
        result = {
            'text': examples[sentence1_key] if sentence2_key is None else examples[sentence1_key] + " " + examples[
                sentence2_key],
            'label': examples["label"] if 'label' in examples else 0}
        return result

    data = list(map(preprocess_function, data['test'])) if 'test' in data else None
    dataset = {"text": [], "label": []}
    for item in data:
        dataset['text'].append(item['text'])
        dataset['label'].append(item['label'])
    dataset = Dataset.from_dict(dataset)
    dataset = DatasetDict({"train": None, "test": dataset})

    return dataset


def load_20ng():
    if os.path.exists('./data/newsgroup/newsgroup/dataset_dict.json'):
        dataset = DatasetDict().load_from_disk('./data/newsgroup/newsgroup/')
        return dataset

    subsets = ('18828_alt.atheism', '18828_comp.graphics', '18828_comp.os.ms-windows.misc',
               '18828_comp.sys.ibm.pc.hardware', '18828_comp.sys.mac.hardware', '18828_comp.windows.x',
               '18828_misc.forsale', '18828_rec.autos', '18828_rec.motorcycles',
               '18828_rec.sport.baseball', '18828_rec.sport.hockey', '18828_sci.crypt',
               '18828_sci.electronics', '18828_sci.med', '18828_sci.space',
               '18828_soc.religion.christian', '18828_talk.politics.guns',
               '18828_talk.politics.mideast', '18828_talk.politics.misc', '18828_talk.religion.misc')
    train_data = []
    test_data = []
    for i, subset in enumerate(subsets):
        data = load_dataset('newsgroup', subset)['train']
        examples = [{'text': d['text'], 'label': i} for d in data]
        random.shuffle(examples)
        num_train = int(0.8 * len(examples))
        train_data += examples[:num_train]
        test_data += examples[num_train:]

    train_dataset = {"text": [], "label": []}
    test_dataset = {"text": [], "label": []}
    for item in train_data:
        train_dataset['text'].append(item['text'])
        train_dataset['label'].append(item['label'])
    for item in test_data:
        test_dataset['text'].append(item['text'])
        test_dataset['label'].append(item['label'])

    train_dataset = Dataset.from_dict(train_dataset)
    test_dataset = Dataset.from_dict(test_dataset)
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
    dataset.save_to_disk('./data/newsgroup/newsgroup/')

    return dataset


def load_glue(task):
    data = load_dataset("glue", task)

    if task == 'mnli':
        test_dataset = [d for d in data['test_matched']] + [d for d in data['test_mismatched']]
        ood_data = {'test': test_dataset}
    elif task == 'rte':
        ood_data = {'test': data['test']}
    else:
        ood_data = None

    return ood_data


def load_wmt16():
    data = load_dataset('wmt16', 'de-en')
    test_dataset = [d['translation'] for d in data['test']]
    ood_data = {'test': test_dataset}

    return ood_data


def load_multi30k():
    test_dataset = []
    for file_name in ('./data/multi30k/test_2016_flickr.en',
                      './data/multi30k/test_2017_mscoco.en',
                      './data/multi30k/test_2018_flickr.en'):
        with open(file_name, 'r') as fh:
            for line in fh:
                line = line.strip()
                if len(line) > 0:
                    example = {'text': line, 'label': 0}
                    test_dataset.append(example)
    ood_data = {'test': test_dataset}
    return ood_data


def load_noise():
    if os.path.exists('./data/public/noise/noise.pkl'):
        file = open('./data/public/noise/noise.pkl', 'rb')
        dataset = pickle.load(file)
        dataset = Dataset.from_dict(dataset)
        ood_data = DatasetDict({"train": None, "test": dataset})
        return ood_data

    dataset = {"text": [], "label": []}
    for i in range(10000):
        sentence = ''
        sentence_length = random.randint(1, 512)
        for j in range(sentence_length):
            word_length = random.randint(1, 15)
            for k in range(word_length):
                char = random.randint(65, 122)
                while 91 <= char <= 96:
                    char = random.randint(65, 122)
                sentence += chr(char)
            sentence += ' '
        sentence.rstrip()
        dataset['text'].append(sentence)
        dataset['label'].append(0)

    file = open('./data/noise/noise.pkl', 'wb')
    pickle.dump(dataset, file)
    dataset = Dataset.from_dict(dataset)
    ood_data = DatasetDict({"train": None, "test": dataset})

    return ood_data


def load_clean_adv_data(path, attack):
    dataset_name = path.split('/')[-3]
    file = open(path + f'{dataset_name}_{attack}_result.pkl', 'rb')
    data = pickle.load(file)
    clean_data = {'text': data['clean'], 'label': data['label']}
    adv_data = {'text': data['adversarial'], 'target': data['target']}
    clean_dataset = Dataset.from_dict(clean_data)
    adv_dataset = Dataset.from_dict(adv_data)
    clean_dataset_dict = DatasetDict({"train": None, "test": clean_dataset})
    adv_dataset_dict = DatasetDict({"train": None, "test": adv_dataset})

    return clean_dataset_dict, adv_dataset_dict


if __name__ == "__main__":

    datasets = load_data('newsgroup')

    # datasets = dataset_2_wordlist(datasets)


    # data = load_data('wmt16')
    # print('aaaa')

    # dataset.save_to_disk('./data/multi30k/')
    # dataset = DatasetDict()
    # a = dataset.load_from_disk('./data/multi30k/')
    # b = a['train']
    # c = a['test']
    # print('1111')

    # dataset = load_noise()
    # print('aaa')

    # load_clean_adv_data(path='./data/sst2/Adversarial/', attack='SCPNAttacker')

