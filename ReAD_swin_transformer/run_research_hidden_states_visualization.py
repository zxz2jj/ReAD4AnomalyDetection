import matplotlib
matplotlib.get_backend()
import matplotlib.pyplot as plt
import os.path
import torch
from datasets import load_dataset, Dataset
from sklearn.manifold import TSNE
import numpy as np
from transformers import SwinConfig

import global_config
from global_config import num_of_labels
from load_data import load_gtsrb
from ReAD import encode_abstraction, concatenate_data_between_layers, statistic_of_neural_value
from model_swin_transformer import SwinForImageClassification
from train_swin_models import image_tokenizer
from torch.utils.data import Subset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_DISABLED"] = "true"

# swin-tiny config
depths = [1, 2 + 1, 2 + 1, 6 + 1, 2]
hidden_size = [[96],
               [96, 96, 192],
               [192, 192, 384],
               [384, 384, 384, 384, 384, 384, 768],
               [768, 768]]


def get_hidden_states(id_dataset, data, checkpoint, split):

    config = SwinConfig.from_pretrained(checkpoint, num_labels=global_config.num_of_labels[id_dataset],
                                        output_hidden_states=True, ignore_mismatched_sizes=True)
    model = SwinForImageClassification.from_pretrained(checkpoint, config=config)

    data = image_tokenizer(data=data, model_checkpoint=checkpoint, mode='test')
    data = Subset(data, range(0, data.shape[0])).__getitem__([_ for _ in range(data.shape[0])])
    # data = Subset(data, range(0, 100)).__getitem__([_ for _ in range(100)])

    correct_prediction = list()
    correct_pooled_output = list()
    correct_hidden_states = [[[]],
                             [[] for _ in range(depths[1])],
                             [[] for _ in range(depths[2])],
                             [[] for _ in range(depths[3])],
                             [[] for _ in range(depths[4])]]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    pooler = torch.nn.AdaptiveAvgPool1d(1)
    for p, batch, label in zip(range(len(data['pixel_values'])), data['pixel_values'], data['label']):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print('\rImage: {} / {}'.format(p + 1, len(data['pixel_values'])), end='')
        outputs = model(torch.unsqueeze(batch, 0).to(device))
        prediction = torch.argmax(outputs.logits, 1)
        hidden_states = outputs.hidden_states
        if prediction == label:
            correct_prediction.append(prediction.cpu().numpy())
            correct_pooled_output.append(model.get_pooled_output().cpu().detach().numpy())
            for h, hidden_state in enumerate(hidden_states):
                if not isinstance(hidden_state, tuple):
                    hidden_state = torch.flatten(pooler(hidden_state.transpose(1, 2)), 1)
                    correct_hidden_states[h][0].append(hidden_state.cpu().detach().numpy())
                else:
                    for s, state in enumerate(hidden_state):
                        state = torch.flatten(pooler(state.transpose(1, 2)), 1)
                        correct_hidden_states[h][s].append(state.cpu().detach().numpy())

    if not os.path.exists(f'./data/{id_dataset}/hidden_states/'):
        os.mkdir(f'./data/{id_dataset}/hidden_states/')
    np.save(f'./data/{id_dataset}/hidden_states/{split}_predictions.npy', np.concatenate(correct_prediction))
    np.save(f'./data/{id_dataset}/hidden_states/{split}_pooled_output.npy', np.concatenate(correct_pooled_output))
    for h, hidden_state in enumerate(correct_hidden_states):
        for s, state in enumerate(hidden_state):
            state = np.concatenate(state)
            np.save(f'./data/{id_dataset}/hidden_states/{split}_hidden_states_stage_{h}_encoder_{s}_AvgPool.npy', state)


def t_sne_visualization(hidden_states_data, abstraction_data, category_number, save_path, mode='show'):
    label_to_color = {0: 'red', 1: 'green', 2: 'blue', 3: 'orange', 4: 'pink',
                      5: 'grey', 6: 'black', 7: 'gold', 8: 'darkred', 9: 'chocolate',
                      10: 'saddlebrown', 11: 'olive', 12: 'yellow', 13: 'limegreen', 14: 'turquoise',
                      15: 'cyan', 16: 'deepskyblue', 17: 'blueviolet', 18: 'violet', 19: 'purple',
                      20: 'rosybrown', 21: 'lightcoral', 22: 'brown', 23: 'saddlebrown', 24: 'goldenrod',
                      25: 'yellowgreen', 26: 'darkgreen', 27: 'palegreen', 28: 'darkslategrey', 29: 'cadetblue',
                      30: 'dodgerblue', 31: 'midnightblue', 32: 'slateblue', 33: 'darkslateblue', 34: 'mediumpurple',
                      35: 'indigo', 36: 'darkviolet', 37: 'plum', 38: 'm', 39: 'orchid',
                      40: 'deeppink', 41: 'hotpink', 42: 'crimson'}

    hidden_states = list()
    abstraction = list()
    colour_label = list()
    for category in range(category_number):
        hidden_states.append(hidden_states_data[category]['correct_pictures'])
        abstraction.append(abstraction_data[category]['correct_pictures'])
        colour_label.extend([category for _ in range(abstraction_data[category]['correct_pictures'].shape[0])])
    colour_label_str = []
    for c in colour_label:
        colour_label_str.append(label_to_color[c])

    hidden_states = np.concatenate(hidden_states, axis=0)
    hidden_states_embedded = TSNE(n_components=2).fit_transform(hidden_states)
    max1, min1 = np.max(hidden_states_embedded, 0), np.min(hidden_states_embedded, 0)
    hidden_states_embedded = hidden_states_embedded / (max1 - min1)

    abstraction = np.concatenate(abstraction, axis=0)
    abstraction_embedded = TSNE(n_components=2).fit_transform(abstraction)
    max2, min2 = np.max(abstraction_embedded, 0), np.min(abstraction_embedded, 0)
    abstraction_embedded = abstraction_embedded / (max2 - min2)

    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(121)
    ax1.scatter(hidden_states_embedded[:, 0], hidden_states_embedded[:, 1], c=colour_label_str, s=1)
    ax1.set_title('Hidden States')
    ax2 = fig.add_subplot(122)
    ax2.scatter(abstraction_embedded[:, 0], abstraction_embedded[:, 1], c=colour_label_str, s=1)
    ax2.set_title('ReAD')
    if mode == 'show':
        plt.show()
    elif mode == 'save':
        plt.savefig(save_path + '.png')
        plt.savefig(save_path + '.pdf')
    plt.close()


if __name__ == '__main__':

    # dataset_name = 'mnist'
    # dataset_name = 'fashion_mnist'
    # dataset_name = 'cifar10'
    dataset_name = 'gtsrb'

    hidden_states_is_existed = True
    for k, depth in enumerate(depths):
        for d in range(depth):
            print(f'Stage:{k}, Encoder:{d}|', end='')
            train_hidden_states_file = f'./data/{dataset_name}/hidden_states/train_hidden_states_stage_{k}_encoder_{d}_AvgPool.npy'
            train_predictions_file = f'./data/{dataset_name}/hidden_states/train_predictions.npy'
            test_hidden_states_file = f'./data/{dataset_name}/hidden_states/test_hidden_states_stage_{k}_encoder_{d}_AvgPool.npy'
            test_predictions_file = f'./data/{dataset_name}/hidden_states/test_predictions.npy'
            if os.path.exists(train_hidden_states_file) and os.path.exists(train_predictions_file) and \
                    os.path.exists(test_hidden_states_file) and os.path.exists(test_predictions_file):
                print('Hidden States is Existed!')
            else:
                print('Hidden States is not Existed!')
                hidden_states_is_existed = False
                break
    if not hidden_states_is_existed:
        print('Get Hidden States ...')
        if dataset_name == 'mnist':
            row_names = ('image', 'label')
            finetuned_checkpoint = "./models/swin-finetuned-mnist/best"
            dataset = load_dataset("./data/mnist/mnist/")
            detector_path = './data/mnist/detector/'

        elif dataset_name == 'fashion_mnist':
            row_names = ('image', 'label')
            finetuned_checkpoint = "./models/swin-finetuned-fashion_mnist/best"
            dataset = load_dataset("./data/fashion_mnist/fashion_mnist/")
            detector_path = './data/fashion_mnist/detector/'

        elif dataset_name == 'cifar10':
            row_names = ('img', 'label')
            finetuned_checkpoint = "./models/swin-finetuned-cifar10/best"
            dataset = load_dataset("./data/cifar10/cifar10/")
            detector_path = './data/cifar10/detector/'

        elif dataset_name == 'gtsrb':
            row_names = ('image', 'label')
            finetuned_checkpoint = "./models/swin-finetuned-gtsrb/best"
            dataset = load_gtsrb()
            detector_path = './data/gtsrb/detector/'

        else:
            print('dataset is not exist.')
            exit()

        rename_train_data = Dataset.from_dict({'image': dataset['train'][row_names[0]],
                                               'label': dataset['train'][row_names[1]]})
        rename_test_data = Dataset.from_dict({'image': dataset['test'][row_names[0]],
                                              'label': dataset['test'][row_names[1]]})

        get_hidden_states(id_dataset=dataset_name, data=rename_train_data, checkpoint=finetuned_checkpoint, split='train')
        get_hidden_states(id_dataset=dataset_name, data=rename_test_data, checkpoint=finetuned_checkpoint, split='test')

    for k, depth in enumerate(depths):
        for d in range(depth):
            print(f'-----------------------------------Stage:{k}, Encoder:{d}----------------------------------------')
            global_config.swin_config[dataset_name]['neurons_of_each_layer'] = [hidden_size[k][d]]

            print('Hidden States of Training data:')
            train_hidden_states = np.load(f'./data/{dataset_name}/hidden_states/train_hidden_states_stage_{k}_encoder_{d}_AvgPool.npy')
            train_predictions = np.load(f'./data/{dataset_name}/hidden_states/train_predictions.npy')
            train_hidden_states_dict = {}
            layer_hidden_state = {}
            for i in range(num_of_labels[dataset_name]):
                print('Class {}: {} correct predictions.'.format(i, sum(train_predictions == i)))
                hidden_states_category = {'correct_pictures': train_hidden_states[(train_predictions == i)],
                                          'correct_predictions': train_predictions[(train_predictions == i)],
                                          'wrong_pictures': np.array([]),
                                          'wrong_predictions': np.array([])}
                layer_hidden_state[i] = hidden_states_category
            train_hidden_states_dict[-2] = layer_hidden_state
            train_nv_statistic = statistic_of_neural_value(id_dataset=dataset_name,
                                                           neural_value=train_hidden_states_dict)

            print('Hidden States of Testing data:')
            test_hidden_states = np.load(f'./data/{dataset_name}/hidden_states/test_hidden_states_stage_{k}_encoder_{d}_AvgPool.npy')
            test_predictions = np.load(f'./data/{dataset_name}/hidden_states/test_predictions.npy')
            test_hidden_states_dict = {}
            layer_hidden_state = {}
            for i in range(num_of_labels[dataset_name]):
                print('Class {}: {} correct predictions.'.format(i, sum(test_predictions == i)))
                hidden_states_category = {'correct_pictures': test_hidden_states[(test_predictions == i)],
                                          'correct_predictions': test_predictions[(test_predictions == i)],
                                          'wrong_pictures': np.array([]),
                                          'wrong_predictions': np.array([])}
                layer_hidden_state[i] = hidden_states_category
            test_hidden_states_dict[-2] = layer_hidden_state

            test_hidden_states_concatenated = concatenate_data_between_layers(id_dataset=dataset_name,
                                                                              data_dict=test_hidden_states_dict)
            test_ReAD_abstractions = encode_abstraction(id_dataset=dataset_name, neural_value=test_hidden_states_dict,
                                                        train_dataset_statistic=train_nv_statistic)
            test_ReAD_concatenated = concatenate_data_between_layers(id_dataset=dataset_name,
                                                                     data_dict=test_ReAD_abstractions)

            t_sne_visualization(hidden_states_data=test_hidden_states_concatenated,
                                abstraction_data=test_ReAD_concatenated,
                                category_number=num_of_labels[dataset_name],
                                save_path=f'./data/{dataset_name}/visualization/Stage_{k}_Encoder_{d}',
                                # mode='show',
                                mode='save')



























