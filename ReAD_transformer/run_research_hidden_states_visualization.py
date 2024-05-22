import matplotlib
matplotlib.get_backend()
import matplotlib.pyplot as plt
import os.path
import torch
from datasets import load_dataset, Dataset
from sklearn.manifold import TSNE
import numpy as np
from transformers import RobertaConfig

import global_config
from global_config import num_of_labels
from load_data import load_data
from ReAD import encode_abstraction, concatenate_data_between_layers, statistic_of_neural_value
from mode_roberta import RobertaForSequenceClassification
from train_roberta_models import sentence_tokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_DISABLED"] = "true"


def get_hidden_states(id_dataset, tokenized_data, checkpoint, split):

    config = RobertaConfig.from_pretrained(checkpoint, output_hidden_states=True)
    model = RobertaForSequenceClassification.from_pretrained(checkpoint, config=config)

    correct_prediction = list()
    correct_confidence = list()
    correct_pooled_output = list()
    correct_hidden_states = [[] for _ in range(config.num_hidden_layers+1)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for p, batch, label in zip(range(tokenized_data.shape[0]), tokenized_data, tokenized_data['label']):
        print('\rImage: {} / {}'.format(p + 1, tokenized_data.shape[0]), end='')
        outputs = model(input_ids=torch.unsqueeze(torch.tensor(batch['input_ids']), 0).to(device),
                        attention_mask=torch.unsqueeze(torch.tensor(batch['attention_mask']), 0).to(device))
        prediction = torch.argmax(outputs.logits, 1)
        hidden_states = outputs.hidden_states
        confidence = torch.softmax(torch.squeeze(outputs.logits), dim=-1)[prediction]
        if prediction == label:
            correct_prediction.append(prediction.cpu().numpy())
            correct_confidence.append(confidence.cpu().detach().numpy())
            correct_pooled_output.append(model.get_hidden_fully_connected_states().cpu().detach().numpy())
            for s, state in enumerate(hidden_states):
                correct_hidden_states[s].append(state[:, 0, :].cpu().detach().numpy())

    if not os.path.exists(f'./data/{id_dataset}/hidden_states/'):
        os.mkdir(f'./data/{id_dataset}/hidden_states/')
    np.save(f'./data/{id_dataset}/hidden_states/{split}_predictions.npy', np.concatenate(correct_prediction))
    np.save(f'./data/{id_dataset}/hidden_states/{split}_confidence.npy', np.concatenate(correct_prediction))
    np.save(f'./data/{id_dataset}/hidden_states/{split}_fc_hidden_state.npy', np.concatenate(correct_pooled_output))
    for s, hidden_state in enumerate(correct_hidden_states):
        hidden_state = np.concatenate(hidden_state)
        np.save(f'./data/{id_dataset}/hidden_states/{split}_hidden_states_encoder_{s}.npy', hidden_state)


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

    # dataset_name = 'sst2'
    # dataset_name = 'imdb'
    # dataset_name = 'trec'
    dataset_name = 'newsgroup'

    if not os.path.exists(f'./data/{dataset_name}/visualization/'):
        os.mkdir(f'./data/{dataset_name}/visualization/')

    finetuned_checkpoint = f"./models/roberta-base-finetuned-{dataset_name}/best"
    detector_path = f'./data/{dataset_name}/detector/'
    model_config = RobertaConfig.from_pretrained(finetuned_checkpoint)

    hidden_states_is_existed = True
    for i in range(model_config.num_hidden_layers+1):
        print(f'Encoder:{i}|', end='')
        train_hidden_states_file = f'./data/{dataset_name}/hidden_states/train_hidden_states_encoder_{i}.npy'
        train_predictions_file = f'./data/{dataset_name}/hidden_states/train_predictions.npy'
        test_hidden_states_file = f'./data/{dataset_name}/hidden_states/test_hidden_states_encoder_{i}.npy'
        test_predictions_file = f'./data/{dataset_name}/hidden_states/test_predictions.npy'
        if os.path.exists(train_hidden_states_file) and os.path.exists(train_predictions_file) and \
                os.path.exists(test_hidden_states_file) and os.path.exists(test_predictions_file):
            print('Hidden States is Existed!')
        else:
            print('Hidden States is not Existed!')
            hidden_states_is_existed = False
            break
    if not hidden_states_is_existed:

        dataset = load_data(dataset_name)
        train_data, test_data = sentence_tokenizer(dataset, finetuned_checkpoint)

        get_hidden_states(id_dataset=dataset_name, tokenized_data=train_data,
                          checkpoint=finetuned_checkpoint, split='train')
        get_hidden_states(id_dataset=dataset_name, tokenized_data=test_data,
                          checkpoint=finetuned_checkpoint, split='test')

    for s in range(model_config.num_hidden_layers+1):
        print(f'-----------------------------------Encoder:{s}----------------------------------------')
        print('Hidden States of Training data:')
        train_hidden_states = np.load(f'./data/{dataset_name}/hidden_states/train_hidden_states_encoder_{s}.npy')
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
        test_hidden_states = np.load(f'./data/{dataset_name}/hidden_states/test_hidden_states_encoder_{s}.npy')
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
                            save_path=f'./data/{dataset_name}/visualization/Encoder_{s}',
                            # mode='show',
                            mode='save')



























