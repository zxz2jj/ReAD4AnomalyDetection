import matplotlib
matplotlib.get_backend()
import matplotlib.pyplot as plt
import os.path
import torch
import pickle
from datasets import load_dataset, Dataset
from sklearn.manifold import TSNE
import numpy as np
from transformers import RobertaConfig

import global_config
from global_config import num_of_labels, roberta_config
from load_data import load_data,load_clean_adv_data
from ReAD import (encode_abstraction, concatenate_data_between_layers,
                  statistic_of_neural_value, k_means, statistic_distance, sk_auc)
from mode_roberta import RobertaForSequenceClassification
from train_roberta_models import sentence_tokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_DISABLED"] = "true"


def get_hidden_states(id_dataset, tokenized_data, checkpoint, split):

    config = RobertaConfig.from_pretrained(checkpoint, output_hidden_states=True)
    model = RobertaForSequenceClassification.from_pretrained(checkpoint, config=config)

    predictions_list = list()
    confidence_list = list()
    pooled_output_list = list()
    hidden_states_list = [[] for _ in range(config.num_hidden_layers+1)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if split == 'train' or split == 'test':
        for p, batch, label in zip(range(tokenized_data.shape[0]), tokenized_data, tokenized_data['label']):
            print('\rImage: {} / {}'.format(p + 1, tokenized_data.shape[0]), end='')
            outputs = model(input_ids=torch.unsqueeze(torch.tensor(batch['input_ids']), 0).to(device),
                            attention_mask=torch.unsqueeze(torch.tensor(batch['attention_mask']), 0).to(device))
            prediction = torch.argmax(outputs.logits, 1)
            hidden_states = outputs.hidden_states
            confidence = torch.softmax(torch.squeeze(outputs.logits), dim=-1)[prediction]
            if prediction == label:
                predictions_list.append(prediction.cpu().numpy())
                confidence_list.append(confidence.cpu().detach().numpy())
                pooled_output_list.append(model.get_hidden_fully_connected_states().cpu().detach().numpy())
                for s, state in enumerate(hidden_states):
                    hidden_states_list[s].append(state[:, 0, :].cpu().detach().numpy())
    else:
        for p, batch in zip(range(tokenized_data.shape[0]), tokenized_data):
            print('\rImage: {} / {}'.format(p + 1, tokenized_data.shape[0]), end='')
            outputs = model(input_ids=torch.unsqueeze(torch.tensor(batch['input_ids']), 0).to(device),
                            attention_mask=torch.unsqueeze(torch.tensor(batch['attention_mask']), 0).to(device))
            prediction = torch.argmax(outputs.logits, 1)
            hidden_states = outputs.hidden_states
            confidence = torch.softmax(torch.squeeze(outputs.logits), dim=-1)[prediction]
            predictions_list.append(prediction.cpu().numpy())
            confidence_list.append(confidence.cpu().detach().numpy())
            pooled_output_list.append(model.get_hidden_fully_connected_states().cpu().detach().numpy())
            for s, state in enumerate(hidden_states):
                hidden_states_list[s].append(state[:, 0, :].cpu().detach().numpy())

    if not os.path.exists(f'./data/{id_dataset}/hidden_states/'):
        os.mkdir(f'./data/{id_dataset}/hidden_states/')
    np.save(f'./data/{id_dataset}/hidden_states/{split}_predictions.npy', np.concatenate(predictions_list))
    np.save(f'./data/{id_dataset}/hidden_states/{split}_confidence.npy', np.concatenate(confidence_list))
    np.save(f'./data/{id_dataset}/hidden_states/{split}_fc_hidden_state.npy', np.concatenate(pooled_output_list))
    for s, hidden_state in enumerate(hidden_states_list):
        hidden_state = np.concatenate(hidden_state)
        np.save(f'./data/{id_dataset}/hidden_states/{split}_hidden_states_encoder_{s}.npy', hidden_state)


if __name__ == '__main__':

    if os.path.exists(f'./data/sst2/single_layer_selection_ood.pkl') and \
            os.path.exists(f'./data/imdb/single_layer_selection_ood.pkl') and \
            os.path.exists(f'./data/trec/single_layer_selection_ood.pkl') and \
            os.path.exists(f'./data/newsgroup/single_layer_selection_ood.pkl') and \
            os.path.exists(f'./data/sst2/single_layer_selection_adversarial.pkl') and \
            os.path.exists(f'./data/trec/single_layer_selection_adversarial.pkl') and \
            os.path.exists(f'./data/newsgroup/single_layer_selection_adversarial.pkl') and \
            os.path.exists(f'./data/sst2/multi_layers_selection_ood.pkl') and \
            os.path.exists(f'./data/imdb/multi_layers_selection_ood.pkl') and \
            os.path.exists(f'./data/trec/multi_layers_selection_ood.pkl') and \
            os.path.exists(f'./data/newsgroup/multi_layers_selection_ood.pkl') and \
            os.path.exists(f'./data/sst2/multi_layers_selection_adversarial.pkl') and \
            os.path.exists(f'./data/trec/multi_layers_selection_adversarial.pkl') and \
            os.path.exists(f'./data/newsgroup/multi_layers_selection_adversarial.pkl'):

        # # Single Layer
        print('Single Layer Selection Analyzing...')
        x = [a for a in range(1, 12+1)]
        single_ood_file1 = open(f'./data/sst2/single_layer_selection_ood.pkl', 'rb')
        single_layer_ood_performance_1 = pickle.load(single_ood_file1)
        single_ood_file2 = open(f'./data/imdb/single_layer_selection_ood.pkl', 'rb')
        single_layer_ood_performance_2 = pickle.load(single_ood_file2)
        single_ood_file3 = open(f'./data/trec/single_layer_selection_ood.pkl', 'rb')
        single_layer_ood_performance_3 = pickle.load(single_ood_file3)
        single_ood_file4 = open(f'./data/newsgroup/single_layer_selection_ood.pkl', 'rb')
        single_layer_ood_performance_4 = pickle.load(single_ood_file4)
        single_adv_file1 = open(f'./data/sst2/single_layer_selection_adversarial.pkl', 'rb')
        single_layer_adv_performance_1 = pickle.load(single_adv_file1)
        single_adv_file2 = open(f'./data/trec/single_layer_selection_adversarial.pkl', 'rb')
        single_layer_adv_performance_2 = pickle.load(single_adv_file2)
        single_adv_file3 = open(f'./data/newsgroup/single_layer_selection_adversarial.pkl', 'rb')
        single_layer_adv_performance_3 = pickle.load(single_adv_file3)

        SST2vsTREC = single_layer_ood_performance_1['trec']
        SST2vs20NG = single_layer_ood_performance_1['newsgroup']
        SST2vsMNLI = single_layer_ood_performance_1['mnli']
        SST2vsRTE = single_layer_ood_performance_1['rte']
        SST2vsWMT16 = single_layer_ood_performance_1['wmt16']
        SST2vsMulti30k = single_layer_ood_performance_1['multi30k']
        SST2vsNoise = single_layer_ood_performance_1['noise']
        IMDBvsTREC = single_layer_ood_performance_2['trec']
        IMDBvs20NG = single_layer_ood_performance_2['newsgroup']
        IMDBvsMNLI = single_layer_ood_performance_2['mnli']
        IMDBvsRTE = single_layer_ood_performance_2['rte']
        IMDBvsWMT16 = single_layer_ood_performance_2['wmt16']
        IMDBvsMulti30k = single_layer_ood_performance_2['multi30k']
        IMDBvsNoise = single_layer_ood_performance_2['noise']
        TRECvsSST2 = single_layer_ood_performance_3['sst2']
        TRECvsIMDB = single_layer_ood_performance_3['imdb']
        TRECvs20NG = single_layer_ood_performance_3['newsgroup']
        TRECvsMNLI = single_layer_ood_performance_3['mnli']
        TRECvsRTE = single_layer_ood_performance_3['rte']
        TRECvsWMT16 = single_layer_ood_performance_3['wmt16']
        TRECvsMulti30k = single_layer_ood_performance_3['multi30k']
        TRECvsNoise = single_layer_ood_performance_3['noise']
        NewsGvsSST2 = single_layer_ood_performance_4['sst2']
        NewsGvsIMDB = single_layer_ood_performance_4['imdb']
        NewsGvsTREC = single_layer_ood_performance_4['trec']
        NewsGvsMNLI = single_layer_ood_performance_4['mnli']
        NewsGvsRTE = single_layer_ood_performance_4['rte']
        NewsGvsWMT16 = single_layer_ood_performance_4['wmt16']
        NewsGvsMulti30k = single_layer_ood_performance_4['multi30k']
        NewsGvsNoise = single_layer_ood_performance_4['noise']
        SST2vsSCPN = single_layer_adv_performance_1['SCPNAttacker']
        SST2vsGAN = single_layer_adv_performance_1['GANAttacker']
        SST2vsTextFooler = single_layer_adv_performance_1['TextFoolerAttacker']
        SST2vsPWWS = single_layer_adv_performance_1['PWWSAttacker']
        SST2vsTextBugger = single_layer_adv_performance_1['TextBuggerAttacker']
        SST2vsVIPER = single_layer_adv_performance_1['VIPERAttacker']
        TRECvsSCPN = single_layer_adv_performance_2['SCPNAttacker']
        TRECvsGAN = single_layer_adv_performance_2['GANAttacker']
        TRECvsTextFooler = single_layer_adv_performance_2['TextFoolerAttacker']
        TRECvsPWWS = single_layer_adv_performance_2['PWWSAttacker']
        TRECvsTextBugger = single_layer_adv_performance_2['TextBuggerAttacker']
        TRECvsVIPER = single_layer_adv_performance_2['VIPERAttacker']
        NewsGvsSCPN = single_layer_adv_performance_3['SCPNAttacker']
        NewsGvsGAN = single_layer_adv_performance_3['GANAttacker']
        NewsGvsTextFooler = single_layer_adv_performance_3['TextFoolerAttacker']
        NewsGvsPWWS = single_layer_adv_performance_3['PWWSAttacker']
        NewsGvsTextBugger = single_layer_adv_performance_3['TextBuggerAttacker']
        NewsGvsVIPER = single_layer_adv_performance_3['VIPERAttacker']

        layer = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12']
        plt.ylim(0, 1)
        ax = plt.gca()
        y_major_locator = plt.MultipleLocator(0.1)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.xticks(x, layer)

        # plt.plot(x, SST2vsTREC, label='SST2 vs. TREC-6 (OOD)', marker='.')
        # plt.plot(x, SST2vs20NG, label='SST2 vs. 20NG (OOD)', marker='.')
        plt.plot(x, SST2vsMNLI, label='SST2 vs. MNLI (OOD)', marker='.')
        plt.plot(x, SST2vsRTE, label='SST2 vs. RTE (OOD)', marker='.')
        # plt.plot(x, SST2vsWMT16, label='SST2 vs. WMT16 (OOD)', marker='.')
        # plt.plot(x, SST2vsMulti30k, label='SST2 vs. Multi30k (OOD)', marker='.')
        # plt.plot(x, SST2vsNoise, label='SST2 vs. Noise (OOD)', marker='.')
        # plt.plot(x, IMDBvsTREC, label='IMDB vs. TREC-6 (OOD)', marker='.')
        plt.plot(x, IMDBvs20NG, label='IMDB vs. 20NG (OOD)', marker='.')
        # plt.plot(x, IMDBvsMNLI, label='IMDB vs. MNLI (OOD)', marker='.')
        # plt.plot(x, IMDBvsRTE, label='IMDB vs. RTE (OOD)', marker='.')
        # plt.plot(x, IMDBvsWMT16, label='IMDB vs. WMT16 (OOD)', marker='.')
        # plt.plot(x, IMDBvsMulti30k, label='IMDB vs. Multi30k (OOD)', marker='.')
        plt.plot(x, IMDBvsNoise, label='IMDB vs. Noise (OOD)', marker='.')
        plt.plot(x, TRECvsSST2, label='TREC vs. SST2 (OOD)', marker='.')
        # plt.plot(x, TRECvsIMDB, label='TREC vs. IMDB (OOD)', marker='.')
        # plt.plot(x, TRECvs20NG, label='TREC vs. 20NG (OOD)', marker='.')
        # plt.plot(x, TRECvsMNLI, label='TREC vs. MNLI (OOD)', marker='.')
        # plt.plot(x, TRECvsRTE, label='TREC vs. RTE (OOD)', marker='.')
        # plt.plot(x, TRECvsWMT16, label='TREC vs. WMT16 (OOD)', marker='.')
        plt.plot(x, TRECvsMulti30k, label='TREC vs. Multi30k (OOD)', marker='.')
        # plt.plot(x, TRECvsNoise, label='TREC vs. Noise (OOD)', marker='.')
        # plt.plot(x, NewsGvsSST2, label='20NG vs. SST2 (OOD)', marker='.')
        # plt.plot(x, NewsGvsIMDB, label='20NG vs. IMDB (OOD)', marker='.')
        # plt.plot(x, NewsGvsTREC, label='20NG vs. TREC (OOD)', marker='.')
        # plt.plot(x, NewsGvsMNLI, label='20NG vs. MNLI (OOD)', marker='.')
        # plt.plot(x, NewsGvsRTE, label='20NG vs. RTE (OOD)', marker='.')
        plt.plot(x, NewsGvsWMT16, label='20NG vs. WMT16 (OOD)', marker='.')
        plt.plot(x, NewsGvsMulti30k, label='20NG vs. Multi30k (OOD)', marker='.')
        # plt.plot(x, NewsGvsNoise, label='20NG vs. Noise (OOD)', marker='.')
        # plt.plot(x, SST2vsSCPN, label='SST2 vs. SCPN (Adversarial)', marker='.')
        plt.plot(x, SST2vsGAN, label='SST2 vs. GAN (Adversarial)', marker='.')
        plt.plot(x, SST2vsTextFooler, label='SST2 vs. TextFooler (Adversarial)', marker='.')
        # plt.plot(x, SST2vsPWWS, label='SST2 vs. PWWS (Adversarial)', marker='.')
        # plt.plot(x, SST2vsTextBugger, label='SST2 vs. TextBugger (Adversarial)', marker='.')
        # plt.plot(x, SST2vsVIPER, label='SST2 vs. VIPER (Adversarial)', marker='.')
        plt.plot(x, TRECvsSCPN, label='TREC-6 vs. SCPN (Adversarial)', marker='.')
        # plt.plot(x, TRECvsGAN, label='TREC-6 vs. GAN (Adversarial)', marker='.')
        # plt.plot(x, TRECvsTextFooler, label='TREC-6 vs. TextFooler (Adversarial)', marker='.')
        # plt.plot(x, TRECvsPWWS, label='TREC-6 vs. PWWS (Adversarial)', marker='.')
        plt.plot(x, TRECvsTextBugger, label='TREC-6 vs. TextBugger (Adversarial)', marker='.')
        # plt.plot(x, TRECvsVIPER, label='TREC-6 vs. VIPER (Adversarial)', marker='.')
        # plt.plot(x, NewsGvsSCPN, label='20NG vs. SCPN (Adversarial)', marker='.')
        # plt.plot(x, NewsGvsGAN, label='20NG vs. GAN (Adversarial)', marker='.')
        # plt.plot(x, NewsGvsTextFooler, label='20NG vs. TextFooler (Adversarial)', marker='.')
        plt.plot(x, NewsGvsPWWS, label='20NG vs. PWWS (Adversarial)', marker='.')
        # plt.plot(x, NewsGvsTextBugger, label='20NG vs. TextBugger (Adversarial)', marker='.')
        plt.plot(x, NewsGvsVIPER, label='20NG vs. VIPER (Adversarial)', marker='.')
        plt.legend(loc='lower right', prop={'size': 13})
        plt.tick_params(labelsize=15)
        plt.show()

        # # Layer Aggregation...
        print('Layers Aggregation Selection Analyzing...')
        x = [a for a in range(1, 13+1)]
        multi_ood_file1 = open(f'./data/sst2/multi_layers_selection_ood.pkl', 'rb')
        multi_layers_ood_performance_1 = pickle.load(multi_ood_file1)
        multi_ood_file2 = open(f'./data/imdb/multi_layers_selection_ood.pkl', 'rb')
        multi_layers_ood_performance_2 = pickle.load(multi_ood_file2)
        multi_ood_file3 = open(f'./data/trec/multi_layers_selection_ood.pkl', 'rb')
        multi_layers_ood_performance_3 = pickle.load(multi_ood_file3)
        multi_ood_file4 = open(f'./data/newsgroup/multi_layers_selection_ood.pkl', 'rb')
        multi_layers_ood_performance_4 = pickle.load(multi_ood_file4)
        multi_adv_file1 = open(f'./data/sst2/multi_layers_selection_adversarial.pkl', 'rb')
        multi_layers_adv_performance_1 = pickle.load(multi_adv_file1)
        multi_adv_file2 = open(f'./data/trec/multi_layers_selection_adversarial.pkl', 'rb')
        multi_layers_adv_performance_2 = pickle.load(multi_adv_file2)
        multi_adv_file3 = open(f'./data/newsgroup/multi_layers_selection_adversarial.pkl', 'rb')
        multi_layers_adv_performance_3 = pickle.load(multi_adv_file3)

        SST2vsTREC = multi_layers_ood_performance_1['trec']
        SST2vs20NG = multi_layers_ood_performance_1['newsgroup']
        SST2vsMNLI = multi_layers_ood_performance_1['mnli']
        SST2vsRTE = multi_layers_ood_performance_1['rte']
        SST2vsWMT16 = multi_layers_ood_performance_1['wmt16']
        SST2vsMulti30k = multi_layers_ood_performance_1['multi30k']
        SST2vsNoise = multi_layers_ood_performance_1['noise']
        IMDBvsTREC = multi_layers_ood_performance_2['trec']
        IMDBvs20NG = multi_layers_ood_performance_2['newsgroup']
        IMDBvsMNLI = multi_layers_ood_performance_2['mnli']
        IMDBvsRTE = multi_layers_ood_performance_2['rte']
        IMDBvsWMT16 = multi_layers_ood_performance_2['wmt16']
        IMDBvsMulti30k = multi_layers_ood_performance_2['multi30k']
        IMDBvsNoise = multi_layers_ood_performance_2['noise']
        TRECvsSST2 = multi_layers_ood_performance_3['sst2']
        TRECvsIMDB = multi_layers_ood_performance_3['imdb']
        TRECvs20NG = multi_layers_ood_performance_3['newsgroup']
        TRECvsMNLI = multi_layers_ood_performance_3['mnli']
        TRECvsRTE = multi_layers_ood_performance_3['rte']
        TRECvsWMT16 = multi_layers_ood_performance_3['wmt16']
        TRECvsMulti30k = multi_layers_ood_performance_3['multi30k']
        TRECvsNoise = multi_layers_ood_performance_3['noise']
        NewsGvsSST2 = multi_layers_ood_performance_4['sst2']
        NewsGvsIMDB = multi_layers_ood_performance_4['imdb']
        NewsGvsTREC = multi_layers_ood_performance_4['trec']
        NewsGvsMNLI = multi_layers_ood_performance_4['mnli']
        NewsGvsRTE = multi_layers_ood_performance_4['rte']
        NewsGvsWMT16 = multi_layers_ood_performance_4['wmt16']
        NewsGvsMulti30k = multi_layers_ood_performance_4['multi30k']
        NewsGvsNoise = multi_layers_ood_performance_4['noise']
        SST2vsSCPN = multi_layers_adv_performance_1['SCPNAttacker']
        SST2vsGAN = multi_layers_adv_performance_1['GANAttacker']
        SST2vsTextFooler = multi_layers_adv_performance_1['TextFoolerAttacker']
        SST2vsPWWS = multi_layers_adv_performance_1['PWWSAttacker']
        SST2vsTextBugger = multi_layers_adv_performance_1['TextBuggerAttacker']
        SST2vsVIPER = multi_layers_adv_performance_1['VIPERAttacker']
        TRECvsSCPN = multi_layers_adv_performance_2['SCPNAttacker']
        TRECvsGAN = multi_layers_adv_performance_2['GANAttacker']
        TRECvsTextFooler = multi_layers_adv_performance_2['TextFoolerAttacker']
        TRECvsPWWS = multi_layers_adv_performance_2['PWWSAttacker']
        TRECvsTextBugger = multi_layers_adv_performance_2['TextBuggerAttacker']
        TRECvsVIPER = multi_layers_adv_performance_2['VIPERAttacker']
        NewsGvsSCPN = multi_layers_adv_performance_3['SCPNAttacker']
        NewsGvsGAN = multi_layers_adv_performance_3['GANAttacker']
        NewsGvsTextFooler = multi_layers_adv_performance_3['TextFoolerAttacker']
        NewsGvsPWWS = multi_layers_adv_performance_3['PWWSAttacker']
        NewsGvsTextBugger = multi_layers_adv_performance_3['TextBuggerAttacker']
        NewsGvsVIPER = multi_layers_adv_performance_3['VIPERAttacker']

        layer = ['FC', 'FC-H12', 'FC-H11', 'FC-H10', 'FC-H9', 'FC-H8', 'FC-H7',
                 'FC-H6', 'FC-H5', 'FC-H4', 'FC-H3', 'FC-H2', 'FC-H1']
        plt.ylim(0, 1)
        ax = plt.gca()
        y_major_locator = plt.MultipleLocator(0.1)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.xticks(x, layer, rotation=-45)

        # plt.plot(x, SST2vsTREC, label='SST2 vs. TREC-6 (OOD)', marker='.')
        # plt.plot(x, SST2vs20NG, label='SST2 vs. 20NG (OOD)', marker='.')
        plt.plot(x, SST2vsMNLI, label='SST2 vs. MNLI (OOD)', marker='.')
        plt.plot(x, SST2vsRTE, label='SST2 vs. RTE (OOD)', marker='.')
        # plt.plot(x, SST2vsWMT16, label='SST2 vs. WMT16 (OOD)', marker='.')
        # plt.plot(x, SST2vsMulti30k, label='SST2 vs. Multi30k (OOD)', marker='.')
        # plt.plot(x, SST2vsNoise, label='SST2 vs. Noise (OOD)', marker='.')
        # plt.plot(x, IMDBvsTREC, label='IMDB vs. TREC-6 (OOD)', marker='.')
        plt.plot(x, IMDBvs20NG, label='IMDB vs. 20NG (OOD)', marker='.')
        # plt.plot(x, IMDBvsMNLI, label='IMDB vs. MNLI (OOD)', marker='.')
        # plt.plot(x, IMDBvsRTE, label='IMDB vs. RTE (OOD)', marker='.')
        # plt.plot(x, IMDBvsWMT16, label='IMDB vs. WMT16 (OOD)', marker='.')
        # plt.plot(x, IMDBvsMulti30k, label='IMDB vs. Multi30k (OOD)', marker='.')
        plt.plot(x, IMDBvsNoise, label='IMDB vs. Noise (OOD)', marker='.')
        plt.plot(x, TRECvsSST2, label='TREC vs. SST2 (OOD)', marker='.')
        # plt.plot(x, TRECvsIMDB, label='TREC vs. IMDB (OOD)', marker='.')
        # plt.plot(x, TRECvs20NG, label='TREC vs. 20NG (OOD)', marker='.')
        # plt.plot(x, TRECvsMNLI, label='TREC vs. MNLI (OOD)', marker='.')
        # plt.plot(x, TRECvsRTE, label='TREC vs. RTE (OOD)', marker='.')
        # plt.plot(x, TRECvsWMT16, label='TREC vs. WMT16 (OOD)', marker='.')
        plt.plot(x, TRECvsMulti30k, label='TREC vs. Multi30k (OOD)', marker='.')
        # plt.plot(x, TRECvsNoise, label='TREC vs. Noise (OOD)', marker='.')
        # plt.plot(x, NewsGvsSST2, label='20NG vs. SST2 (OOD)', marker='.')
        # plt.plot(x, NewsGvsIMDB, label='20NG vs. IMDB (OOD)', marker='.')
        # plt.plot(x, NewsGvsTREC, label='20NG vs. TREC (OOD)', marker='.')
        # plt.plot(x, NewsGvsMNLI, label='20NG vs. MNLI (OOD)', marker='.')
        # plt.plot(x, NewsGvsRTE, label='20NG vs. RTE (OOD)', marker='.')
        plt.plot(x, NewsGvsWMT16, label='20NG vs. WMT16 (OOD)', marker='.')
        plt.plot(x, NewsGvsMulti30k, label='20NG vs. Multi30k (OOD)', marker='.')
        # plt.plot(x, NewsGvsNoise, label='20NG vs. Noise (OOD)', marker='.')
        # plt.plot(x, SST2vsSCPN, label='SST2 vs. SCPN (Adversarial)', marker='.')
        plt.plot(x, SST2vsGAN, label='SST2 vs. GAN (Adversarial)', marker='.')
        plt.plot(x, SST2vsTextFooler, label='SST2 vs. TextFooler (Adversarial)', marker='.')
        # plt.plot(x, SST2vsPWWS, label='SST2 vs. PWWS (Adversarial)', marker='.')
        # plt.plot(x, SST2vsTextBugger, label='SST2 vs. TextBugger (Adversarial)', marker='.')
        # plt.plot(x, SST2vsVIPER, label='SST2 vs. VIPER (Adversarial)', marker='.')
        plt.plot(x, TRECvsSCPN, label='TREC-6 vs. SCPN (Adversarial)', marker='.')
        # plt.plot(x, TRECvsGAN, label='TREC-6 vs. GAN (Adversarial)', marker='.')
        # plt.plot(x, TRECvsTextFooler, label='TREC-6 vs. TextFooler (Adversarial)', marker='.')
        # plt.plot(x, TRECvsPWWS, label='TREC-6 vs. PWWS (Adversarial)', marker='.')
        plt.plot(x, TRECvsTextBugger, label='TREC-6 vs. TextBugger (Adversarial)', marker='.')
        # plt.plot(x, TRECvsVIPER, label='TREC-6 vs. VIPER (Adversarial)', marker='.')
        # plt.plot(x, NewsGvsSCPN, label='20NG vs. SCPN (Adversarial)', marker='.')
        # plt.plot(x, NewsGvsGAN, label='20NG vs. GAN (Adversarial)', marker='.')
        # plt.plot(x, NewsGvsTextFooler, label='20NG vs. TextFooler (Adversarial)', marker='.')
        plt.plot(x, NewsGvsPWWS, label='20NG vs. PWWS (Adversarial)', marker='.')
        # plt.plot(x, NewsGvsTextBugger, label='20NG vs. TextBugger (Adversarial)', marker='.')
        plt.plot(x, NewsGvsVIPER, label='20NG vs. VIPER (Adversarial)', marker='.')
        plt.legend(loc='lower right', prop={'size': 13})
        plt.tick_params(labelsize=15)
        plt.show()

    else:
        # dataset_name = 'sst2'
        # dataset_name = 'imdb'
        dataset_name = 'trec'
        # dataset_name = 'newsgroup'

        OOD_dataset = roberta_config[dataset_name]['ood_settings']
        single_layer_OOD_performance = {ood: [] for ood in OOD_dataset}
        multi_layers_OOD_performance = {ood: [] for ood in OOD_dataset}
        Adv_dataset = roberta_config[dataset_name]['adversarial_settings']
        if Adv_dataset is not None:
            single_layer_Adv_performance = {adv: [] for adv in Adv_dataset}
            multi_layers_Adv_performance = {adv: [] for adv in Adv_dataset}
        else:
            single_layer_Adv_performance = None
            multi_layers_Adv_performance = None

        finetuned_checkpoint = f"./models/roberta-base-finetuned-{dataset_name}/best"
        detector_path = f'./data/{dataset_name}/detector/'
        model_config = RobertaConfig.from_pretrained(finetuned_checkpoint)

        train_hidden_states_is_existed = True
        for i in range(model_config.num_hidden_layers+1):
            train_hidden_states_file = f'./data/{dataset_name}/hidden_states/train_hidden_states_encoder_{i}.npy'
            train_predictions_file = f'./data/{dataset_name}/hidden_states/train_predictions.npy'
            if os.path.exists(train_hidden_states_file) and os.path.exists(train_predictions_file):
                pass
            else:
                train_hidden_states_is_existed = False
                break
        if train_hidden_states_is_existed:
            print(f'ID dataset-{dataset_name} Hidden States of Train Dataset is Existed!')
        else:
            print(f'ID dataset-{dataset_name} Hidden States of Train Dataset is not Existed! Getting...')
            dataset = load_data(dataset_name)
            train_data, test_data = sentence_tokenizer(dataset, finetuned_checkpoint)
            get_hidden_states(id_dataset=dataset_name, tokenized_data=train_data,
                              checkpoint=finetuned_checkpoint, split='train')

        test_hidden_states_is_existed = True
        for i in range(model_config.num_hidden_layers+1):
            test_hidden_states_file = f'./data/{dataset_name}/hidden_states/test_hidden_states_encoder_{i}.npy'
            test_predictions_file = f'./data/{dataset_name}/hidden_states/test_predictions.npy'
            if os.path.exists(test_hidden_states_file) and os.path.exists(test_predictions_file):
                pass
            else:
                test_hidden_states_is_existed = False
                break
        if test_hidden_states_is_existed:
            print(f'ID dataset-{dataset_name} Hidden States of Test Dataset is Existed!')
        else:
            print(f'ID dataset-{dataset_name} Hidden States of Test Dataset is not Existed! Getting...')
            dataset = load_data(dataset_name)
            train_data, test_data = sentence_tokenizer(dataset, finetuned_checkpoint)
            get_hidden_states(id_dataset=dataset_name, tokenized_data=test_data,
                              checkpoint=finetuned_checkpoint, split='test')

        OOD_dataset = roberta_config[dataset_name]['ood_settings']
        for ood in OOD_dataset:
            print(f'\nIn-Distribution Data: {dataset_name}, Out-of-Distribution Data: {ood}.')
            ood_hidden_states_is_existed = True
            for i in range(model_config.num_hidden_layers + 1):
                ood_hidden_states_file = f'./data/{dataset_name}/hidden_states/ood_{ood}_hidden_states_encoder_{i}.npy'
                ood_predictions_file = f'./data/{dataset_name}/hidden_states/ood_{ood}_predictions.npy'
                if os.path.exists(ood_hidden_states_file) and os.path.exists(ood_predictions_file):
                    pass
                else:
                    ood_hidden_states_is_existed = False
                    break
            if ood_hidden_states_is_existed:
                print(f'OOD dataset-{ood} Hidden States is Existed!')
            else:
                print(f'OOD dataset-{ood} Hidden States is not Existed! Getting...')
                ood_data = load_data(ood)
                _, ood_data = sentence_tokenizer(ood_data, finetuned_checkpoint)
                print('\nGet neural value of ood dataset...')
                get_hidden_states(id_dataset=dataset_name, tokenized_data=ood_data,
                                  checkpoint=finetuned_checkpoint, split=f'ood_{ood}')

        Adv_dataset = roberta_config[dataset_name]['adversarial_settings']
        if Adv_dataset is not None:
            for attack in Adv_dataset:
                print(f'\nClean Data: {dataset_name}, Adversarial Data: {attack}.')
                adv_hidden_states_is_existed = True
                for i in range(model_config.num_hidden_layers + 1):
                    adv_hidden_states_file = f'./data/{dataset_name}/hidden_states/adv_{attack}_hidden_states_encoder_{i}.npy'
                    adv_predictions_file = f'./data/{dataset_name}/hidden_states/adv_{attack}_predictions.npy'
                    clean_hidden_states_file = f'./data/{dataset_name}/hidden_states/clean_{attack}_hidden_states_encoder_{i}.npy'
                    clean_predictions_file = f'./data/{dataset_name}/hidden_states/clean_{attack}_predictions.npy'
                    if os.path.exists(adv_hidden_states_file) and os.path.exists(adv_predictions_file) \
                            and os.path.exists(clean_hidden_states_file) and os.path.exists(clean_predictions_file):
                        pass
                    else:
                        adv_hidden_states_is_existed = False
                        break
                if adv_hidden_states_is_existed:
                    print(f'Adversarial data-{attack} Hidden States is Existed!')
                else:
                    print(f'Adversarial data-{attack} Hidden States is not Existed! Getting...')
                    clean_data, adv_data = load_clean_adv_data(clean_dataset=dataset_name, attack=attack)
                    _, clean_data = sentence_tokenizer(clean_data, finetuned_checkpoint)
                    _, adv_data = sentence_tokenizer(adv_data, finetuned_checkpoint)
                    print('\nGet neural value of Clean dataset...')
                    get_hidden_states(id_dataset=dataset_name, tokenized_data=clean_data,
                                      checkpoint=finetuned_checkpoint, split=f'clean_{attack}')
                    print('\nGet neural value of Adversarial dataset...')
                    get_hidden_states(id_dataset=dataset_name, tokenized_data=adv_data,
                                      checkpoint=finetuned_checkpoint, split=f'adv_{attack}')

        run_single_layer_selection_research = False
        if run_single_layer_selection_research:
            # ********************************* single layer selection *************************************** #
            num_of_hidden_layers = 12

            for layer in range(1, num_of_hidden_layers + 1):
                print(f'------------------------------layer: {layer}---------------------------------------------------')
                train_nv_statistic = None
                k_means_centers = None
                if os.path.exists(detector_path + f'train_nv_statistic_encoder_{layer}.pkl') and \
                        os.path.exists(detector_path + f'k_means_centers_encoder_{layer}.pkl'):
                    print(f"\n Detector for Encoder-{layer} is existed!")
                    file1 = open(detector_path + f'train_nv_statistic_encoder_{layer}.pkl', 'rb')
                    file2 = open(detector_path + f'k_means_centers_encoder_{layer}.pkl', 'rb')
                    train_nv_statistic = pickle.load(file1)
                    k_means_centers = pickle.load(file2)
                else:
                    print(f'\n********************** Train Detector for Encoder-{layer} ****************************')

                    layer_neural_value = {}
                    train_neural_value = {-2: layer_neural_value}
                    train_hidden_states_of_layer = np.load(
                        f'./data/{dataset_name}/hidden_states/train_hidden_states_encoder_{layer}.npy')
                    train_predictions = np.load(f'./data/{dataset_name}/hidden_states/train_predictions.npy')
                    for i in range(num_of_labels[dataset_name]):
                        print('Class {}: {}'.format(i, sum(train_predictions == i)))
                        neural_value_category = {'correct_pictures': train_hidden_states_of_layer[(train_predictions == i)],
                                                 'correct_predictions': train_predictions[(train_predictions == i)],
                                                 'wrong_pictures': np.array([]),
                                                 'wrong_predictions': np.array([])}
                        layer_neural_value[i] = neural_value_category

                    print('\nStatistic of train data neural value...')
                    train_nv_statistic = statistic_of_neural_value(id_dataset=dataset_name, neural_value=train_neural_value)
                    print('finished!')
                    file1 = open(detector_path + f'train_nv_statistic_encoder_{layer}.pkl', 'wb')
                    pickle.dump(train_nv_statistic, file1)
                    file1.close()
                    print('\nEncoding ReAD abstraction for train dataset:')
                    train_ReAD_abstractions = encode_abstraction(id_dataset=dataset_name, neural_value=train_neural_value,
                                                                 train_dataset_statistic=train_nv_statistic)
                    train_ReAD_concatenated = \
                        concatenate_data_between_layers(id_dataset=dataset_name, data_dict=train_ReAD_abstractions)
                    print('\nK-Means Clustering of ReAD on train data:')
                    k_means_centers = k_means(data=train_ReAD_concatenated,
                                              category_number=num_of_labels[dataset_name])
                    file2 = open(detector_path + f'k_means_centers_encoder_{layer}.pkl', 'wb')
                    pickle.dump(k_means_centers, file2)
                    file2.close()

                print('\n********************** OOD Detection on Single Layer ****************************')
                test_layer_neural_value = {}
                test_neural_value = {-2: test_layer_neural_value}
                test_hidden_states_of_layer = np.load(
                    f'./data/{dataset_name}/hidden_states/test_hidden_states_encoder_{layer}.npy')
                test_predictions = np.load(f'./data/{dataset_name}/hidden_states/test_predictions.npy')
                for i in range(num_of_labels[dataset_name]):
                    print('Class {}: {}'.format(i, sum(test_predictions == i)))
                    neural_value_category = {
                        'correct_pictures': test_hidden_states_of_layer[test_predictions == i],
                        'correct_predictions': test_predictions[test_predictions == i],
                        'wrong_pictures': np.array([]),
                        'wrong_predictions': np.array([])}
                    test_layer_neural_value[i] = neural_value_category

                print('\nEncoding ReAD abstraction for test dataset:')
                test_ReAD_abstractions = encode_abstraction(id_dataset=dataset_name, neural_value=test_neural_value,
                                                            train_dataset_statistic=train_nv_statistic)
                test_ReAD_concatenated = \
                    concatenate_data_between_layers(id_dataset=dataset_name, data_dict=test_ReAD_abstractions)
                print('\nCalculate distance between ReAD and cluster centers ...')
                test_distance = statistic_distance(id_dataset=dataset_name, abstractions=test_ReAD_concatenated,
                                                   cluster_centers=k_means_centers)
                for ood in OOD_dataset:
                    print(f'\nID dataset: {dataset_name}, OOD dataset: {ood}')
                    ood_layer_neural_value = {}
                    ood_neural_value = {-2: ood_layer_neural_value}
                    ood_hidden_states_of_layer = np.load(
                        f'./data/{dataset_name}/hidden_states/ood_{ood}_hidden_states_encoder_{layer}.npy')
                    ood_predictions = np.load(f'./data/{dataset_name}/hidden_states/ood_{ood}_predictions.npy')
                    for i in range(num_of_labels[dataset_name]):
                        print('Class {}: {}'.format(i, sum(ood_predictions == i)))
                        neural_value_category = {'correct_pictures': np.array([]),
                                                 'correct_predictions': np.array([]),
                                                 'wrong_pictures': ood_hidden_states_of_layer[ood_predictions == i],
                                                 'wrong_predictions': ood_predictions[ood_predictions == i]}
                        ood_layer_neural_value[i] = neural_value_category

                    print('\nEncoding ReAD abstraction for ood dataset...')
                    ood_ReAD_abstractions = encode_abstraction(id_dataset=dataset_name, neural_value=ood_neural_value,
                                                               train_dataset_statistic=train_nv_statistic)
                    ood_ReAD_concatenated = \
                        concatenate_data_between_layers(id_dataset=dataset_name, data_dict=ood_ReAD_abstractions)
                    print('\nCalculate distance between ReAD and cluster centers ...')
                    ood_distance = statistic_distance(id_dataset=dataset_name, abstractions=ood_ReAD_concatenated,
                                                      cluster_centers=k_means_centers)
                    performance = sk_auc(dataset_name, test_distance, ood_distance)
                    single_layer_OOD_performance[ood].append(performance[-1]['avg']['AUROC'])

                if Adv_dataset is not None:
                    print('\n********************** Adversarial Detection on Single Layer ****************************')
                    for adv in Adv_dataset:
                        print(f'\nClean dataset: {dataset_name}, Adversarial dataset: {adv}')
                        clean_layer_neural_value = {}
                        clean_neural_value = {-2: clean_layer_neural_value}
                        clean_hidden_states_of_layer = np.load(
                            f'./data/{dataset_name}/hidden_states/clean_{adv}_hidden_states_encoder_{layer}.npy')
                        clean_predictions = np.load(f'./data/{dataset_name}/hidden_states/clean_{adv}_predictions.npy')
                        for i in range(num_of_labels[dataset_name]):
                            print('Class {}: {}'.format(i, sum(clean_predictions == i)))
                            neural_value_category = {
                                'correct_pictures': clean_hidden_states_of_layer[clean_predictions == i],
                                'correct_predictions': clean_predictions[clean_predictions == i],
                                'wrong_pictures': np.array([]),
                                'wrong_predictions': np.array([])}
                            clean_layer_neural_value[i] = neural_value_category

                        adv_layer_neural_value = {}
                        adv_neural_value = {-2: adv_layer_neural_value}
                        adv_hidden_states_of_layer = np.load(
                            f'./data/{dataset_name}/hidden_states/adv_{adv}_hidden_states_encoder_{layer}.npy')
                        adv_predictions = np.load(f'./data/{dataset_name}/hidden_states/adv_{adv}_predictions.npy')
                        for i in range(num_of_labels[dataset_name]):
                            print('Class {}: {}'.format(i, sum(clean_predictions == i)))
                            neural_value_category = {'correct_pictures': np.array([]),
                                                     'correct_predictions': np.array([]),
                                                     'wrong_pictures': adv_hidden_states_of_layer[adv_predictions == i],
                                                     'wrong_predictions': adv_predictions[adv_predictions == i]}
                            adv_layer_neural_value[i] = neural_value_category

                        print('\nEncoding ReAD abstraction for CLEAN data...')
                        clean_ReAD_abstractions = encode_abstraction(id_dataset=dataset_name,
                                                                     neural_value=clean_neural_value,
                                                                     train_dataset_statistic=train_nv_statistic)
                        clean_ReAD_concatenated = \
                            concatenate_data_between_layers(id_dataset=dataset_name, data_dict=clean_ReAD_abstractions)
                        print('\nEncoding ReAD abstraction for ADVERSARIAL data...')
                        adv_ReAD_abstractions = encode_abstraction(id_dataset=dataset_name, neural_value=adv_neural_value,
                                                                   train_dataset_statistic=train_nv_statistic)
                        adv_ReAD_concatenated = \
                            concatenate_data_between_layers(id_dataset=dataset_name, data_dict=adv_ReAD_abstractions)

                        print('\nCalculate distance between ReAD and cluster centers for CLEAN data ...')
                        clean_distance = statistic_distance(id_dataset=dataset_name, abstractions=clean_ReAD_concatenated,
                                                            cluster_centers=k_means_centers)
                        print('\nCalculate distance between ReAD and cluster centers for ADVERSARIAL data ...')
                        adv_distance = statistic_distance(id_dataset=dataset_name, abstractions=adv_ReAD_concatenated,
                                                          cluster_centers=k_means_centers)
                        performance = sk_auc(dataset_name, clean_distance, adv_distance)
                        single_layer_Adv_performance[adv].append(performance[-1]['avg']['AUROC'])

            single_ood_performance_file = open(f'./data/{dataset_name}/single_layer_selection_ood.pkl', 'wb')
            pickle.dump(single_layer_OOD_performance, single_ood_performance_file)
            single_ood_performance_file.close()
            if single_layer_Adv_performance is not None:
                single_adv_performance_file = open(f'./data/{dataset_name}/single_layer_selection_adversarial.pkl', 'wb')
                pickle.dump(single_layer_Adv_performance, single_adv_performance_file)
                single_adv_performance_file.close()

        run_multi_layers_selection_research = True
        if run_multi_layers_selection_research:
            # ********************************* layer aggregation selection *************************************** #
            print('************************************** layer aggregation selection *****************************')
            num_of_hidden_layers = 12

            for hidden_layers in range(0, num_of_hidden_layers+1):
                print(f'-------------------------FC to Layer: {13 - hidden_layers} ---------------------------------')
                train_nv_statistic = None
                k_means_centers = None
                if os.path.exists(detector_path + f'train_nv_statistic_fc_to_encoder_{13 - hidden_layers}.pkl') and \
                        os.path.exists(detector_path + f'k_means_centers_fc_to_encoder_{13 - hidden_layers}.pkl'):
                    print(f"\n Detector for FC_to_Encoder_{13 - hidden_layers} is existed!")
                    file1 = open(detector_path + f'train_nv_statistic_fc_to_encoder_{13 - hidden_layers}.pkl', 'rb')
                    file2 = open(detector_path + f'k_means_centers_fc_to_encoder_{13 - hidden_layers}.pkl', 'rb')
                    train_nv_statistic = pickle.load(file1)
                    k_means_centers = pickle.load(file2)
                else:
                    print(f'\n*************** Train Detector for FC_to_Encoder_{13 - hidden_layers} *****************')

                    print('\nGetting neural value of FC layer for training data...')
                    train_neural_value = {}
                    layer_neural_value = {}
                    train_hidden_states_of_layer = np.load(
                        f'./data/{dataset_name}/hidden_states/train_fc_hidden_state.npy')
                    train_predictions = np.load(f'./data/{dataset_name}/hidden_states/train_predictions.npy')
                    for i in range(num_of_labels[dataset_name]):
                        print('Class {}: {}'.format(i, sum(train_predictions == i)))
                        neural_value_category = {
                            'correct_pictures': train_hidden_states_of_layer[(train_predictions == i)],
                            'correct_predictions': train_predictions[(train_predictions == i)],
                            'wrong_pictures': np.array([]),
                            'wrong_predictions': np.array([])}
                        layer_neural_value[i] = neural_value_category
                    train_neural_value[-1] = layer_neural_value
                    global_config.roberta_config[dataset_name]['layers_of_getting_value'] = [-1]

                    for j in range(hidden_layers):
                        print(f'\nGetting neural value of Hidden Layer {12 - j}  for training data...')
                        layer_neural_value = {}
                        train_hidden_states_of_layer = np.load(
                            f'./data/{dataset_name}/hidden_states/train_hidden_states_encoder_{12 - hidden_layers}.npy')
                        for i in range(num_of_labels[dataset_name]):
                            neural_value_category = {'correct_pictures': train_hidden_states_of_layer[(train_predictions == i)],
                                                     'correct_predictions': train_predictions[(train_predictions == i)],
                                                     'wrong_pictures': np.array([]),
                                                     'wrong_predictions': np.array([])}
                            layer_neural_value[i] = neural_value_category
                        train_neural_value[-(j + 2)] = layer_neural_value
                        global_config.roberta_config[dataset_name]['layers_of_getting_value'].append(-(j + 2))

                    global_config.roberta_config[dataset_name]['neurons_of_each_layer'] = [768 for _ in
                                                                                           range(hidden_layers + 1)]

                    print('\nStatistic of train data neural value...')
                    train_nv_statistic = statistic_of_neural_value(id_dataset=dataset_name, neural_value=train_neural_value)
                    print('finished!')
                    file1 = open(detector_path + f'train_nv_statistic_fc_to_encoder_{13 - hidden_layers}.pkl', 'wb')
                    pickle.dump(train_nv_statistic, file1)
                    file1.close()
                    print('\nEncoding ReAD abstraction for train dataset:')
                    train_ReAD_abstractions = encode_abstraction(id_dataset=dataset_name, neural_value=train_neural_value,
                                                                 train_dataset_statistic=train_nv_statistic)
                    train_ReAD_concatenated = \
                        concatenate_data_between_layers(id_dataset=dataset_name, data_dict=train_ReAD_abstractions)
                    print('\nK-Means Clustering of ReAD on train data:')
                    k_means_centers = k_means(data=train_ReAD_concatenated,
                                              category_number=num_of_labels[dataset_name])
                    file2 = open(detector_path + f'k_means_centers_fc_to_encoder_{13 - hidden_layers}.pkl', 'wb')
                    pickle.dump(k_means_centers, file2)
                    file2.close()

                print('\n********************** OOD Detection on Layers Aggregation****************************')
                print('\nGetting neural value of FC layer for Testing data...')
                test_neural_value = {}
                layer_neural_value = {}
                test_hidden_states_of_layer = np.load(f'./data/{dataset_name}/hidden_states/test_fc_hidden_state.npy')
                test_predictions = np.load(f'./data/{dataset_name}/hidden_states/test_predictions.npy')
                for i in range(num_of_labels[dataset_name]):
                    print('Class {}: {}'.format(i, sum(test_predictions == i)))
                    neural_value_category = {
                        'correct_pictures': test_hidden_states_of_layer[(test_predictions == i)],
                        'correct_predictions': test_predictions[(test_predictions == i)],
                        'wrong_pictures': np.array([]),
                        'wrong_predictions': np.array([])}
                    layer_neural_value[i] = neural_value_category
                test_neural_value[-1] = layer_neural_value
                global_config.roberta_config[dataset_name]['layers_of_getting_value'] = [-1]

                for j in range(hidden_layers):
                    print(f'\nGetting neural value of Hidden Layer {12 - j}  for testing data...')
                    layer_neural_value = {}
                    test_hidden_states_of_layer = np.load(
                        f'./data/{dataset_name}/hidden_states/test_hidden_states_encoder_{12 - hidden_layers}.npy')
                    for i in range(num_of_labels[dataset_name]):
                        neural_value_category = {
                            'correct_pictures': test_hidden_states_of_layer[(test_predictions == i)],
                            'correct_predictions': test_predictions[(test_predictions == i)],
                            'wrong_pictures': np.array([]),
                            'wrong_predictions': np.array([])}
                        layer_neural_value[i] = neural_value_category
                    test_neural_value[-(j + 2)] = layer_neural_value
                    global_config.roberta_config[dataset_name]['layers_of_getting_value'].append(-(j + 2))

                print('\nEncoding ReAD abstraction for test dataset:')
                test_ReAD_abstractions = encode_abstraction(id_dataset=dataset_name, neural_value=test_neural_value,
                                                            train_dataset_statistic=train_nv_statistic)
                test_ReAD_concatenated = \
                    concatenate_data_between_layers(id_dataset=dataset_name, data_dict=test_ReAD_abstractions)
                print('\nCalculate distance between ReAD and cluster centers ...')
                test_distance = statistic_distance(id_dataset=dataset_name, abstractions=test_ReAD_concatenated,
                                                   cluster_centers=k_means_centers)
                for ood in OOD_dataset:
                    print(f'\nID dataset: {dataset_name}, OOD dataset: {ood}')
                    print(f'\nGetting neural value of FC layer for OOD-{ood} data...')
                    ood_neural_value = {}
                    layer_neural_value = {}
                    ood_hidden_states_of_layer = np.load(
                        f'./data/{dataset_name}/hidden_states/ood_{ood}_fc_hidden_state.npy')
                    ood_predictions = np.load(f'./data/{dataset_name}/hidden_states/ood_{ood}_predictions.npy')
                    for i in range(num_of_labels[dataset_name]):
                        print('Class {}: {}'.format(i, sum(ood_predictions == i)))
                        neural_value_category = {
                            'correct_pictures': np.array([]),
                            'correct_predictions': np.array([]),
                            'wrong_pictures': ood_hidden_states_of_layer[(ood_predictions == i)],
                            'wrong_predictions': ood_predictions[(ood_predictions == i)]}
                        layer_neural_value[i] = neural_value_category
                    ood_neural_value[-1] = layer_neural_value

                    for j in range(hidden_layers):
                        print(f'\nGetting neural value of Hidden Layer {12 - j}  for OOD-{ood} data...')
                        layer_neural_value = {}
                        ood_hidden_states_of_layer = np.load(
                            f'./data/{dataset_name}/hidden_states/ood_{ood}_hidden_states_encoder_{12 - hidden_layers}.npy')
                        for i in range(num_of_labels[dataset_name]):
                            neural_value_category = {
                                'correct_pictures': np.array([]),
                                'correct_predictions': np.array([]),
                                'wrong_pictures': ood_hidden_states_of_layer[(ood_predictions == i)],
                                'wrong_predictions': ood_predictions[(ood_predictions == i)]}
                            layer_neural_value[i] = neural_value_category
                        ood_neural_value[-(j + 2)] = layer_neural_value

                    print('\nEncoding ReAD abstraction for ood dataset...')
                    ood_ReAD_abstractions = encode_abstraction(id_dataset=dataset_name, neural_value=ood_neural_value,
                                                               train_dataset_statistic=train_nv_statistic)
                    ood_ReAD_concatenated = \
                        concatenate_data_between_layers(id_dataset=dataset_name, data_dict=ood_ReAD_abstractions)
                    print('\nCalculate distance between ReAD and cluster centers ...')
                    ood_distance = statistic_distance(id_dataset=dataset_name, abstractions=ood_ReAD_concatenated,
                                                      cluster_centers=k_means_centers)
                    performance = sk_auc(dataset_name, test_distance, ood_distance)
                    multi_layers_OOD_performance[ood].append(performance[-1]['avg']['AUROC'])

                if Adv_dataset is not None:
                    print('\n********************** Adversarial Detection on Single Layer ****************************')
                    for attack in Adv_dataset:
                        print(f'\nClean dataset: {dataset_name}, Adversarial dataset: {attack}')
                        print('\nGetting neural value of FC layer for CLEAN data...')
                        clean_neural_value = {}
                        layer_neural_value = {}
                        clean_hidden_states_of_layer = np.load(
                            f'./data/{dataset_name}/hidden_states/clean_{attack}_fc_hidden_state.npy')
                        clean_predictions = np.load(f'./data/{dataset_name}/hidden_states/clean_{attack}_predictions.npy')
                        for i in range(num_of_labels[dataset_name]):
                            print('Class {}: {}'.format(i, sum(clean_predictions == i)))
                            neural_value_category = {
                                'correct_pictures': clean_hidden_states_of_layer[(clean_predictions == i)],
                                'correct_predictions': clean_predictions[(clean_predictions == i)],
                                'wrong_pictures': np.array([]),
                                'wrong_predictions': np.array([])}
                            layer_neural_value[i] = neural_value_category
                        clean_neural_value[-1] = layer_neural_value

                        for j in range(hidden_layers):
                            print(f'\nGetting neural value of Hidden Layer {12 - j}  for CLEAN data...')
                            layer_neural_value = {}
                            clean_hidden_states_of_layer = np.load(
                                f'./data/{dataset_name}/hidden_states/clean_{attack}_hidden_states_encoder_{12 - hidden_layers}.npy')
                            for i in range(num_of_labels[dataset_name]):
                                neural_value_category = {
                                    'correct_pictures': clean_hidden_states_of_layer[(clean_predictions == i)],
                                    'correct_predictions': clean_predictions[(clean_predictions == i)],
                                    'wrong_pictures': np.array([]),
                                    'wrong_predictions': np.array([])}
                                layer_neural_value[i] = neural_value_category
                            clean_neural_value[-(j + 2)] = layer_neural_value

                        print('\nGetting neural value of FC layer for ADVERSARIAL data...')
                        adv_neural_value = {}
                        layer_neural_value = {}
                        adv_hidden_states_of_layer = np.load(
                            f'./data/{dataset_name}/hidden_states/adv_{attack}_fc_hidden_state.npy')
                        adv_predictions = np.load(f'./data/{dataset_name}/hidden_states/adv_{attack}_predictions.npy')
                        for i in range(num_of_labels[dataset_name]):
                            print('Class {}: {}'.format(i, sum(adv_predictions == i)))
                            neural_value_category = {'correct_pictures': np.array([]),
                                                     'correct_predictions': np.array([]),
                                                     'wrong_pictures': adv_hidden_states_of_layer[adv_predictions == i],
                                                     'wrong_predictions': adv_predictions[adv_predictions == i]}
                            layer_neural_value[i] = neural_value_category
                        adv_neural_value[-1] = layer_neural_value

                        for j in range(hidden_layers):
                            print(f'\nGetting neural value of Hidden Layer {12 - j}  for ADVERSARIAL data...')
                            layer_neural_value = {}
                            adv_hidden_states_of_layer = np.load(
                                f'./data/{dataset_name}/hidden_states/adv_{attack}_hidden_states_encoder_{12 - hidden_layers}.npy')
                            for i in range(num_of_labels[dataset_name]):
                                neural_value_category = {'correct_pictures': np.array([]),
                                                         'correct_predictions': np.array([]),
                                                         'wrong_pictures': adv_hidden_states_of_layer[adv_predictions == i],
                                                         'wrong_predictions': adv_predictions[adv_predictions == i]}
                                layer_neural_value[i] = neural_value_category
                            adv_neural_value[-(j + 2)] = layer_neural_value

                        print('\nEncoding ReAD abstraction for CLEAN data...')
                        clean_ReAD_abstractions = encode_abstraction(id_dataset=dataset_name, neural_value=clean_neural_value,
                                                                     train_dataset_statistic=train_nv_statistic)
                        clean_ReAD_concatenated = \
                            concatenate_data_between_layers(id_dataset=dataset_name, data_dict=clean_ReAD_abstractions)
                        print('\nEncoding ReAD abstraction for ADVERSARIAL data...')
                        adv_ReAD_abstractions = encode_abstraction(id_dataset=dataset_name, neural_value=adv_neural_value,
                                                                   train_dataset_statistic=train_nv_statistic)
                        adv_ReAD_concatenated = \
                            concatenate_data_between_layers(id_dataset=dataset_name, data_dict=adv_ReAD_abstractions)

                        print('\nCalculate distance between ReAD and cluster centers for CLEAN data ...')
                        clean_distance = statistic_distance(id_dataset=dataset_name, abstractions=clean_ReAD_concatenated,
                                                            cluster_centers=k_means_centers)
                        print('\nCalculate distance between ReAD and cluster centers for ADVERSARIAL data ...')
                        adv_distance = statistic_distance(id_dataset=dataset_name, abstractions=adv_ReAD_concatenated,
                                                          cluster_centers=k_means_centers)
                        performance = sk_auc(dataset_name, clean_distance, adv_distance)
                        multi_layers_Adv_performance[attack].append(performance[-1]['avg']['AUROC'])

            multi_ood_performance_file = open(f'./data/{dataset_name}/multi_layers_selection_ood.pkl', 'wb')
            pickle.dump(multi_layers_OOD_performance, multi_ood_performance_file)
            multi_ood_performance_file.close()
            if multi_layers_Adv_performance is not None:
                multi_adv_performance_file = open(f'./data/{dataset_name}/multi_layers_selection_adversarial.pkl', 'wb')
                pickle.dump(multi_layers_Adv_performance, multi_adv_performance_file)
                multi_adv_performance_file.close()


















