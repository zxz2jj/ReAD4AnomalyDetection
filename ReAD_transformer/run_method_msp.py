import numpy as np
from sklearn.metrics import auc, roc_curve
from transformers import RobertaConfig, RobertaForSequenceClassification
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

import global_config
from load_data import load_data, load_clean_adv_data
from train_roberta_models import sentence_tokenizer


def get_confidence(dataset_name, data, checkpoint, is_ood=False):
    config = RobertaConfig.from_pretrained(checkpoint, num_labels=global_config.num_of_labels[dataset_name],
                                           output_hidden_states=False, ignore_mismatched_sizes=True)
    model = RobertaForSequenceClassification.from_pretrained(checkpoint, config=config)
    labels = data['label'] if not is_ood else None

    probability_list = list()
    predictions_list = list()
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}, BatchSize: {batch_size}')
    model.to(device)
    batches = DataLoader(dataset=data, batch_size=batch_size)
    for batch in tqdm(batches):
        input_ids = torch.stack(batch['input_ids']).transpose(0, 1)
        attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1)
        outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        prediction = torch.argmax(outputs.logits, 1)
        probability = torch.softmax(torch.squeeze(outputs.logits), dim=-1)
        if probability.dim() == 1:
            probability = torch.unsqueeze(probability, 0)
        predictions_list.append(prediction.cpu().detach().numpy())
        probability_list.append(probability.cpu().detach().numpy())
    probabilities = np.concatenate(probability_list)
    predictions = np.concatenate(predictions_list)

    confidence = list()
    if not is_ood:
        new_probabilities = probabilities[predictions == labels]
        new_predictions = predictions[predictions == labels]
        for pred, prob in zip(new_predictions, new_probabilities):
            confidence.append(prob[pred])
    else:
        for pred, prob in zip(predictions, probabilities):
            confidence.append(prob[pred])

    return confidence


if __name__ == "__main__":

    # id_dataset = 'sst2'
    # ood_dataset = ['trec', 'newsgroup', 'mnli', 'rte', 'wmt16', 'multi30k', 'noise']
    # attacks = ['SCPNAttacker', 'GANAttacker', 'TextFoolerAttacker', 'PWWSAttacker', 'TextBuggerAttacker', 'VIPERAttacker']

    # id_dataset = 'imdb'
    # ood_dataset = ['trec', 'newsgroup', 'mnli', 'rte', 'wmt16', 'multi30k', 'noise']
    # neuron_value_path = './data/imdb/'

    # id_dataset = 'trec'
    # ood_dataset = ['sst2', 'imdb', 'newsgroup', 'mnli', 'rte', 'wmt16', 'multi30k', 'noise']
    # attacks = ['SCPNAttacker', 'GANAttacker', 'TextFoolerAttacker', 'PWWSAttacker', 'TextBuggerAttacker', 'VIPERAttacker']

    id_dataset = 'newsgroup'
    ood_dataset = ['sst2', 'imdb', 'trec', 'mnli', 'rte', 'wmt16', 'multi30k', 'noise']
    attacks = ['SCPNAttacker', 'GANAttacker', 'TextFoolerAttacker', 'PWWSAttacker', 'TextBuggerAttacker', 'VIPERAttacker']

    finetuned_checkpoint = f"./models/roberta-base-finetuned-{id_dataset}/best"
    dataset = load_data(id_dataset)
    train_data, test_data = sentence_tokenizer(dataset, finetuned_checkpoint)

    test_confidence = get_confidence(dataset_name=id_dataset, data=test_data,
                                     checkpoint=finetuned_checkpoint, is_ood=False)
    for ood in ood_dataset:
        print('\nOOD dataset: {}'.format(ood))
        ood_data = load_data(ood)
        _, ood_data = sentence_tokenizer(ood_data, finetuned_checkpoint)
        ood_confidence = get_confidence(dataset_name=id_dataset, data=ood_data,
                                        checkpoint=finetuned_checkpoint, is_ood=True)
        y_true = [0 for _ in range(ood_confidence.__len__())] + [1 for _ in range(test_confidence.__len__())]
        y_score = ood_confidence + test_confidence
        fpr, tpr, threshold = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        target_tpr = 0.95
        target_threshold_idx = np.argmax(tpr >= target_tpr)
        target_fpr = fpr[target_threshold_idx]
        print('\nAUROC: {:.6f}, FAR95: {:.6f}, TPR: {}'.format(roc_auc, target_fpr, tpr[target_threshold_idx]))
        print('*************************************')

    for attack in attacks:
        print(f'\nATTACK: {attack}')
        clean_data, adv_data = load_clean_adv_data(clean_dataset=id_dataset, attack=attack)
        _, clean_data = sentence_tokenizer(clean_data, finetuned_checkpoint)
        _, adv_data = sentence_tokenizer(adv_data, finetuned_checkpoint)

        clean_confidence = get_confidence(dataset_name=id_dataset, data=clean_data,
                                          checkpoint=finetuned_checkpoint, is_ood=False)
        adv_confidence = get_confidence(dataset_name=id_dataset, data=adv_data,
                                        checkpoint=finetuned_checkpoint, is_ood=True)

        y_true = [0 for _ in range(adv_confidence.__len__())] + [1 for _ in range(clean_confidence.__len__())]
        y_score = adv_confidence + clean_confidence
        fpr, tpr, threshold = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        target_tpr = 0.95
        target_threshold_idx = np.argmax(tpr >= target_tpr)
        target_fpr = fpr[target_threshold_idx]
        print('\nAUROC: {:.6f}, FAR95: {:.6f}, TPR: {}'.format(roc_auc, target_fpr, tpr[target_threshold_idx]))
        print('*************************************')


