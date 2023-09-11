import os.path
import torch
from datasets import load_dataset, Dataset
import numpy as np
from transformers import SwinConfig

import global_config
from global_config import swin_config
from load_data import load_gtsrb, load_ood_data
from model_swin_transformer import SwinForImageClassification
from train_swin_models import image_tokenizer
from torch.utils.data import Subset
from sklearn.metrics import auc, roc_curve


os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["WANDB_DISABLED"] = "true"


def get_confidence(id_dataset, data, checkpoint, is_ood=False, ood_name=''):

    config = SwinConfig.from_pretrained(checkpoint, num_labels=global_config.num_of_labels[id_dataset],
                                        output_hidden_states=False, ignore_mismatched_sizes=True)
    model = SwinForImageClassification.from_pretrained(checkpoint, config=config)

    data = image_tokenizer(data=data, model_checkpoint=checkpoint, mode='test')
    data = Subset(data, range(0, data.shape[0])).__getitem__([_ for _ in range(data.shape[0])])
    # data = Subset(data, range(0, 100)).__getitem__([_ for _ in range(100)])

    confidence_list = list()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if not is_ood:
        for p, batch, label in zip(range(len(data['pixel_values'])), data['pixel_values'], data['label']):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print('\rImage: {} / {}'.format(p + 1, len(data['pixel_values'])), end='')
            outputs = model(torch.unsqueeze(batch, 0).to(device))
            prediction = torch.argmax(outputs.logits, 1)
            confidence = torch.softmax(torch.squeeze(outputs.logits), dim=-1)[prediction]
            if prediction == label:
                confidence_list.append(confidence.cpu().detach().numpy())
    else:
        for p, batch in zip(range(len(data['pixel_values'])), data['pixel_values']):
            print('\rImage: {} / {}'.format(p + 1, len(data['pixel_values'])), end='')
            outputs = model(torch.unsqueeze(batch, 0).to(device))
            prediction = torch.argmax(outputs.logits, 1)
            confidence = torch.softmax(torch.squeeze(outputs.logits), dim=-1)[prediction]
            confidence_list.append(confidence.cpu().detach().numpy())
    print('\n')
    if not os.path.exists(f'./data/{id_dataset}/hidden_states/'):
        os.mkdir(f'./data/{id_dataset}/hidden_states/')
    confidence = np.concatenate(confidence_list)
    if is_ood:
        np.save(f'./data/{id_dataset}/hidden_states/ood_{ood_name}_confidence.npy', confidence)
    else:
        np.save(f'./data/{id_dataset}/hidden_states/id_{id_dataset}_confidence.npy', confidence)


if __name__ == "__main__":

    dataset_name = 'mnist'
    # dataset_name = 'fashion_mnist'
    # dataset_name = 'cifar10'
    # dataset_name = 'gtsrb'

    if dataset_name == 'mnist':
        row_names = ('image', 'label')
        finetuned_checkpoint = "./models/swin-finetuned-mnist/best"
        dataset = load_dataset("./data/mnist/mnist/")

    elif dataset_name == 'fashion_mnist':
        row_names = ('image', 'label')
        finetuned_checkpoint = "./models/swin-finetuned-fashion_mnist/best"
        dataset = load_dataset("./data/fashion_mnist/fashion_mnist/")

    elif dataset_name == 'cifar10':
        row_names = ('img', 'label')
        finetuned_checkpoint = "./models/swin-finetuned-cifar10/best"
        dataset = load_dataset("./data/cifar10/cifar10/")

    elif dataset_name == 'gtsrb':
        row_names = ('image', 'label')
        finetuned_checkpoint = "./models/swin-finetuned-gtsrb/best"
        dataset = load_gtsrb()

    else:
        print('dataset is not exist.')
        exit()

    id_confidence_file = f'./data/{dataset_name}/hidden_states/id_{dataset_name}_confidence.npy'
    if os.path.exists(id_confidence_file):
        print(f'ID-{dataset_name} Confidence File is Existed!')
        test_confidence = np.load(id_confidence_file).tolist()
    else:
        print(f'Get ID-{dataset_name} Confidence File ...')
        rename_test_data = Dataset.from_dict({'image': dataset['test'][row_names[0]],
                                              'label': dataset['test'][row_names[1]]})
        get_confidence(id_dataset=dataset_name, data=rename_test_data, checkpoint=finetuned_checkpoint, is_ood=False)
        test_confidence = np.load(id_confidence_file).tolist()

    OOD_dataset = swin_config[dataset_name]['ood_settings']
    for ood in OOD_dataset:
        print('')
        ood_confidence_file = f'./data/{dataset_name}/hidden_states/ood_{ood}_confidence.npy'
        if os.path.exists(ood_confidence_file):
            print(f'OOD-{ood} Confidence File is Existed!')
            ood_confidence = np.load(ood_confidence_file).tolist()
        else:
            print(f'Get OOD-{ood} Confidence File ...')
            ood_data = load_ood_data(ood_dataset=ood)
            get_confidence(id_dataset=dataset_name, data=ood_data, checkpoint=finetuned_checkpoint,
                           is_ood=True, ood_name=ood)
            ood_confidence = np.load(ood_confidence_file).tolist()

        y_true = [0 for _ in range(ood_confidence.__len__())] + [1 for _ in range(test_confidence.__len__())]
        y_score = ood_confidence + test_confidence
        fpr, tpr, threshold = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        print('\nPerformance of Detector: AUROC: {:.6f}'.format(roc_auc))
        print('*************************************')
