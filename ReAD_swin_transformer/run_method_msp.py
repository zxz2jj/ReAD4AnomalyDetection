import os.path
import torch
from datasets import load_dataset, Dataset
import numpy as np
from tqdm import tqdm
import math
from transformers import SwinConfig

import global_config
from global_config import swin_config
from load_data import load_gtsrb, load_ood_data
from model_swin_transformer import SwinForImageClassification
from train_swin_models import image_tokenizer
from torch.utils.data import Subset
from sklearn.metrics import auc, roc_curve


os.environ["WANDB_DISABLED"] = "true"


def get_confidence(id_dataset, data, checkpoint, is_ood=False):

    config = SwinConfig.from_pretrained(checkpoint, num_labels=global_config.num_of_labels[id_dataset],
                                        output_hidden_states=False, ignore_mismatched_sizes=True)
    labels = data['label'] if not is_ood else None
    model = SwinForImageClassification.from_pretrained(checkpoint, config=config)
    data = image_tokenizer(data=data, model_checkpoint=checkpoint, mode='test')

    probability_list = list()
    predictions_list = list()
    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}, BatchSize: {batch_size}')
    model.to(device)
    batch_index_start = 0
    batch_index_end = batch_size
    for _ in tqdm(range(math.ceil(data.shape[0] / batch_size))):
        batch = data.__getitem__([_ for _ in range(batch_index_start, batch_index_end)])
        pixel_values = torch.stack(batch['pixel_values'])
        outputs = model(pixel_values.to(device))
        prediction = torch.argmax(outputs.logits, 1)
        probability = torch.softmax(torch.squeeze(outputs.logits), dim=-1)
        predictions_list.append(prediction.cpu().detach().numpy())
        probability_list.append(probability.cpu().detach().numpy())
        batch_index_start = batch_index_end
        batch_index_end = (batch_index_end + batch_size) \
            if (batch_index_end + batch_size) < data.shape[0] else data.shape[0]
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

    # dataset_name = 'mnist'
    # dataset_name = 'fashion_mnist'
    dataset_name = 'svhn'
    # dataset_name = 'cifar100'
    # dataset_name = 'cifar10'
    # dataset_name = 'gtsrb'

    if dataset_name == 'mnist':
        row_names = ('image', 'label')
        finetuned_checkpoint = "./models/swin-finetuned-mnist/best"
        dataset = load_dataset("./data/mnist/mnist/", cache_dir='./dataset/')

    elif dataset_name == 'fashion_mnist':
        row_names = ('image', 'label')
        finetuned_checkpoint = "./models/swin-finetuned-fashion_mnist/best"
        dataset = load_dataset("./data/fashion_mnist/fashion_mnist/", cache_dir='./dataset/')

    elif dataset_name == 'svhn':
        row_names = ('image', 'label')
        finetuned_checkpoint = "./models/swin-finetuned-svhn/best"
        dataset = load_dataset('./data/svhn/svhn/', 'cropped_digits', cache_dir='./dataset/')
        del dataset['extra']
        detector_path = './data/svhn/detector/'

    elif dataset_name == 'cifar10':
        row_names = ('img', 'label')
        finetuned_checkpoint = "./models/swin-finetuned-cifar10/best"
        dataset = load_dataset("./data/cifar10/cifar10/", cache_dir='./dataset/')

    elif dataset_name == 'cifar100':
        row_names = ('img', 'fine_label')
        finetuned_checkpoint = "./models/swin-finetuned-cifar100/best"
        dataset = load_dataset("./data/cifar100/cifar100/", cache_dir='./dataset/')

    elif dataset_name == 'gtsrb':
        row_names = ('image', 'label')
        finetuned_checkpoint = "./models/swin-finetuned-gtsrb/best"
        dataset = load_gtsrb()

    else:
        print('dataset is not exist.')
        exit()

    print(f'Get ID-{dataset_name} Confidence File ...')
    rename_test_data = Dataset.from_dict({'image': dataset['test'][row_names[0]],
                                          'label': dataset['test'][row_names[1]]})
    test_confidence = get_confidence(id_dataset=dataset_name, data=rename_test_data,
                                     checkpoint=finetuned_checkpoint, is_ood=False)

    OOD_dataset = swin_config[dataset_name]['ood_settings']
    for ood in OOD_dataset:
        print(f'Get OOD-{ood} Confidence File ...')
        ood_data = load_ood_data(ood_dataset=ood)
        ood_confidence = get_confidence(id_dataset=dataset_name, data=ood_data,
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
