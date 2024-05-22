import numpy as np
from tqdm import tqdm
from datasets import load_dataset, load_metric,DatasetDict
from transformers import AutoFeatureExtractor, TrainingArguments, Trainer, SwinConfig
from transformers.trainer_utils import IntervalStrategy
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

import torch
import os

from model_swin_transformer import SwinModel, SwinForImageClassification
from load_data import load_gtsrb
from global_config import num_of_labels

os.environ["WANDB_DISABLED"] = "true"


def image_tokenizer(data, model_checkpoint, mode):

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    # preprocessing the data
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    train_transforms = Compose(
        [
            RandomResizedCrop((feature_extractor.size['height'], feature_extractor.size['width'])),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    test_transforms = Compose(
        [
            Resize((feature_extractor.size['height'], feature_extractor.size['width'])),
            CenterCrop((feature_extractor.size['height'], feature_extractor.size['width'])),
            ToTensor(),
            normalize,
        ]
    )

    def preprocess_train(example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    def preprocess_test(example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [test_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    if mode == 'train':
        data.set_transform(preprocess_train)
    elif mode == 'test':
        data.set_transform(preprocess_test)
    else:
        print('mode must be \'train\' or \'test\'. ')
        return None

    return data


def swin_models(model_checkpoint, dataset, train_data, test_data, output_hidden_states=False, mode='test'):

    metric = load_metric("accuracy")
    batch_size = 64
    if mode == 'train':
        swin_model = SwinModel.from_pretrained(model_checkpoint)
        config = SwinConfig.from_pretrained(model_checkpoint, num_labels=num_of_labels[dataset],
                                            output_hidden_states=output_hidden_states, ignore_mismatched_sizes=True,)
        swin_model_for_classification = SwinForImageClassification(config)
        swin_model_for_classification.swin = swin_model
    elif mode == 'test':
        swin_model_for_classification = SwinForImageClassification.from_pretrained(model_checkpoint)
    else:
        print('mode must be train or test.')
        exit()
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    print(f'fine tune models on {model_checkpoint} using {dataset} dataset...')
    args = TrainingArguments(
        f"./models/swin-finetuned-{dataset}",
        remove_unused_columns=False,
        evaluation_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        save_total_limit=3,
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    trainer = Trainer(
        swin_model_for_classification,
        args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    return trainer


def train_models(id_dataset):
    if id_dataset == 'mnist':
        row_names = ('image', 'label')
        pretrained_checkpoint = "./models/pretrained_model/swin-tiny-patch4-window7-224"
        finetuned_checkpoint = "./models/swin-finetuned-mnist/best"
        dataset = load_dataset("./data/mnist/mnist/", cache_dir='./dataset/')

    elif id_dataset == 'fashion_mnist':
        row_names = ('image', 'label')
        pretrained_checkpoint = "./models/pretrained_model/swin-tiny-patch4-window7-224"
        finetuned_checkpoint = "./models/swin-finetuned-fashion_mnist/best"
        dataset = load_dataset("./data/fashion_mnist/fashion_mnist/", cache_dir='./dataset/')

    elif id_dataset == 'cifar10':
        row_names = ('img', 'label')
        pretrained_checkpoint = "./models/pretrained_model/swin-tiny-patch4-window7-224"
        finetuned_checkpoint = "./models/swin-finetuned-cifar10/best"
        dataset = load_dataset("./data/cifar10/cifar10/", cache_dir='./dataset/')

    elif id_dataset == 'svhn':
        row_names = ('image', 'label')
        pretrained_checkpoint = "./models/pretrained_model/swin-tiny-patch4-window7-224"
        finetuned_checkpoint = "./models/swin-finetuned-svhn/best"
        dataset = load_dataset('./data/svhn/svhn/', 'cropped_digits', cache_dir='./dataset/')
        del dataset['extra']

    elif id_dataset == 'gtsrb':
        row_names = ('image', 'label')
        pretrained_checkpoint = "./models/pretrained_model/swin-tiny-patch4-window7-224"
        finetuned_checkpoint = "./models/swin-finetuned-gtsrb/best"
        dataset = load_gtsrb()

    elif id_dataset == 'imagenet100':
        row_names = ('image', 'label')
        pretrained_checkpoint = "./models/pretrained_model/swin-tiny-patch4-window7-224"
        finetuned_checkpoint = "./models/swin-finetuned-imagenet100/best"
        dataset = load_dataset("./data/imagenet100/imagenet100/", cache_dir='./dataset/')
        dataset = DatasetDict({'train': dataset['train'], 'test': dataset['validation']})

    elif id_dataset == 'cifar100':
        row_names = ('img', 'fine_label')
        pretrained_checkpoint = "./models/pretrained_model/swin-tiny-patch4-window7-224"
        finetuned_checkpoint = "./models/swin-finetuned-cifar100/best"
        dataset = load_dataset("./data/cifar100/cifar100/", cache_dir='./dataset/')
    else:
        print('dataset is not exist.')
        return

    train_data = dataset['train']
    test_data = dataset['test']
    if row_names[0] != 'image':
        train_data = train_data.rename_column(row_names[0], 'image')
        test_data = test_data.rename_column(row_names[0], 'image')
    if row_names[1] != 'label':
        train_data = train_data.rename_column(row_names[1], 'label')
        test_data = test_data.rename_column(row_names[1], 'label')

    if os.path.exists(f'{finetuned_checkpoint}/pytorch_model.bin'):
        print(f'finetuned model is existed! \n {finetuned_checkpoint}/pytorch_model.bin')
        train_data_tokenized = image_tokenizer(data=train_data, model_checkpoint=finetuned_checkpoint, mode='test')
        test_dataset_tokenized = image_tokenizer(data=test_data, model_checkpoint=finetuned_checkpoint, mode='test')
        trainer = swin_models(model_checkpoint=finetuned_checkpoint,
                              dataset=id_dataset,
                              train_data=train_data_tokenized,
                              test_data=test_dataset_tokenized,
                              output_hidden_states=False,
                              mode='test')
        metrics = trainer.evaluate()
        trainer.log_metrics('test', metrics)

    else:
        train_data_tokenized = image_tokenizer(data=train_data, model_checkpoint=pretrained_checkpoint, mode='train')
        test_dataset_tokenized = image_tokenizer(data=test_data, model_checkpoint=pretrained_checkpoint, mode='train')
        trainer = swin_models(model_checkpoint=pretrained_checkpoint,
                              dataset=id_dataset,
                              train_data=train_data_tokenized,
                              test_data=test_dataset_tokenized,
                              output_hidden_states=False,
                              mode='train')
        train_results = trainer.train()
        trainer.save_model(f"./models/swin-finetuned-{id_dataset}/best")
        trainer.log_metrics("train", train_results.metrics)


if __name__ == '__main__':

    # id_data = 'mnist'
    # train_models(id_data)
    # id_data = 'fashion_mnist'
    # train_models(id_data)
    # id_data = 'cifar10'
    # train_models(id_data)
    # id_data = 'svhn'
    # train_models(id_data)
    # id_data = 'gtsrb'
    # train_models(id_data)
    # id_data = 'imagenet100'
    # train_models(id_data)
    id_data = 'cifar100'
    train_models(id_data)
