from transformers import AutoTokenizer, RobertaConfig
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import IntervalStrategy
import os
import numpy as np

from global_config import num_of_labels
from mode_roberta import RobertaForSequenceClassification
from load_data import load_data
from datasets import load_metric

os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def sentence_tokenizer(dataset, model_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding='max_length', max_length=512, truncation=True,)
    if dataset["train"] is not None:
        tokenized_train_data = dataset["train"].map(tokenize_function, batched=True)
    else:
        tokenized_train_data = None
    if dataset["test"] is not None:
        tokenized_test_data = dataset["test"].map(tokenize_function, batched=True)
    else:
        tokenized_test_data = None

    return tokenized_train_data, tokenized_test_data


def roberta_models(id_dataset, model_checkpoint, train_data, test_data, output_hidden_states=False):

    metric = load_metric("accuracy")
    batch_size = 16
    config = RobertaConfig.from_pretrained(model_checkpoint, num_labels=num_of_labels[id_dataset],
                                           output_hidden_states=output_hidden_states)
    roberta_model_for_classification = RobertaForSequenceClassification.from_pretrained(model_checkpoint, config=config)

    # Training the model
    print(f'fine tune models on {model_checkpoint} using {id_dataset} dataset...')
    args = TrainingArguments(
        f"./models/roberta-base-finetuned-{id_dataset}",
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
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        roberta_model_for_classification,
        args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
    )

    return trainer


def train_models(id_dataset):
    if id_dataset == 'sst2':
        pretrained_checkpoint = "./models/pretrained_model/roberta-base"
        finetuned_checkpoint = "./models/roberta-base-finetuned-sst2/best"

    elif id_dataset == 'imdb':
        pretrained_checkpoint = "./models/pretrained_model/roberta-base"
        finetuned_checkpoint = "./models/roberta-base-finetuned-imdb/best"

    elif id_dataset == 'trec':
        pretrained_checkpoint = "./models/pretrained_model/roberta-base"
        finetuned_checkpoint = "./models/roberta-base-finetuned-trec/best"

    elif id_dataset == 'newsgroup':
        pretrained_checkpoint = "./models/pretrained_model/roberta-base"
        finetuned_checkpoint = "./models/roberta-base-finetuned-newsgroup/best"

    else:
        print('dataset is not exist.')
        return

    dataset = load_data(id_dataset)

    if os.path.exists(f'{finetuned_checkpoint}/pytorch_model.bin'):
        print(f'finetuned model is existed! \n {finetuned_checkpoint}/pytorch_model.bin')
        train_data, test_data = sentence_tokenizer(dataset, finetuned_checkpoint)
        trainer = roberta_models(id_dataset=id_dataset,
                                 model_checkpoint=finetuned_checkpoint,
                                 train_data=train_data,
                                 test_data=test_data,
                                 output_hidden_states=False)
        metrics = trainer.evaluate()
        trainer.log_metrics(split='test', metrics=metrics)

    else:
        train_data, test_data = sentence_tokenizer(dataset, pretrained_checkpoint)
        trainer = roberta_models(id_dataset=id_dataset,
                                 model_checkpoint=pretrained_checkpoint,
                                 train_data=train_data,
                                 test_data=test_data,
                                 output_hidden_states=False)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint)
        tokenizer.save_pretrained(f"./models/roberta-base-finetuned-{id_dataset}/best")
        train_results = trainer.train()
        trainer.save_model(f"./models/roberta-base-finetuned-{id_dataset}/best")
        trainer.log_metrics(split="train", metrics=train_results.metrics)


if __name__ == '__main__':
    id_data = "sst2"
    # id_data = "imdb"
    # id_data = "trec"
    # id_data = 'newsgroup'

    train_models(id_data)







