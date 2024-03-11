import os

from load_data import load_data, load_clean_adv_data
from ReAD import get_neural_value, statistic_of_neural_value, \
    encode_abstraction, concatenate_data_between_layers, k_means, statistic_distance, auroc, sk_auc
from global_config import num_of_labels, roberta_config
from train_roberta_models import sentence_tokenizer

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"


if __name__ == "__main__":

    # id_dataset = 'sst2'
    # id_dataset = 'imdb'
    # id_dataset = 'trec'
    id_dataset = 'newsgroup'

    finetuned_checkpoint = f"./models/roberta-base-finetuned-{id_dataset}/best"
    dataset = load_data(id_dataset)
    train_data, test_data = sentence_tokenizer(dataset, finetuned_checkpoint)

    # ********************** Train Detector **************************** #
    print('\n********************** Train Detector ****************************')

    print('\nGet neural value of train dataset:')
    train_picture_neural_value = get_neural_value(id_dataset=id_dataset, dataset=train_data,
                                                  checkpoint=finetuned_checkpoint, is_anomaly=False)
    train_nad_concatenated = \
        concatenate_data_between_layers(id_dataset=id_dataset, data_dict=train_picture_neural_value)
    print('\nK-Means Clustering of Combination Abstraction on train data:')
    k_means_centers = k_means(data=train_nad_concatenated,
                              category_number=num_of_labels[id_dataset])

    # ********************** Evaluate OOD Detection **************************** #
    print('\n********************** Evaluate OOD Detection ****************************')
    print('\nGet neural value of test dataset:')
    test_picture_neural_value = get_neural_value(id_dataset=id_dataset, dataset=test_data,
                                                 checkpoint=finetuned_checkpoint, is_anomaly=False)
    test_nad_concatenated = concatenate_data_between_layers(id_dataset=id_dataset, data_dict=test_picture_neural_value)
    print('\nCalculate distance between abstractions and cluster centers ...')
    test_distance = statistic_distance(id_dataset=id_dataset, abstractions=test_nad_concatenated,
                                       cluster_centers=k_means_centers)
    OOD_dataset = roberta_config[id_dataset]['ood_settings']
    for ood in OOD_dataset:
        print('\n************Evaluating*************')
        print(f'In-Distribution Data: {id_dataset}, Out-of-Distribution Data: {ood}.')
        ood_data = load_data(ood)
        _, ood_data = sentence_tokenizer(ood_data, finetuned_checkpoint)
        print('\nGet neural value of ood dataset...')
        ood_picture_neural_value = get_neural_value(id_dataset=id_dataset, dataset=ood_data,
                                                    checkpoint=finetuned_checkpoint,
                                                    is_anomaly=True)
        ood_nad_concatenated = concatenate_data_between_layers(id_dataset=id_dataset, data_dict=ood_picture_neural_value)
        print('\nCalculate distance between abstractions and cluster centers ...')
        ood_distance = statistic_distance(id_dataset=id_dataset, abstractions=ood_nad_concatenated,
                                          cluster_centers=k_means_centers)
        auc = sk_auc(id_dataset, test_distance, ood_distance)
        print('*************************************\n')

    # ********************** Evaluate Adversarial Detection **************************** #
    print('\n********************** Evaluate Adversarial Detection ****************************')
    attacks = roberta_config[id_dataset]['adversarial_settings']
    for attack in attacks:
        print(f'\nATTACK: {attack}')
        clean_data, adv_data = load_clean_adv_data(clean_dataset=id_dataset, attack=attack)
        _, clean_data = sentence_tokenizer(clean_data, finetuned_checkpoint)
        _, adv_data = sentence_tokenizer(adv_data, finetuned_checkpoint)

        print('\nGet neural value of CLEAN data:')
        clean_picture_neural_value = get_neural_value(id_dataset=id_dataset, dataset=clean_data,
                                                      checkpoint=finetuned_checkpoint, is_anomaly=False)
        print('\nGet neural value of ADVERSARIAL data:')
        adv_picture_neural_value = get_neural_value(id_dataset=id_dataset, dataset=adv_data,
                                                    checkpoint=finetuned_checkpoint, is_anomaly=True)

        clean_nad_concatenated = concatenate_data_between_layers(id_dataset=id_dataset,
                                                                 data_dict=clean_picture_neural_value)
        adv_nad_concatenated = concatenate_data_between_layers(id_dataset=id_dataset,
                                                               data_dict=adv_picture_neural_value)

        print('\nCalculate distance of CLEAN data:')
        clean_distance = statistic_distance(id_dataset=id_dataset, abstractions=clean_nad_concatenated,
                                            cluster_centers=k_means_centers)
        print('\nCalculate distance of CLEAN data:')
        adv_distance = statistic_distance(id_dataset=id_dataset, abstractions=adv_nad_concatenated,
                                          cluster_centers=k_means_centers)

        sk_auc(id_dataset, clean_distance, adv_distance)

        print('*************************************\n')

