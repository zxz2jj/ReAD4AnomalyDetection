import pickle
import os
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset

from load_data import load_data, load_clean_adv_data
from ReAD import get_neural_value, statistic_of_neural_value, \
    encode_abstraction, concatenate_data_between_layers, k_means, statistic_distance, auroc, sk_auc
from global_config import num_of_labels, roberta_config
from train_roberta_models import sentence_tokenizer


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_DISABLED"] = "true"


if __name__ == "__main__":

    clean_dataset = 'sst2'
    # clean_dataset = 'trec'
    # clean_dataset = 'newsgroup'

    finetuned_checkpoint = f"./models/roberta-base-finetuned-{clean_dataset}/best"
    detector_path = f'./data/{clean_dataset}/detector/'
    dataset = load_data(clean_dataset)
    train_data, test_data = sentence_tokenizer(dataset, finetuned_checkpoint)

    train_nv_statistic = None
    k_means_centers = None
    distance_train_statistic = None
    if os.path.exists(detector_path + 'train_nv_statistic.pkl') and \
            os.path.exists(detector_path + 'k_means_centers.pkl') and \
            os.path.exists(detector_path + 'distance_of_ReAD_for_train_data.pkl'):
        print("\nDetector is existed!")
        file1 = open(detector_path + 'train_nv_statistic.pkl', 'rb')
        file2 = open(detector_path + 'k_means_centers.pkl', 'rb')
        file3 = open(detector_path + 'distance_of_ReAD_for_train_data.pkl', 'rb')
        train_nv_statistic = pickle.load(file1)
        k_means_centers = pickle.load(file2)
        distance_train_statistic = pickle.load(file3)

    else:
        if not os.path.exists(f'./data/{clean_dataset}'):
            os.mkdir(f'./data/{clean_dataset}')
        if not os.path.exists(detector_path):
            os.mkdir(detector_path)
        # ********************** Train Detector **************************** #
        print('\n********************** Train Detector ****************************')
        print('\nGet neural value of train dataset:')
        train_picture_neural_value = get_neural_value(id_dataset=clean_dataset, dataset=train_data,
                                                      checkpoint=finetuned_checkpoint, is_anomaly=False)
        print('\nStatistic of train data neural value:')
        train_nv_statistic = statistic_of_neural_value(id_dataset=clean_dataset,
                                                       neural_value=train_picture_neural_value)
        print('finished!')

        file1 = open(detector_path + 'train_nv_statistic.pkl', 'wb')
        pickle.dump(train_nv_statistic, file1)
        file1.close()

        print('\nEncoding ReAD abstractions of train dataset:')
        train_ReAD_abstractions = encode_abstraction(id_dataset=clean_dataset, neural_value=train_picture_neural_value,
                                                     train_dataset_statistic=train_nv_statistic)
        train_ReAD_concatenated = \
            concatenate_data_between_layers(id_dataset=clean_dataset, data_dict=train_ReAD_abstractions)

        print('\nK-Means Clustering of Combination Abstraction on train data:')
        k_means_centers = k_means(data=train_ReAD_concatenated,
                                  category_number=num_of_labels[clean_dataset])
        file2 = open(detector_path + 'k_means_centers.pkl', 'wb')
        pickle.dump(k_means_centers, file2)
        file2.close()

        print('\nCalculate distance between abstractions and cluster centers ...')
        distance_train_statistic = statistic_distance(id_dataset=clean_dataset, abstractions=train_ReAD_concatenated,
                                                      cluster_centers=k_means_centers)
        file3 = open(detector_path + 'distance_of_ReAD_for_train_data.pkl', 'wb')
        pickle.dump(distance_train_statistic, file3)
        file3.close()

    # ********************** Evaluate Adversarial Detection **************************** #
    print('\n********************** Evaluate Adversarial Detection ****************************')
    attacks = roberta_config[clean_dataset]['adversarial_settings']
    for attack in attacks:
        print(f'\nATTACK: {attack}')
        clean_data, adv_data = load_clean_adv_data(clean_dataset=clean_dataset, attack=attack)
        clean_data = sentence_tokenizer(clean_data, finetuned_checkpoint)
        adv_data = sentence_tokenizer(adv_data, finetuned_checkpoint)

        print('\nGet neural value of CLEAN data:')
        clean_picture_neural_value = get_neural_value(id_dataset=clean_dataset, dataset=clean_data,
                                                      checkpoint=finetuned_checkpoint, is_anomaly=False)
        print('\nGet neural value of ADVERSARIAL data:')
        adv_picture_neural_value = get_neural_value(id_dataset=clean_dataset, dataset=adv_data,
                                                    checkpoint=finetuned_checkpoint, is_anomaly=True)

        print('\nEncoding ReAD abstraction of CLEAN data:')
        clean_ReAD_abstractions = encode_abstraction(id_dataset=clean_dataset, neural_value=clean_picture_neural_value,
                                                     train_dataset_statistic=train_nv_statistic)
        print('\nEncoding ReAD abstraction of ADVERSARIAL data:')
        adv_ReAD_abstractions = encode_abstraction(id_dataset=clean_dataset, neural_value=adv_picture_neural_value,
                                                   train_dataset_statistic=train_nv_statistic)

        clean_ReAD_concatenated = concatenate_data_between_layers(id_dataset=clean_dataset,
                                                                  data_dict=clean_ReAD_abstractions)
        adv_ReAD_concatenated = concatenate_data_between_layers(id_dataset=clean_dataset,
                                                                data_dict=adv_ReAD_abstractions)

        print('\nCalculate distance of CLEAN data:')
        clean_distance = statistic_distance(id_dataset=clean_dataset, abstractions=clean_ReAD_concatenated,
                                            cluster_centers=k_means_centers)
        print('\nCalculate distance of CLEAN data:')
        adv_distance = statistic_distance(id_dataset=clean_dataset, abstractions=adv_ReAD_concatenated,
                                          cluster_centers=k_means_centers)

        # auc = auroc(distance_train_statistic, clean_distance, adv_distance, num_of_labels[clean_dataset])
        auc = sk_auc(clean_dataset, clean_distance, adv_distance)

        print('\nPerformance of Detector:')
        print('AUROC: {:.6f}'.format(auc))
        print('*************************************\n')

        # distance_list = []
        # for i in range(10):
        #     distance_list.append(clean_distance[i]['correct_pictures'])
        #     distance_list.append(adv_distance[i]['wrong_pictures'])
        # data = {'T0': distance_list[0]}
        # data = pd.DataFrame(data, columns=['T0'])
        # data['F0'] = pd.Series(distance_list[1])
        # data['T1'] = pd.Series(distance_list[2])
        # data['F1'] = pd.Series(distance_list[3])
        # data['T2'] = pd.Series(distance_list[4])
        # data['F2'] = pd.Series(distance_list[5])
        # data['T3'] = pd.Series(distance_list[6])
        # data['F3'] = pd.Series(distance_list[7])
        # data['T4'] = pd.Series(distance_list[8])
        # data['F4'] = pd.Series(distance_list[9])
        # data['T5'] = pd.Series(distance_list[10])
        # data['F5'] = pd.Series(distance_list[11])
        # data['T6'] = pd.Series(distance_list[12])
        # data['F6'] = pd.Series(distance_list[13])
        # data['T7'] = pd.Series(distance_list[14])
        # data['F7'] = pd.Series(distance_list[15])
        # data['T8'] = pd.Series(distance_list[16])
        # data['F8'] = pd.Series(distance_list[17])
        # data['T9'] = pd.Series(distance_list[18])
        # data['F9'] = pd.Series(distance_list[19])
        # data = pd.DataFrame(data)
        # sns.boxplot(data=data)
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        # plt.show()

