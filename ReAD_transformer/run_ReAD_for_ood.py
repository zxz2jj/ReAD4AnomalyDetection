import pickle
import os
import time
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

from load_data import load_data
from ReAD import get_neural_value, statistic_of_neural_value, \
    encode_abstraction, concatenate_data_between_layers, k_means, statistic_distance, auroc, sk_auc
from global_config import num_of_labels, roberta_config
from train_roberta_models import sentence_tokenizer


os.environ["WANDB_DISABLED"] = "true"


if __name__ == "__main__":

    id_dataset = 'sst2'
    # id_dataset = 'imdb'
    # id_dataset = 'trec'
    # id_dataset = 'newsgroup'

    finetuned_checkpoint = f"./models/roberta-base-finetuned-{id_dataset}/best"
    detector_path = f'./data/{id_dataset}/detector/'
    dataset = load_data(id_dataset)
    train_data, test_data = sentence_tokenizer(dataset, finetuned_checkpoint)

    train_nv_statistic = None
    k_means_centers = None
    distance_train_statistic = None
    if os.path.exists(detector_path + 'train_nv_statistic.pkl') and \
            os.path.exists(detector_path + 'k_means_centers.pkl'):
        print("\nDetector is existed!")
        file1 = open(detector_path + 'train_nv_statistic.pkl', 'rb')
        file2 = open(detector_path + 'k_means_centers.pkl', 'rb')
        train_nv_statistic = pickle.load(file1)
        k_means_centers = pickle.load(file2)

    else:
        if not os.path.exists(f'./data/{id_dataset}'):
            os.mkdir(f'./data/{id_dataset}')
        if not os.path.exists(detector_path):
            os.mkdir(detector_path)
        # ********************** Train Detector **************************** #
        print('\n********************** Train Detector ****************************')
        train_start_time = time.time()

        print('\nGet neural value of train dataset:')
        train_picture_neural_value = get_neural_value(id_dataset=id_dataset, dataset=train_data,
                                                      checkpoint=finetuned_checkpoint, is_anomaly=False)
        print('\nStatistic of train data neural value:')
        train_nv_statistic = statistic_of_neural_value(id_dataset=id_dataset,
                                                       neural_value=train_picture_neural_value)
        print('finished!')

        file1 = open(detector_path + 'train_nv_statistic.pkl', 'wb')
        pickle.dump(train_nv_statistic, file1)
        file1.close()

        print('\nEncoding ReAD abstractions of train dataset:')
        train_ReAD_abstractions = encode_abstraction(id_dataset=id_dataset, neural_value=train_picture_neural_value,
                                                     train_dataset_statistic=train_nv_statistic)
        train_ReAD_concatenated = \
            concatenate_data_between_layers(id_dataset=id_dataset, data_dict=train_ReAD_abstractions)

        print('\nK-Means Clustering of ReAD on train data:')
        k_means_centers = k_means(data=train_ReAD_concatenated,
                                  category_number=num_of_labels[id_dataset])
        file2 = open(detector_path + 'k_means_centers.pkl', 'wb')
        pickle.dump(k_means_centers, file2)
        file2.close()

        train_end_time = time.time()
        train_total_time = train_end_time-train_start_time
        train_time_for_per_example = train_total_time / train_data.num_rows
        print(f"\nTrain Process: total time-{train_total_time}s, per example-{train_time_for_per_example}s")

    # ********************** Evaluate OOD Detection **************************** #
    print('\n********************** Evaluate OOD Detection ****************************')
    test_start_time = time.time()

    print('\nGet neural value of test dataset:')
    test_picture_neural_value = get_neural_value(id_dataset=id_dataset, dataset=test_data,
                                                 checkpoint=finetuned_checkpoint, is_anomaly=False)
    print('\nEncoding ReAD abstraction of test dataset:')
    test_ReAD_abstractions = encode_abstraction(id_dataset=id_dataset, neural_value=test_picture_neural_value,
                                                train_dataset_statistic=train_nv_statistic)
    test_ReAD_concatenated = concatenate_data_between_layers(id_dataset=id_dataset, data_dict=test_ReAD_abstractions)
    print('\nCalculate distance between abstractions and cluster centers ...')
    test_distance = statistic_distance(id_dataset=id_dataset, abstractions=test_ReAD_concatenated,
                                       cluster_centers=k_means_centers)
    test_end_time = time.time()
    test_total_time = test_end_time - test_start_time
    test_time_for_per_example = test_total_time / test_data.num_rows
    print(f"\nDetection Process(test): total time-{test_total_time}s, per example-{test_time_for_per_example}s")

    OOD_dataset = roberta_config[id_dataset]['ood_settings']
    for ood in OOD_dataset:
        print('\n************Evaluating*************')
        ood_start_time = time.time()
        print(f'In-Distribution Data: {id_dataset}, Out-of-Distribution Data: {ood}.')
        ood_data = load_data(ood)
        _, ood_data = sentence_tokenizer(ood_data, finetuned_checkpoint)
        print('\nGet neural value of ood dataset...')
        ood_picture_neural_value = get_neural_value(id_dataset=id_dataset, dataset=ood_data,
                                                    checkpoint=finetuned_checkpoint,
                                                    is_anomaly=True)
        print('\nEncoding ReAD abstraction of ood dataset...')
        ood_ReAD_abstractions = encode_abstraction(id_dataset=id_dataset, neural_value=ood_picture_neural_value,
                                                   train_dataset_statistic=train_nv_statistic)
        ood_ReAD_concatenated = concatenate_data_between_layers(id_dataset=id_dataset, data_dict=ood_ReAD_abstractions)
        print('\nCalculate distance between abstractions and cluster centers ...')
        ood_distance = statistic_distance(id_dataset=id_dataset, abstractions=ood_ReAD_concatenated,
                                          cluster_centers=k_means_centers)
        ood_end_time = time.time()
        ood_total_time = ood_end_time - ood_start_time
        ood_time_for_per_example = ood_total_time / ood_data.num_rows
        auc = sk_auc(id_dataset, test_distance, ood_distance)
        print(f"Detection Process(ood): total time-{ood_total_time}s, per example-{ood_time_for_per_example}s")
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

