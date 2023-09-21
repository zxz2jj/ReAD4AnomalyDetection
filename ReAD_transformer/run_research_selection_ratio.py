import matplotlib
matplotlib.get_backend()
import matplotlib.pyplot as plt
import pickle
import os
from load_data import load_data, load_clean_adv_data
from ReAD import statistic_of_neural_value, get_neural_value
from ReAD import encode_abstraction, concatenate_data_between_layers, k_means, statistic_distance, sk_auc
import global_config
from global_config import num_of_labels, roberta_config
from train_roberta_models import sentence_tokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["WANDB_DISABLED"] = "true"


if __name__ == "__main__":

    if os.path.exists(f'./data/sst2/selection_ratio_adversarial.pkl') and \
            os.path.exists(f'./data/trec/selection_ratio_adversarial.pkl') and \
            os.path.exists(f'./data/newsgroup/selection_ratio_adversarial.pkl'):
        print('Research is finished!')

        file1 = open(f'./data/sst2/selection_ratio_adversarial.pkl', 'rb')
        file2 = open(f'./data/trec/selection_ratio_adversarial.pkl', 'rb')
        file3 = open(f'./data/newsgroup/selection_ratio_adversarial.pkl', 'rb')
        sst2_performance = pickle.load(file1)
        trec_performance = pickle.load(file2)
        newsgroup_performance = pickle.load(file3)
        attacks = ['SCPNAttacker', 'GANAttacker', 'TextFoolerAttacker',
                   'PWWSAttacker', 'TextBuggerAttacker', 'VIPERAttacker']

        fig = plt.figure(figsize=(9, 9))
        x = [0.01*a for a in range(1, 101)]
        for attack in attacks:
            if sst2_performance[attack]:
                plt.plot(x, sst2_performance[attack], label=f'SST2 vs {attack}')
            if trec_performance[attack]:
                plt.plot(x, trec_performance[attack], label=f'TREC-6 vs {attack}')
            if newsgroup_performance[attack]:
                plt.plot(x, newsgroup_performance[attack], label=f'Cifar10 vs {attack}')
        plt.legend(loc='lower right', prop={'size': 13})
        plt.tick_params(labelsize=20)
        plt.ylim(0.8, 1)
        ax = plt.gca()
        y_major_locator = plt.MultipleLocator(0.05)
        ax.yaxis.set_major_locator(y_major_locator)
        ax.vlines(0.6, 0.7, 1.0, linestyles='--', colors='darkgray')
        plt.savefig('./data/research_selection_ratio_adversarial.pdf')
        plt.show()
        exit()

    rate = [x for x in range(1, 101)]

    id_dataset = 'sst2'
    # id_dataset = 'trec'
    # id_dataset = 'newsgroup'

    if os.path.exists(f'./data/{id_dataset}/selection_ratio_adversarial.pkl'):
        print(f'Research on Dataset {id_dataset} is done!')
        exit()
    adversarial_performance = {'SCPNAttacker': [], 'GANAttacker': [], 'TextFoolerAttacker': [],
                               'PWWSAttacker': [], 'TextBuggerAttacker': [], 'VIPERAttacker': []}
    detector_path = f'./data/{id_dataset}/detector/selection_ratio_research/'
    if not os.path.exists(detector_path):
        if not os.path.exists(f'./data/{id_dataset}/detector/'):
            os.mkdir(f'./data/{id_dataset}/detector/')
        os.mkdir(detector_path)

    num_of_category = num_of_labels[id_dataset]
    finetuned_checkpoint = f"./models/roberta-base-finetuned-{id_dataset}/best"
    dataset = load_data(id_dataset)
    train_data, test_data = sentence_tokenizer(dataset, finetuned_checkpoint)

    print('\nGet neural value of train dataset:')
    train_picture_neural_value = get_neural_value(id_dataset=id_dataset, dataset=train_data,
                                                  checkpoint=finetuned_checkpoint, is_anomaly=False)
    print('\nStatistic of train data neural value:')
    train_nv_statistic = statistic_of_neural_value(id_dataset=id_dataset,
                                                   neural_value=train_picture_neural_value)

    adversarial_attacks = roberta_config[id_dataset]['adversarial_settings']
    for attack in adversarial_attacks:
        clean_data, adv_data = load_clean_adv_data(clean_dataset=id_dataset, attack=attack)
        clean_data = sentence_tokenizer(clean_data, finetuned_checkpoint)
        adv_data = sentence_tokenizer(adv_data, finetuned_checkpoint)

        print('\nGet neural value of CLEAN dataset:')
        clean_neural_value = get_neural_value(id_dataset=id_dataset, dataset=clean_data,
                                              checkpoint=finetuned_checkpoint, is_anomaly=False)
        print('\nGet neural value of ADVERSARIAL dataset:')
        adversarial_neural_value = get_neural_value(id_dataset=id_dataset, dataset=adv_data,
                                                    checkpoint=finetuned_checkpoint, is_anomaly=True)
        print(f'Clean Data: {id_dataset}, Adversarial Attacks: {attack}.')

        for r in rate:
            print(f'-------------------------------------{r*0.01}---------------------------------------')
            global_config.selective_rate = r*0.01
            k_means_centers = None
            if os.path.exists(detector_path + f'train_nv_statistic.pkl') and \
                    os.path.exists(detector_path + f'k_means_centers_{r}.pkl'):
                print("\nDetector is existed!")
                file1 = open(detector_path + f'train_nv_statistic.pkl', 'rb')
                file2 = open(detector_path + f'k_means_centers_{r}.pkl', 'rb')
                train_nv_statistic = pickle.load(file1)
                k_means_centers = pickle.load(file2)

            else:
                print('\n********************** Train Detector ****************************')

                print('\nEncoding ReAD of train dataset:')
                train_ReAD_abstractions = encode_abstraction(id_dataset=id_dataset,
                                                             neural_value=train_picture_neural_value,
                                                             train_dataset_statistic=train_nv_statistic)
                train_ReAD_concatenated = \
                    concatenate_data_between_layers(id_dataset=id_dataset, data_dict=train_ReAD_abstractions)

                print('\nK-Means Clustering of ReAD on train data:')
                k_means_centers = k_means(data=train_ReAD_concatenated,
                                          category_number=num_of_labels[id_dataset])
                file2 = open(detector_path + f'k_means_centers_{r}.pkl', 'wb')
                pickle.dump(k_means_centers, file2)
                file2.close()

            # ********************** Evaluate Detector **************************** #
            print('\n********************** Evaluate Adversarial Detection ****************************')
            print('\nEncoding ReAD abstraction of CLEAN dataset:')
            clean_ReAD_abstractions = encode_abstraction(id_dataset=id_dataset, neural_value=clean_neural_value,
                                                         train_dataset_statistic=train_nv_statistic)
            clean_ReAD_concatenated = concatenate_data_between_layers(id_dataset=id_dataset,
                                                                      data_dict=clean_ReAD_abstractions)
            print('\nEncoding ReAD abstraction of ADVERSARIAL dataset:')
            adv_ReAD_abstractions = encode_abstraction(id_dataset=id_dataset, neural_value=adversarial_neural_value,
                                                       train_dataset_statistic=train_nv_statistic)
            adv_ReAD_concatenated = concatenate_data_between_layers(id_dataset=id_dataset,
                                                                    data_dict=adv_ReAD_abstractions)
            print('\nCalculate distance between ReAD and cluster centers ...')
            clean_distance = statistic_distance(id_dataset=id_dataset, abstractions=clean_ReAD_concatenated,
                                                cluster_centers=k_means_centers)
            adv_distance = statistic_distance(id_dataset=id_dataset, abstractions=adv_ReAD_concatenated,
                                              cluster_centers=k_means_centers)

            auc = sk_auc(id_dataset, clean_distance, adv_distance)

            print('\nPerformance of Detector:')
            print('AUROC: {:.6f}'.format(auc))
            print('*************************************\n')

            adversarial_performance[attack].append(auc)

    adv_performance_file = open(f'./data/{id_dataset}/selection_ratio_adversarial.pkl', 'wb')
    pickle.dump(adversarial_performance, adv_performance_file)
    adv_performance_file.close()






