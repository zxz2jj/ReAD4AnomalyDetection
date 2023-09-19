import os
import OpenAttack as oa
from adversarial_attacks import AttackEval
from transformers import AutoTokenizer
import pickle

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from load_data import load_data, dataset_mapping
from mode_roberta import RobertaForSequenceClassification


if __name__ == '__main__':

    tasks = ['sst2', 'trec', 'newsgroup']
    attacks = ['SCPNAttacker', 'GANAttacker', 'PWWSAttacker', 'TextBuggerAttacker', 'VIPERAttacker']

    for task in tasks:
        if os.path.exists(f'./data/{task}/adversarial/'):
            os.mkdir(f'./data/{task}/adversarial/')

        """Dataset"""
        dataset = load_data(task)
        test_dataset = dataset['test'].map(function=dataset_mapping)

        """Victim"""
        checkpoint = f'./models/roberta-base-finetuned-{task}/best'
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = RobertaForSequenceClassification.from_pretrained(checkpoint)
        victim = oa.victim.classifiers.TransformersClassifier(model, tokenizer, model.roberta.embeddings.word_embeddings)

        for attack in attacks:
            print(f'-------------------------------------{task} on {attack}----------------------------------')
            """Attacker"""
            # DeepWordBugAttacker, TextFoolerAttacker, PSOAttacker, SCPNAttacker, P
            if attack == 'SCPNAttacker':
                attacker = oa.attackers.SCPNAttacker()
            elif attack == 'GANAttacker':
                attacker = oa.attackers.GANAttacker()
            elif attack == 'TextFoolerAttacker':
                attacker = oa.attackers.TextFoolerAttacker()
            elif attack == 'PWWSAttacker':
                attacker = oa.attackers.PWWSAttacker()
            elif attack == 'TextBuggerAttacker':
                attacker = oa.attackers.TextBuggerAttacker()
            elif attack == 'VIPERAttacker':
                attacker = oa.attackers.VIPERAttacker()
            else:
                attacker = None

            """Attacking..."""
            attack_eval = AttackEval(attacker, victim)
            summary, attack_results = attack_eval.eval(test_dataset, visualize=True)

            file1 = open(f'./data/{task}/adversarial/{task}_{attack}_result.pkl', 'wb')
            pickle.dump(attack_results, file1)
            file2 = open(f'./data/{task}/adversarial/{task}_{attack}_summary.txt', 'w')
            print(summary, file=file2)





