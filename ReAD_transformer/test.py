# from transformers import RobertaConfig
#
# config = RobertaConfig.from_pretrained('./models/pretrained_model/roberta-base')
#
# print(config)

import numpy as np

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
for i in range(0, len(a), 4):
    batch = a[i:i + 4]
    print(batch)
#
# w = np.array([2, 3])
#
# b = np.average(a, weights=w)
#
# print(b)