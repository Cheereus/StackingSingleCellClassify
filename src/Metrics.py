'''
Description: 
Author: CheeReus_11
Date: 2020-08-10 16:38:03
LastEditTime: 2020-08-10 16:49:56
LastEditors: CheeReus_11
'''

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score

def accuracy(predict_labels, true_labels):
    if len(predict_labels) != len(true_labels):
        print('Label Length Error')
        return 0
    label_length = len(predict_labels)
    correct = 0
    for i in range(label_length):
        if predict_labels[i] == true_labels[i]:
            correct += 1
    return correct / label_length

def ARI(true_labels, predict_labels):
    return adjusted_rand_score(true_labels, predict_labels)

def NMI(true_labels, predict_labels):
    return normalized_mutual_info_score(true_labels, predict_labels)

def F1(true_labels, predict_labels):
    return f1_score(true_labels, predict_labels, average='weighted')