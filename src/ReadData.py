'''
Description: Data Read
Author: CheeReus_11
Date: 2020-08-08 15:13:16
LastEditTime: 2020-08-09 11:29:08
LastEditors: CheeReus_11
'''

import scipy.io as scio
import pandas as pd
import numpy as np


# Read data from `.mat` file
def read_from_mat(filePath):
    data = scio.loadmat(filePath)
    return data


# Read data from `.csv` file
def read_from_csv(filePath):
    data = pd.read_csv(filePath, header=None, low_memory=False)
    array_data = np.array(data)
    return array_data


# Read data from `.txt` file
def read_from_txt(filePath):
    f = open(filePath)
    line = f.readline()
    data_list = []
    while line:
        num = list(map(str, line.split()))
        data_list.append(num)
        line = f.readline()
    f.close()
    array_data = np.array(data_list)
    return array_data


# Save data to `.csv` file
def data_to_csv(data, filePath):
    np.savetxt(filePath, data, delimiter=',')