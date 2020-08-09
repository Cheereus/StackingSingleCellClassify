'''
Description: Data Read
Author: CheeReus_11
Date: 2020-08-08 15:13:16
LastEditTime: 2020-08-09 10:02:57
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