'''
Description: Data Read
Author: CheeReus_11
Date: 2020-08-08 15:13:16
LastEditTime: 2020-08-08 18:10:00
LastEditors: CheeReus_11
'''

import scipy.io as scio

# Read data from `.mat` file
def read_from_mat(filePath):
    data = scio.loadmat(filePath)
    return data