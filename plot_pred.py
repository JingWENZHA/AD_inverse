import torch
import time
from datetime import datetime
import random
import argparse
import math
import os
import json
import numpy as np
import copy
import sys
import matplotlib.pyplot as plt
from torch import nn, optim
from scipy import stats
from tqdm import tqdm
from collections import OrderedDict
# from torchsummary import summary
from torch.backends import cudnn
from scipy.integrate import odeint
import pickle
import scipy.io


from utils import *


if __name__ == "__main__":
    main_path = "/Users/enze/Desktop/Zhang2022/AD_inverse"
    file_name = "2022-10-17-15-24-53_SimpleNetworkAD_id=100_20000_0.001_2022-10-17-12-12-56.npy"
    ypred = np.load(os.path.join(main_path,"saves",'SimpleNetworkAD_id=100_2022-10-17-12-12-56',"2022-10-17-15-24-53_SimpleNetworkAD_id=100_20000_0.001_2022-10-17-12-12-56.npy"))

    mat = scipy.io.loadmat('./Data/data_20220915.mat')
    y_true = torch.tensor(mat['ptData_stacked_20220915'][:, :, :].reshape(184, 13, 3)).float()
    y_true[y_true < 0] = 0
    y_true_month = torch.tensor(mat['ptMonth_stacked_20220915'][:, :].reshape(184, 13)).float()

    for i in range(10):
        # print(y_true_month[i]/0.1)
        if i == 8:
            print(y_true[i, :, :])
        plt.figure(figsize=(8,6))

        plt.plot(range(1630), ypred[i,:,0],'r')
        plt.plot(y_true_month[i]/0.1, y_true[i, :, 0], 'ro')

        plt.plot(range(1630), ypred[i, :, 1],'g')
        plt.plot(y_true_month[i]/0.1, y_true[i, :, 1], 'gx')

        plt.plot(range(1630), ypred[i, :, 2],'b')
        plt.plot(y_true_month[i]/0.1, y_true[i, :, 2], 'bd')

        plt.legend(["A_pred","A_true","T_pred","A_true","N_pred","N_true"])


        plt.show()
        plt.close()