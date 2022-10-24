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
    folder_name = "SimpleNetworkAD_id=100_2022-10-23-21-41-43_sub=1"
    file_name = "2022-10-23-21-47-22_SimpleNetworkAD_id=100_40000_0.001_2022-10-23-21-41-43.npy"
    ypred = np.load(os.path.join(main_path,"saves",folder_name,file_name))

    mat = scipy.io.loadmat('./Data/data_20220915.mat')
    y_true = torch.tensor(mat['ptData_stacked_20220915'][:, :, :].reshape(184, 13, 3)).float()
    y_true[y_true < 0] = 0
    y_true_month = torch.tensor(mat['ptMonth_stacked_20220915'][:, :].reshape(184, 13)).float()
    print(ypred[:, 0])
    for i in range(1):
        # print(y_true_month[i]/0.1)
        plt.figure(figsize=(8,6))

        plt.plot(range(1630), ypred[:, 0], 'r')
        plt.plot(y_true_month[i+1] / 0.1, y_true[i+1, :, 0], 'ro')

        plt.plot(range(1630), ypred[:, 1], 'g')
        plt.plot(y_true_month[i+1] / 0.1, y_true[i+1, :, 1], 'gx')

        plt.plot(range(1630), ypred[:, 2], 'b')
        plt.plot(y_true_month[i+1] / 0.1, y_true[i+1, :, 2], 'bd')

        plt.legend(["A_pred","A_true","T_pred","A_true","N_pred","N_true"])


        # plt.plot(range(1630), ypred[i,:,0],'r')
        # plt.plot(y_true_month[i]/0.1, y_true[i, :, 0], 'ro')
        #
        # plt.plot(range(1630), ypred[i, :, 1],'g')
        # plt.plot(y_true_month[i]/0.1, y_true[i, :, 1], 'gx')
        #
        # plt.plot(range(1630), ypred[i, :, 2],'b')
        # plt.plot(y_true_month[i]/0.1, y_true[i, :, 2], 'bd')



        # m = MultiSubplotDraw(row=3, col=3, fig_size=(39, 30), tight_layout_flag=True, show_flag=True, save_flag=False)
        # for i in range(3):
        #
        # # param_ls = np.asarray(param_ls)
        # # param_true = np.asarray(param_true)
        # # labels = ["k_a", "k_t", "k_tn", "k_an", "k_atn"]
        # # for i in range(len(param_ls[0])):
        # #     m.add_subplot(x_list=[j for j in range(param_ls.shape[0])], y_lists=[param_ls[:, i], param_true[:, i]],
        # #                   color_list=['b', 'black'], fig_title=labels[i], line_style_list=["solid", "dashed"],
        # #                   legend_list=["para_pred", "para_truth"])
        # #
        # # error_ka = abs(((param_ls[:, 0] - param_true[:, 0]) / param_true[:, 0]))
        # # error_kt = abs(((param_ls[:, 1] - param_true[:, 1]) / param_true[:, 1]))
        # # error_ktn = abs(((param_ls[:, 2] - param_true[:, 2]) / param_true[:, 2]))
        # # error_kan = abs(((param_ls[:, 3] - param_true[:, 3]) / param_true[:, 3]))
        # # error_katn = abs(((param_ls[:, 4] - param_true[:, 4]) / param_true[:, 4]))
        # #
        # # m.add_subplot(x_list=[j for j in range(param_ls.shape[0])],
        # #               y_lists=[error_ka, error_kt, error_ktn, error_kan, error_katn], color_list=['b'] * 5,
        # #               fig_title="relatve error", line_style_list=["solid"] * 5, legend_list=labels)
        #
        # m.draw()