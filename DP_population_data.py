# import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle
import os

from data_prepare_const import *


x = np.load("/Users/enze/Desktop/Zhang2022/AD_inverse/Data/2022-10-23-22-37-26_SimpleNetworkAD_id=1_40000_0.001_2022-10-23-22-07-34.npy")

print(x.shape)


# def one_time_deal_PET(data_path_list=None):
#     if not data_path_list:
#         data_path_list = ["/Users/enze/Desktop/Zhang2022/AD_inverse/Data/Amyloid_Full.xlsx", "/Users/enze/Desktop/Zhang2022/AD_inverse/Data/271tau.csv", "/Users/enze/Desktop/Zhang2022/AD_inverse/Data/FDG_Full.xlsx"]
#     data_a = pd.read_csv(data_path_list[0])
#     data_t = pd.read_csv(data_path_list[1])
#     data_n = pd.read_csv(data_path_list[2])
#     data_a = data_a[COLUMN_NAMES + TITLE_NAMES]
#     data_t = data_t[COLUMN_NAMES + TITLE_NAMES]
#     data_n = data_n[COLUMN_NAMES + TITLE_NAMES]
#
#     class_number = 5
#
#     for type_name, df in zip(["PET-A", "PET-T", "PET-N"], [data_a, data_t, data_n]):
#         save_path = "data/PET/{}_{{}}.npy".format(type_name)
#         collection = np.zeros((class_number, 160))
#         counts = np.zeros(class_number)
#         for index, row in df.iterrows():
#             label = None
#             for one_key in LABELS:
#                 if row["DX"] in LABELS[one_key]:
#                     label = one_key
#                     counts[LABEL_ID[label]] += 1
#                     break
#
#             if not label:
#                 # print("key not found: \"{}\"".format(row["DX"]))
#                 continue
#             for i in range(160):
#                 collection[LABEL_ID[label]][i] += float(row[COLUMN_NAMES[i]])
#         for one_key in LABELS:
#             if counts[LABEL_ID[one_key]] != 0:
#                 avg = collection[LABEL_ID[one_key], :] / counts[LABEL_ID[one_key]]
#                 # print(type_name, "avg({})".format(collection[LABEL_ID[one_key], :].shape))
#                 np.save(save_path.format(one_key), avg)
#         print(type_name, "counts:", counts)
#
#
# if __name__ == "__main__":
#     # one_time_deal_PET()
#     # d = one_time_build_ptid_dictionary()
#     one_time_deal_PET()
#     pass