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

class GroundTruthAD:
    def __init__(self, t_max, length):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.y_true = torch.tensor([[1.724699342, 1.280493663, 2.862967753],[1.743231802, 1.293007944, 2.785592653],[
        #                             1.756844593, 1.328899287, 2.817748564],[1.795983261, 1.387416558, 2.884389651],[
        #                             1.918288236, 1.64447189, 2.913796804]]).float()



        self.y_true = torch.tensor([[1.724699342, 1.280493663, 1.828603319], [1.743231802, 1.293007944, 1.851228219], [
            1.756844593, 1.328899287, 1.88338413], [1.795983261, 1.387416558, 1.950025217], [
                                        1.918288236, 1.64447189, 1.97943237]]).float()

        Amax = 2.3473
        Amin = 1.1658
        Nmax = 2.0049
        Nmin = 1.6222
        Tmax = 1.6819
        Tmin = 0.8740

        scale_arr = np.array([[Amax, Amin], [Tmax, Tmin], [Nmax, Nmin]])

        for i in range(3):
            self.y_true[:,i] = (self.y_true[:,i] - scale_arr[i,1])/(scale_arr[i,0] - scale_arr[i,1])

        print("self.y_true: {} \n".format(self.y_true))

        self.y_true_month = torch.tensor([0, 546, 1092, 1456, 1638])
        # print("------------------------------GROUND_TRUTH_AD--------------------------------------------", args.log_path)


class ConfigAD:
    def __init__(self):
        # myprint("--------------------------------------------------call init--------------------------------------------------", args.log_path)
        self.T_all = 165.0
        self.T = 165.0
        self.T_unit = 0.1
        self.T_N = int(self.T / self.T_unit)
        self.N = int(self.T / self.T_unit) #184 #int(self.T / self.T_unit)

        self.Node = 3
        np.random.seed(0)

        self.ub = self.T
        self.lb = 0.0

        self.only_truth_flag = False  # True means only using loss_1 for truth_rate = 1.00, no other loss
        self.truth_rate = 1 #0.0034 # 0.0034 * 300 = 1 point(s) as truth #1.00 #0.25 #0.0005
        self.truth_length = int(self.truth_rate * self.T / self.T_unit)
        # if not self.only_truth_flag:
        #     myprint("self.truth_length: {} of {} all ".format(self.truth_length, self.T_N), args.log_path)

        self.continue_period = 0.2
        self.round_bit = 3
        self.continue_id = None
        self.mapping_overall_flag = False
        self.loss2_partial_flag = False


def block_design_a(network_unit, sig):
    return nn.Sequential((OrderedDict({
      'lin1': nn.Linear(1, network_unit),
      'sig1': sig,
      'lin2': nn.Linear(network_unit, network_unit),
      'sig2': sig,
      'lin3': nn.Linear(network_unit, network_unit),
      'sig3': sig,
      'lin4': nn.Linear(network_unit, 1),
    })))



class SimpleNetworkAD(nn.Module):
    def __init__(self, config):
        # print("--------------------------------------------------call init of SimpleNetwork AD--------------------------------------------------")
        super(SimpleNetworkAD, self).__init__()
        self.setup_seed(0)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.x, self.y0, self.t0 = None, None, None
        self.generate_x()
        # self.optimizer = optim.LBFGS(self.parameters(), lr=0.001, max_iter=5000, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
        # self.initial_start()
        self.model_name = "SimpleNetworkAD"
        self.gt = GroundTruthAD(self.config.T, self.config.T_N)
        self.gt_ytrue = self.gt.y_true.to(self.device)
        self.gt_ytrue_month = self.gt.y_true_month.to(self.device)
        # self.gt_v = torch.tensor(self.gt.v).to(self.device)


        # parameters
        mat = scipy.io.loadmat('./Data/20220923YOUT.mat')
        parameters = torch.tensor(mat['parameters']).float().reshape(18).to(self.device)

        # Amyloid
        self.k_a = parameters[0]
        self.k_ta = parameters[1]
        self.k_mt = parameters[2]
        self.d_a = parameters[3]
        self.theta = parameters[4]
        # Tau
        self.k_t = parameters[5]
        self.k_at = parameters[6]
        self.k_ma = parameters[7]
        self.d_t = parameters[8]
        self.delta = parameters[9]
        # Neurodegeneration
        self.k_r = parameters[17]
        self.k_tn = parameters[10]
        self.k_mtn = parameters[11]
        self.gamma = parameters[12]
        self.k_an = parameters[13]
        self.k_man = parameters[14]
        self.beta = parameters[15]
        self.k_atn = parameters[16]
        self.true_para = torch.tensor([self.k_a, self.k_ta, self.k_mt,
                                        self.k_t, self.k_at, self.k_ma,
                                        self.k_tn, self.k_mtn,self.k_an,
                                        self.k_man, self.k_atn])

        # self.k_man_nn = nn.Parameter(torch.abs(torch.rand(1)))  # k_atn

        self.general_para = nn.Parameter(torch.abs(torch.rand(28)))
        self.sig = nn.Tanh()
        self.network_unit = 20

        # Design A
        self.sequences_A = nn.Sequential(block_design_a(self.network_unit, self.sig))
        self.sequences_T = nn.Sequential(block_design_a(self.network_unit, self.sig))
        self.sequences_N = nn.Sequential(block_design_a(self.network_unit, self.sig))


    def forward(self, inputs):
        # print("--------------------------------------------------call forward--------------------------------------------------")
        # print("input",inputs.size())
        A_input = inputs
        T_input = inputs
        N_input = inputs

        
        A_output = self.sequences_A(A_input)
        T_output = self.sequences_T(T_input)
        N_output = self.sequences_N(N_input)
        
        outputs = torch.cat((A_output, T_output, N_output), 1)
        return outputs
  
    def generate_x(self):
        # print("--------------------------------------------------call generate x--------------------------------------------------")
        x = [[i*self.config.T_unit] for i in range(self.config.T_N)]  # toy
        x = np.asarray(x)
        x = self.encode_t(x)
        print("XXXX", x)
        # print("continue_id = {}: [0, {}] is mapped to [{}, {}]".format(self.config.continue_id, self.config.T, len(x[0]), len(x[-1])))
        self.x = torch.tensor(x).float().to(self.device)
      




    def encode_t(self, num):
        return num / self.config.T_all * 2.0 

    def decode_t(self, num):
        return (num ) / 2.0 * self.config.T_all


    def loss(self):
        torch.autograd.set_detect_anomaly(True)
        # print("--------------------------------------------------call loss --------------------------------------------------")
        self.eval()
        all_loss, all_loss1, all_loss2, all_loss3, all_loss4 = torch.tensor([0.0]).to(self.device),torch.tensor([0.0]).to(self.device),torch.tensor([0.0]).to(self.device),torch.tensor([0.0]).to(self.device),torch.tensor([0.0]).to(self.device)

        y = self.forward(self.x)
        # print("output x" , self.x.size())
        # print("output y" ,y.size())
        A = y[:,0:1]
        T = y[:,1:2]
        N = y[:,2:3]

        A_t = torch.gradient(A.reshape([self.config.T_N]), spacing=(self.decode_t(self.x).reshape([self.config.T_N]),))[0].reshape([self.config.T_N,1])
        T_t = torch.gradient(T.reshape([self.config.T_N]), spacing=(self.decode_t(self.x).reshape([self.config.T_N]),))[0].reshape([self.config.T_N,1])
        N_t = torch.gradient(N.reshape([self.config.T_N]), spacing=(self.decode_t(self.x).reshape([self.config.T_N]),))[0].reshape([self.config.T_N,1])

        # print("--------------------------------------------------call f_a --------------------------------------------------")


        # f_a = A_t - (self.k_a_nn*A*(1 - A) + (self.k_ta_nn*torch.pow(T,self.theta)) / (torch.pow((self.k_mt_nn),self.theta) + torch.pow(T,self.theta))) #- self.config.d_a*torch.matmul(A,self.Laplacian))
        # f_t = T_t - (self.k_t_nn*T*(1 - T) + (self.k_at_nn*torch.pow(A,self.delta)) / (torch.pow((self.k_ma_nn),self.delta) + torch.pow(A,self.delta))) #- self.config.d_t*torch.matmul(T,self.Laplacian))
        # f_n = N_t - ((self.k_tn_nn*torch.pow(T,self.gamma)) / (torch.pow((self.k_mtn_nn),self.gamma) + torch.pow(T,self.gamma))+
        #             (self.k_an_nn*torch.pow(A,self.beta)) / (torch.pow((self.k_man_nn),self.beta) + torch.pow(A,self.beta))+
        #             self.k_atn_nn*A*T)

        # self.true_para = torch.tensor([self.k_a, self.k_ta, self.k_mt,
        #                                 self.k_t, self.k_at, self.k_ma,
        #                                 self.k_tn, self.k_mtn,self.k_an,
        #                                 self.k_man, self.k_atn])
        # print(self.general_para[4])
        # print("1",T, (torch.pow(-0.2483,self.general_para[4]+1)))
        # print("2", (self.general_para[4]+1), (torch.pow(-0.2483,torch.floor(self.general_para[4]+1))))

        # f_a = A_t - (self.general_para[0] - self.general_para[1]*A  #   A metabolism
        #              + (self.general_para[2]*torch.pow(T,self.general_para[4]+1)) / (torch.pow((self.general_para[3]),self.general_para[4]+1) + torch.pow(T,self.general_para[4]+1)) # T on A
        #              + (self.general_para[5]*torch.pow(N,self.general_para[7]+1)) / (torch.pow((self.general_para[6]),self.general_para[7]+1) + torch.pow(N,self.general_para[7]+1)))  # N on A
        #              #- self.general_para[8]*torch.matmul(A,self.Laplacian))
        #
        # f_t = T_t - (self.general_para[9] - self.general_para[10]*T
        #         + (self.general_para[11]*torch.pow(T,self.general_para[13]+1)) / (torch.pow((self.general_para[12]),self.general_para[13]+1) + torch.pow(T,self.general_para[13]+1)) # T on A
        #         + (self.general_para[14]*torch.pow(N,self.general_para[16]+1)) / (torch.pow((self.general_para[15]),self.general_para[16]+1) + torch.pow(N,self.general_para[16]+1)))  # N on A
        #         #- self.general_para[17]*torch.matmul(A,self.Laplacian))
        #
        # f_n = N_t - (((self.general_para[18]*torch.pow(A,self.general_para[20]+1)) / (torch.pow((self.general_para[19]),self.general_para[20]+1) + torch.pow(T,self.general_para[20]+1)) # T on A
        #         + (self.general_para[21]*torch.pow(T,self.general_para[23]+1)) / (torch.pow((self.general_para[22]),self.general_para[23]+1) + torch.pow(T,self.general_para[23]+1)) # T on A
        #         + (self.general_para[24]*(torch.pow(A,self.general_para[26]+1)+ torch.pow(T,self.general_para[27]+1))) / (torch.pow((self.general_para[25]),self.general_para[26]+1) + torch.pow(A,self.general_para[26]+1 + torch.pow(T,self.general_para[27]+1))))  # N on A
        #         )


        # f_a = A_t - (self.general_para[0] - self.general_para[1]*A  #   A metabolism
        #              + (self.general_para[2]*torch.pow(T,torch.floor(self.general_para[4]+1))) / (torch.pow((self.general_para[3]),torch.floor(self.general_para[4]+1)) + torch.pow(T,torch.floor(self.general_para[4]+1))) # T on A
        #              + (self.general_para[5]*torch.pow(N,torch.floor(self.general_para[7]+1))) / (torch.pow((self.general_para[6]),torch.floor(self.general_para[7]+1)) + torch.pow(N,torch.floor(self.general_para[7]+1))))  # N on A
        #              #- self.general_para[8]*torch.matmul(A,self.Laplacian))
        #
        # f_t = T_t - (self.general_para[9] - self.general_para[10]*T
        #         + (self.general_para[11]*torch.pow(T,torch.floor(self.general_para[13]+1))) / (torch.pow((self.general_para[12]),torch.floor(self.general_para[13]+1)) + torch.pow(T,torch.floor(self.general_para[13]+1))) # T on A
        #         + (self.general_para[14]*torch.pow(N,torch.floor(self.general_para[16]+1))) / (torch.pow((self.general_para[15]),torch.floor(self.general_para[16]+1)) + torch.pow(N,torch.floor(self.general_para[16]+1))))  # N on A
        #         #- self.general_para[17]*torch.matmul(A,self.Laplacian))
        #
        # f_n = N_t - (((self.general_para[18]*torch.pow(A,torch.floor(self.general_para[20]+1))) / (torch.pow((self.general_para[19]),torch.floor(self.general_para[20]+1)) + torch.pow(A,torch.floor(self.general_para[20]+1))) # T on A
        #         + (self.general_para[21]*torch.pow(T,torch.floor(self.general_para[23]+1))) / (torch.pow((self.general_para[22]),torch.floor(self.general_para[23]+1)) + torch.pow(T,torch.floor(self.general_para[23]+1))) # T on A
        #         + (self.general_para[24]*(torch.pow(A,torch.floor(self.general_para[26]+1))+ torch.pow(T,torch.floor(self.general_para[27]+1)))) / (torch.pow((self.general_para[25]),torch.floor(self.general_para[26]+1)) + torch.pow(A,torch.floor(self.general_para[26]+1)) + torch.pow(T,torch.floor(self.general_para[27]+1))))  # N on A
        #         )


        f_a = A_t - (self.general_para[0] - self.general_para[1]*A  #   A metabolism
                     + (self.general_para[2]*torch.pow(T,2)) / (torch.pow((self.general_para[3]),2) + torch.pow(T,2)) # T on A
                     + (self.general_para[5]*torch.pow(N,2)) / (torch.pow((self.general_para[6]),2) + torch.pow(N,2)))  # N on A
                     #- self.general_para[8]*torch.matmul(A,self.Laplacian))

        f_t = T_t - (self.general_para[9] - self.general_para[10]*T
                + (self.general_para[11]*torch.pow(T,2)) / (torch.pow((self.general_para[12]),2) + torch.pow(T,2)) # T on A
                + (self.general_para[14]*torch.pow(N,2)) / (torch.pow((self.general_para[15]),2) + torch.pow(N,2)))  # N on A
                #- self.general_para[17]*torch.matmul(A,self.Laplacian))

        f_n = N_t - (((self.general_para[18]*torch.pow(A,2)) / (torch.pow((self.general_para[19]),2) + torch.pow(A,2)) # T on A
                + (self.general_para[21]*torch.pow(T,2)) / (torch.pow((self.general_para[22]),2) + torch.pow(T,2)) # T on A
                + (self.general_para[24]*(torch.pow(A,2)+ torch.pow(T,2))) / (torch.pow((self.general_para[25]),2) + torch.pow(A,2) + torch.pow(T,2))))  # N on A





        # f_a = A_t - (self.k_a*A*(1 - A) + (self.k_ta*torch.pow(T,self.theta)) / (torch.pow((self.k_mt),self.theta) + torch.pow(T,self.theta))) #- self.config.d_a*torch.matmul(A,self.Laplacian))
        # f_t = T_t - (self.k_t*T*(1 - T) + (self.k_at*torch.pow(A,self.delta)) / (torch.pow((self.k_ma),self.delta) + torch.pow(A,self.delta))) #- self.config.d_t*torch.matmul(T,self.Laplacian))
        # f_n = N_t - ((self.k_tn*torch.pow(T,self.gamma)) / (torch.pow((self.k_mtn),self.gamma) + torch.pow(T,self.gamma))+
        #             (self.k_an*torch.pow(A,self.beta)) / (torch.pow((self.k_man),self.beta) + torch.pow(A,self.beta))+
        #             self.k_atn*A*T)

        # print(self.general_para)
        f_y = torch.cat((f_a, f_t, f_n), 1)
        # print(f_y)




        # print("--------------------------------------------------calculate gradient--------------------------------------------------")

        # L2 norm
        self.loss_norm = torch.nn.MSELoss().to(self.device)
        # zeros_1D = torch.tensor([[0.0]] * self.config.N).to(self.device)

        zeros_2D = torch.tensor([[0.0 for i in range(self.config.Node )] for j in range( self.config.N )]).to(self.device)

        # print(y.shape, self.gt_data.shape)
        # print("loss-y",y.cpu().detach().numpy()[1, :],y.cpu().detach().numpy()[100, :],y.cpu().detach().numpy()[1000, :],y.cpu().detach().numpy()[1600, :])
        # print("loss-truey",self.gt_data[num_sub,1, :],self.gt_data[num_sub,100, :],self.gt_data[num_sub,1000, :],self.gt_data[num_sub,1600, :])
        # loss_1 = self.loss_norm(y[:self.config.truth_length, :], self.gt_data[num_sub,:self.config.truth_length,:])
        # print("1",self.gt_ytrue)

        # check_point = (self.gt_ytrue_month[num_sub, :] / 0.1).int()-1
        # check_point[check_point<0] = 0
        # print(check_point)
        # print("2", y)

        y_totrain = torch.index_select(y, 0, self.gt_ytrue_month)
        # print("y_totrain", y_totrain.shape)
        zeros_test = torch.zeros([13, 3]).to(self.device)
        # print("test:", self.loss_norm(y_totrain, zeros_test))

        # print(self.gt_ytrue[num_sub, :] == 0)
        # y_totrain[self.gt_ytrue[num_sub, :] == 0] = 0
        # print(self.gt_ytrue)


        # print("3", y_totrain)
        # print("4", gt_ytrue_sub)
        loss_1 = self.loss_norm(y_totrain, self.gt_ytrue)

        # self.y_true = torch.tensor(mat['ptData_stacked_20220915'][:, 1:, :].reshape(184, 13, 3)).float()


        if self.config.loss2_partial_flag:
            new_period = int(self.config.continue_period * self.config.T_all / self.config.T_unit)
            loss_2 = self.loss_norm(f_y[-new_period:, :], zeros_2D[-new_period:, :])
        else:
            loss_2 = self.loss_norm(f_y, zeros_2D)  # + torch.var(torch.square(f_y))

        loss_3 = self.loss_norm(torch.abs(y), y)*1e5

        loss_4 = self.loss_norm(torch.abs(self.general_para), self.general_para) * 1e10

        loss_5 = self.loss_norm(torch.diff(y, dim=0), abs(torch.diff(y, dim=0)))*1e8


        # loss_4 = (self.loss_norm(torch.abs(self.k1), self.k1) + \
        #     self.loss_norm(torch.abs(self.k2), self.k2) + \
        #     self.loss_norm(torch.abs(self.k3), self.k3) +\
        #     self.loss_norm(torch.abs(self.k4), self.k4) +\
        #     self.loss_norm(torch.abs(self.k5), self.k5) ) *1e5

        loss = loss_1 + loss_2 + loss_3 + loss_4 +  loss_5 # + loss_3)#+ loss_4 + loss_5) / 1e5
        all_loss += loss
        all_loss1 += loss_1
        all_loss2 += loss_2
        all_loss4 += loss_4
        all_loss4 += loss_3
        all_loss4 += loss_5

            
        self.train()
        return all_loss, [all_loss1, all_loss2, all_loss4], []
          
    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.deterministic = True

def get_now_string():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))


def train_ad(model, args, config, now_string):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model_framework(config).to(device)
    model.train()
    model_save_path_last = f"{args.main_path}/train/{args.name}_{args.epoch}_{args.epoch_step}_{args.lr}_{now_string}_last.pt"
    model_save_path_best = f"{args.main_path}/train/{args.name}_{args.epoch}_{args.epoch_step}_{args.lr}_{now_string}_best.pt"
    loss_save_path = f"{args.main_path}/loss/{args.name}_{args.epoch}_{args.epoch_step}_{args.lr}_{now_string}_loss_{args.epoch}.npy"
    board_save_path = f"{args.main_path}/board/{args.name}_{args.epoch}_{args.epoch_step}_{args.lr}_{now_string}_board"
    myprint("using {}".format(str(device)), args.log_path)
    myprint("epoch = {}".format(args.epoch), args.log_path)
    myprint("epoch_step = {}".format(args.epoch_step), args.log_path)
    myprint("model_name = {}".format(model.model_name), args.log_path)
    myprint("now_string = {}".format(now_string), args.log_path)
    myprint("model_save_path_last = {}".format(model_save_path_last), args.log_path)
    myprint("model_save_path_best = {}".format(model_save_path_best), args.log_path)
    myprint("loss_save_path = {}".format(loss_save_path), args.log_path)
    myprint("args = {}".format({item[0]: item[1] for item in args.__dict__.items() if item[0][0] != "_"}),
            args.log_path)
    myprint("config = {}".format({item[0]: item[1] for item in config.__dict__.items() if item[0][0] != "_"}),
            args.log_path)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    initial_lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch / 10000 + 1))  # decade
    # scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size=10000) # cyclic
    epoch_step = args.epoch_step
    start_time = time.time()
    start_time_0 = start_time
    best_loss = 999999
    now_time = 0
    loss_record = []
    param_ls = []
    param_true = []

    myprint(
        "--------------------------------------------------training start--------------------------------------------------",
        args.log_path)

    for epoch in range(1, args.epoch + 1):
        # print("in epoch ", epoch)
        optimizer.zero_grad()
        inputs = model.x
        outputs = model(inputs)
        if config.only_truth_flag:
            # print("--------------------------------------------------loss only ground truth--------------------------------------------------")
            loss, loss_list, _ = model.loss_only_ground_truth()
        else:
            # print("--------------------------------------------------normal loss--------------------------------------------------")
            loss, loss_list, _ = model.loss()

        # loss_1, loss_2, loss_3 = loss_list[0], loss_list[1], loss_list[2]
        loss.backward()
        # print("loss backward")
        # options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
        # _loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(closure)
        optimizer.step()
        scheduler.step()
        loss_record.append(float(loss.item()))

        if epoch % epoch_step == 0 or epoch == args.epoch:
            now_time = time.time()

            loss_print_part = " ".join(["Loss_{0:d}:{1:.6f}".format(i + 1, loss_part.item()) for i, loss_part in enumerate(loss_list)])
            myprint("Epoch [{0:05d}/{1:05d}] Loss:{2:.6f} {3} Lr:{4:.6f} Time:{5:.6f}s ({6:.2f}min in total, {7:.2f}min remains)".format(epoch, args.epoch, loss.item(), loss_print_part, optimizer.param_groups[0]["lr"], now_time - start_time, (now_time - start_time_0) / 60.0, (now_time - start_time_0) / 60.0 / epoch * (args.epoch - epoch)), args.log_path)
            start_time = time.time()

            torch.save(
                {
                    'epoch': args.epoch,
                    'model_state_dict': model.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()
                }, model_save_path_last)
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(
                    {
                        'epoch': args.epoch,
                        'model_state_dict': model.state_dict(),
                        # 'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item()
                    }, model_save_path_best)
        if epoch % args.save_step == 0:
            test_ad(np.asarray(loss_record),model, args, config, now_string, model.general_para, model.true_para, True, model.gt, None)
            myprint("[Loss]", args.log_path)
            myprint("True parameter : {};".format( model.true_para), args.log_path)
            myprint("General parameter estimation: {};".format(model.general_para),args.log_path)

            # draw_loss(np.asarray(loss_record),args, config, now_string, 1.0)
            np.save(loss_save_path, np.asarray(loss_record))
            # np.save(loss_save_path, np.asarray(loss_record))


    myprint("------------------------- FINISHED TRAINING ---------------------------------------------", args.log_path)
    [equA, equT, equN] = test_ad(np.asarray(loss_record),model, args, config, now_string, model.general_para, model.true_para, True, model.gt, None)
    myprint("[Loss]", args.log_path)
    myprint("True parameter : {};".format(model.true_para), args.log_path)
    myprint("General parameter estimation: {};".format(model.general_para), args.log_path)
    # draw_loss(np.asarray(loss_record),args, config, now_string,  1.0)
    np.save(loss_save_path, np.asarray(loss_record))
    myprint("A: prod, degr, TonA, NonA", args.log_path)
    myprint(np.mean(equA, 1), args.log_path)
    myprint("T: prod, degr, AonT, NonT", args.log_path)
    myprint( np.mean(equT, 1), args.log_path)
    myprint("N: AonN, TonN, ATonN", args.log_path)
    myprint( np.mean(equN, 1), args.log_path)

    best_loss = best_loss
    time_cost = (now_time - start_time_0) / 60.0
    loss_record = np.asarray(loss_record)
    np.save(loss_save_path, loss_record)
    res_dic = {
        "start_time": start_time_0,
        "epoch": args.epoch,
        "model_save_path_last": model_save_path_last,
        # "model_save_path_best": model_save_path_best,
        "loss_save_path": loss_save_path,
        "best_loss": best_loss,
        "loss_record": loss_record
    }
    return model, res_dic


def draw_loss(loss_list,args, config, now_string, last_rate=1.0):
    draw_two_dimension(
        y_lists=[loss_list[-int(len(loss_list) * last_rate):]],
        x_list=range(len(loss_list) - int(len(loss_list) * last_rate) + 1, len(loss_list) + 1),
        color_list=["blue"],
        legend_list=["loss"],
        line_style_list=["solid"],
        fig_title="Loss - lastest {}% - epoch {} to {}".format(int(100 * last_rate),
                                                               len(loss_list) - int(len(loss_list) * last_rate) + 1,
                                                               len(loss_list)),
        fig_x_label="epoch",
        fig_y_label="loss",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=f"{args.main_path}/figure/{args.name}_id={args.seed}_{args.overall_start}/loss{get_now_string()}"
    )

def test_ad(loss_list,model, args, config, now_string, param_ls, param_true, show_flag=True, gt=None, loss_2_details=None):
    # print("--------------------------------------------------call test ad--------------------------------------------------")
    model.eval()
    myprint("Testing & drawing...", args.log_path)
    t = model.x
    y = []
    # for i in range(184):
    # print(model(t,i).shape)
    y.append(model(t).cpu().detach().numpy())
    y = np.asarray(y)

    num_timestep = 1650

    y = y.reshape([num_timestep, 3])

    x = np.asarray([item[0] for item in model.decode_t(t).cpu().detach().numpy()]).reshape(num_timestep)

    figure_save_path_folder = f"{args.main_path}/figure/{args.name}_id={args.seed}_{args.overall_start}/"
    myprint("Test: save figure in {}".format(figure_save_path_folder), args.log_path)
    if not os.path.exists(figure_save_path_folder):
        os.makedirs(figure_save_path_folder)

    colorlist = ['r', 'g', 'b']
    labels = ["A", "T", "N"]
    colorlist_scatter = ['ro', 'go', 'bo']

    m = MultiSubplotDraw(row=3, col=3, fig_size=(39, 30), tight_layout_flag=True, show_flag=False, save_flag=True,
                         save_path="{}/{}".format(figure_save_path_folder,
                                                  f"{get_now_string()}_{args.name}_id={args.seed}_{args.epoch}_{args.lr}_{now_string}_general.png"),
                         save_dpi=100)
    for i in range(3):
        ax = m.add_subplot(
            # y_lists=[y[:, i].reshape(num_timestep)],  # y_lists=[y[:,1:3]]
            # y_lists=[y[:, :].reshape(num_timestep,3)],  # y_lists=[y[:,1:3]]
            # x_list=np.transpose(np.asarray([x,x,x])),
            # # x_list= x,color_list=colorlist[i],
            # color_list = colorlist[i],
            # line_width=6, fig_title_size=40,x_label_size=20,y_label_size=20,
            # line_style_list=["solid"],
            # fig_title=labels[i],
            #
            # fig_x_label="time",
            # fig_y_label="concentration",

            y_lists=[y[:, i].reshape(num_timestep)],  # y_lists=[y[:,1:3]]
            x_list= x,color_list=colorlist[i],
            line_width=6, fig_title_size=60, x_label_size=40, y_label_size=40,
            line_style_list=["solid"],
            fig_title=labels[i],
            fig_x_label="time",
            fig_y_label="concentration",
        )
        ax.scatter(x = (model.gt_ytrue_month.cpu().detach().numpy())/10, y = model.gt_ytrue[:,i].cpu().detach().numpy(), color = colorlist[i], marker = 'x', s = 400, linewidths= 6)

        # ax.scatter(x = (model.gt_ytrue_month.cpu().detach().numpy())/10, y = model.gt_ytrue[:,i].cpu().detach().numpy(), color = colorlist[i], marker = 'x', s = 400, linewidths= 6)

    param_ls = model.general_para.cpu().detach().numpy()
    param_true = model.true_para.cpu().detach().numpy()
    param_ls.reshape(28)
    param_true.reshape(11)

    A = y[:, 0:1]
    T = y[:, 1:2]
    N = y[:, 2:3]

    equA_prod = np.array([param_ls[0]]*num_timestep)
    equA_degr = -param_ls[1]* A
    equA_T = (param_ls[2] * np.power(T, 2)) / (np.power((param_ls[3]), 2) + np.power(T, 2))
    equA_N = (param_ls[5] * np.power(N, 2)) / (np.power((param_ls[6]), 2) + np.power(N, 2))

    equT_prod = np.array([param_ls[9]]*num_timestep)
    equT_degr = -param_ls[10] * A
    equT_A = (param_ls[11] * np.power(T, 2)) / (np.power((param_ls[12]), 2) + np.power(T, 2))
    equT_N = (param_ls[14] * np.power(N, 2)) / (np.power((param_ls[15]), 2) + np.power(N, 2))

    equN_A = (param_ls[18] * np.power(A, 2)) / (np.power((param_ls[19]), 2) + np.power(A, 2))
    equN_T = (param_ls[21] * np.power(T, 2)) / (np.power((param_ls[13]), 2) + np.power(T, 2))
    equN_AT = (param_ls[24] * (np.power(A, 2) + np.power(T, 2))) / (np.power((param_ls[25]), 2) + np.power(T, 2)  + np.power(A, 2))


    # equA = np.array([equA_metabolism.reshape(num_timestep), equA_T.reshape(num_timestep), equA_N.reshape(num_timestep)])
    # equT = np.array([equT_metabolism.reshape(num_timestep), equT_A.reshape(num_timestep), equT_N.reshape(num_timestep)])
    # equN = np.array([equN_A.reshape(num_timestep), equN_T.reshape(num_timestep), equN_AT.reshape(num_timestep)])

    equA = np.array([abs(equA_prod.reshape(num_timestep)), abs(equA_degr.reshape(num_timestep)), abs(equA_T.reshape(num_timestep)), abs(equA_N.reshape(num_timestep))])
    equT = np.array([abs(equT_prod.reshape(num_timestep)), abs(equT_degr.reshape(num_timestep)),abs(equT_A.reshape(num_timestep)), abs(equT_N.reshape(num_timestep))])
    equN = np.array([abs(equN_A.reshape(num_timestep)), abs(equN_T.reshape(num_timestep)), abs(equN_AT.reshape(num_timestep))])

    equA = equA / np.sum(equA, axis=0)
    equT = equT / np.sum(equT, axis=0)
    equN = equN / np.sum(equN, axis=0)

    ax1 = m.add_subplot(x_list=x, y_lists=[equT_prod],  color_list=['w'], fig_title="Equation A",line_style_list=["solid"],line_width=0.1, fig_title_size=40, x_label_size=40,y_label_size=40, fig_x_label = "time", fig_y_label = "% influence")

    ax1.stackplot(x, equA, labels=["equA_prod", "equA_degr",  "equA_T", "equA_N"],
                  colors=["palegoldenrod", "lightblue", "cadetblue","steelblue"])  # cornflowerblue palegoldenrod mediumaquamarine
    ax1.legend(loc='lower left', fontsize=20)

    ax2 = m.add_subplot(x_list=x, y_lists=[equT_prod], color_list=['w'], fig_title="Equation T",line_style_list=["solid"], line_width=0.1, fig_title_size=40, x_label_size=40, y_label_size=40, fig_x_label="time", fig_y_label="% influence")

    ax2.stackplot(x, equT, labels=["equT_prod", "equT_degr", "equA_T", "equT_N"],
                  colors=["palegoldenrod", "lightblue", "cadetblue","steelblue"])  # cornflowerblue palegoldenrod mediumaquamarine
    ax2.legend(loc='lower left', fontsize=20)

    ax3 = m.add_subplot(x_list=x, y_lists=[equT_prod],color_list=['w'], fig_title="Equation N",line_style_list=["solid"],line_width=0.1, fig_title_size=40, x_label_size=40,y_label_size=40, fig_x_label = "time", fig_y_label = "% influence")

    ax3.stackplot(x, equN, labels=["equN_A", "equN_T", "equN_AT"],
                  colors=["palegoldenrod", "lightblue", "cadetblue"])  # cornflowerblue palegoldenrod mediumaquamarine
    ax3.legend(loc='lower left', fontsize=20)

    # m.add_subplot(x_list=x, y_lists=[equA_metabolism, equA_T, equA_N],
    #               color_list=['b', 'r', 'g'], fig_title="Equation A",
    #               line_style_list=["solid", "solid", "solid"],
    #               legend_list=["equA_metabolism", "equA_T", "equA_N"], line_width=6, fig_title_size=40, x_label_size=20,
    #               y_label_size=20, fig_x_label="time",
    #               fig_y_label="influence")
    #
    # m.add_subplot(x_list=x, y_lists=[equT_metabolism, equT_A, equT_N],
    #               color_list=['b', 'r', 'g'], fig_title="Equation T",
    #               line_style_list=["solid", "solid", "solid"],
    #               legend_list=["equT_metabolism", "equA_T", "equT_N"], line_width=6, fig_title_size=40, x_label_size=20,
    #               y_label_size=20, fig_x_label="time",
    #               fig_y_label="influence")
    #
    # m.add_subplot(x_list=x, y_lists=[equN_A, equN_T, equN_AT],
    #               color_list=['b', 'r', 'g'], fig_title="Equation N",
    #               line_style_list=["solid", "solid", "solid"],
    #               legend_list=["equN_A", "equN_T", "equN_AT"], line_width=6, fig_title_size=40, x_label_size=20,
    #               y_label_size=20, fig_x_label="time",
    #               fig_y_label="influence")


    for last_rate in [0.25,0.5,1]:
        m.add_subplot(x_list=range(len(loss_list) - int(len(loss_list) * last_rate) + 1, len(loss_list) + 1),
                      y_lists=[loss_list[-int(len(loss_list) * last_rate):]],
                      color_list=["black"],
                      legend_list=["loss"],
                      line_style_list=["solid"],
                      fig_title="Loss - lastest {}% - epoch {} to {}".format(int(100 * last_rate), len(loss_list) - int(len(loss_list) * last_rate) + 1,len(loss_list)),
                      fig_x_label="epoch",
                      fig_y_label="loss",
                      line_width=6, fig_title_size=40, x_label_size=40, y_label_size=40
                      )



    m.draw()

    pred_save_path_folder = f"{args.main_path}/saves/{args.name}_id={args.seed}_{args.overall_start}_general/"
    myprint("Test: save pred in {}".format(pred_save_path_folder), args.log_path)
    if not os.path.exists(pred_save_path_folder):
        os.makedirs(pred_save_path_folder)
    np.save("{}/{}".format(pred_save_path_folder,
                           f"{get_now_string()}_{args.name}_id={args.seed}_{args.epoch}_{args.lr}_{now_string}_general_pred"),
            y)
    np.save("{}/{}".format(pred_save_path_folder,
                           f"{get_now_string()}_{args.name}_id={args.seed}_{args.epoch}_{args.lr}_{now_string}_general_para"),
            model.general_para.cpu().detach().numpy())
    return [equA, equT, equN]


class Args:
    epoch = 20000
    epoch_step = 2000
    lr = 0.003
    main_path = "."
    save_step = 2000


class TestArgs:
    epoch = 1
    epoch_step = 1
    lr = 0.03
    main_path = "."
    save_step = 1


def run_ad_truth(opt):
    myprint(
        "\n--------------------------------------------------\n NEW RUN \n--------------------------------------------------",
        opt.log_path)
    args = opt
    args.main_path = "."
    if not os.path.exists("{}/train".format(args.main_path)):
        os.makedirs("{}/train".format(args.main_path))
    if not os.path.exists("{}/figure".format(args.main_path)):
        os.makedirs("{}/figure".format(args.main_path))
    if not os.path.exists("{}/loss".format(args.main_path)):
        os.makedirs("{}/loss".format(args.main_path))
    now_string = get_now_string()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ConfigAD()
    model = SimpleNetworkAD(config).to(device)
    model = train_ad(model, args, config, now_string)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100, help="epoch")
    parser.add_argument("--log_path", type=str, default="logs/1.txt", help="log path")
    parser.add_argument("--mode", type=str, default="origin", help="continue or origin")
    parser.add_argument("--epoch_step", type=int, default=10, help="epoch_step")
    parser.add_argument("--name", type=str, default="test", help="name")
    parser.add_argument("--python", type=str, default="ModelBYCC.py", help="python file name")
    parser.add_argument("--id", type=str, default="1-5", help="id")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
    parser.add_argument("--main_path", default=".", help="main_path")
    parser.add_argument("--save_step", type=int, default=100, help="save_step")
    parser.add_argument("--seed", type=int, default=100, help="seed")
    parser.add_argument("--sw", type=int, default=0, help="sliding window flag")
    parser.add_argument("--sw_step", type=int, default=50000, help="sliding window step")
    opt = parser.parse_args()
    opt.overall_start = get_now_string()

    opt.log_path = "{}/{}_{}.txt".format(opt.log_path, opt.name, opt.id)
    myprint("log_path: {}".format(opt.log_path), opt.log_path)
    myprint("cuda is available: {}".format(torch.cuda.is_available()), opt.log_path)

    run_ad_truth(opt)



    
    
