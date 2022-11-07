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
    def __init__(self, num_sub):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mat = scipy.io.loadmat('./Data/20220923YOUT.mat')
        self.data = torch.tensor(mat['pred'][:,1:,:].reshape(184,1630,3)).float()

        mat = scipy.io.loadmat('./Data/data_20220915.mat')
        self.y_true = torch.tensor(mat['ptData_stacked_20220915'][num_sub, :, :].reshape(13, 3)).float()
        self.y_true[self.y_true<0] = 0
        self.y_true_month = torch.tensor(mat['ptMonth_stacked_20220915'][num_sub, :].reshape(13)).float()




        # print("------------------------------GROUND_TRUTH_AD--------------------------------------------", args.log_path)

class ConfigAD:
    def __init__(self):
        # myprint("--------------------------------------------------call init--------------------------------------------------", args.log_path)
        self.T_all = 163.0
        self.T = 163.0 
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
    def __init__(self, config,num_sub):
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
        self.gt = GroundTruthAD(num_sub)
        self.gt_data = self.gt.data.to(self.device)
        self.gt_ytrue = self.gt.y_true.to(self.device)
        self.gt_ytrue_month = self.gt.y_true_month.to(self.device)
        self.num_sub = num_sub
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
                                       self.k_tn, self.k_mtn, self.k_an,
                                       self.k_man, self.k_atn])

        self.Laplacian = torch.tensor(mat['avgNet']).float().to(self.device)[0:10, 0:10]
        self.r = torch.tensor(mat['avgNet']).float().to(self.device) #.reshape([1])

        self.personalized_para = nn.Parameter(torch.abs(torch.rand(11)))

        self.sig = nn.Tanh()
        self.network_unit = 20

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
        all_loss, all_loss1, all_loss2, all_loss3 = torch.tensor([0.0]).to(self.device),torch.tensor([0.0]).to(self.device),torch.tensor([0.0]).to(self.device),torch.tensor([0.0]).to(self.device)

        y = self.forward(self.x)
        # print("output x" , self.x.size())
        # print("output y" ,y.size())
        A = y[:,0:1]
        # print(A)
        T = y[:,1:2]
        N = y[:,2:3]

        A_t = torch.gradient(A.reshape([self.config.T_N]), spacing=(self.decode_t(self.x).reshape([self.config.T_N]),))[0].reshape([self.config.T_N,1])
        T_t = torch.gradient(T.reshape([self.config.T_N]), spacing=(self.decode_t(self.x).reshape([self.config.T_N]),))[0].reshape([self.config.T_N,1])
        N_t = torch.gradient(N.reshape([self.config.T_N]), spacing=(self.decode_t(self.x).reshape([self.config.T_N]),))[0].reshape([self.config.T_N,1])

        # print("--------------------------------------------------call f_a --------------------------------------------------")

        f_a = A_t - (self.personalized_para[0]*A*(1 - A) + (self.personalized_para[1]*torch.pow(T,self.theta)) / (torch.pow((self.personalized_para[2]),self.theta) + torch.pow(T,self.theta))) #- self.config.d_a*torch.matmul(A,self.Laplacian))
        f_t = T_t - (self.personalized_para[3]*T*(1 - T) + (self.personalized_para[4]*torch.pow(A,self.delta)) / (torch.pow((self.personalized_para[5]),self.delta) + torch.pow(A,self.delta))) #- self.config.d_t*torch.matmul(T,self.Laplacian))
        f_n = N_t - ((self.personalized_para[6]*torch.pow(T,self.gamma)) / (torch.pow((self.personalized_para[7]),self.gamma) + torch.pow(T,self.gamma))+
                    (self.personalized_para[8]*torch.pow(A,self.beta)) / (torch.pow((self.personalized_para[9]),self.beta) + torch.pow(A,self.beta))+
                    self.personalized_para[10]*A*T)


        # f_a = A_t - (self.k_a*A*(1 - A) + (self.k_ta*torch.pow(T,self.theta)) / (torch.pow((self.k_mt),self.theta) + torch.pow(T,self.theta))) #- self.config.d_a*torch.matmul(A,self.Laplacian))
        # f_t = T_t - (self.k_t*T*(1 - T) + (self.k_at*torch.pow(A,self.delta)) / (torch.pow((self.k_ma),self.delta) + torch.pow(A,self.delta))) #- self.config.d_t*torch.matmul(T,self.Laplacian))
        # f_n = N_t - ((self.k_tn*torch.pow(T,self.gamma)) / (torch.pow((self.k_mtn),self.gamma) + torch.pow(T,self.gamma))+
        #             (self.k_an*torch.pow(A,self.beta)) / (torch.pow((self.k_man),self.beta) + torch.pow(A,self.beta))+
        #             self.k_atn*A*T)

        f_y = torch.cat((f_a, f_t, f_n), 1)

        # print("--------------------------------------------------calculate gradient--------------------------------------------------")

        # L2 norm
        self.loss_norm = torch.nn.MSELoss().to(self.device)
        # zeros_1D = torch.tensor([[0.0]] * self.config.N).to(self.device)

        zeros_2D = torch.tensor([[0.0 for i in range(self.config.Node )] for j in range( self.config.N )]).to(self.device)

        # print(self.gt_ytrue_month)
        check_point = self.gt_ytrue_month / 0.1 -1
        check_point = check_point.to(torch.long)
        check_point[check_point<0] = 0
        y_totrain = torch.index_select(y, 0, check_point)
        # print("y_totrain", y_totrain.shape)
        zeros_test = torch.zeros([13, 3]).to(self.device)

        # with torch.no_grad():
        #     gt_ytrue_sub = self.gt_ytrue[num_sub, :, :]

        loss_1 = self.loss_norm(y_totrain[self.gt_ytrue != 0], self.gt_ytrue[self.gt_ytrue != 0] )


        if self.config.loss2_partial_flag:
            new_period = int(self.config.continue_period * self.config.T_all / self.config.T_unit)
            loss_2 = self.loss_norm(f_y[-new_period:, :], zeros_2D[-new_period:, :])
        else:
            loss_2 = self.loss_norm(f_y, zeros_2D)  # + torch.var(torch.square(f_y))

        loss_3 = self.loss_norm(torch.abs(y[:self.config.truth_length, :]), y[:self.config.truth_length, :])*1e5
        # print(y.shape)
        # print(torch.diff(y, dim=0).shape)
        # print(torch.diff(y, dim=0)[:,1:10])
        loss_4 = self.loss_norm(torch.diff(y, dim=0), abs(torch.diff(y, dim=0)))*1e8
        # print(loss_4)

        loss = loss_1 + loss_2 + loss_3 +loss_4 # + loss_3)#+ loss_4 + loss_5) / 1e5
        all_loss += loss
        all_loss1 += loss_1
        all_loss2 += loss_2
        all_loss3 += loss_4
            
            
        self.train()
        return all_loss, [all_loss1, all_loss2, all_loss3], []
          
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
        optimizer.step()
        scheduler.step()
        loss_record.append(float(loss.item()))
        # param_ls.append([float(model.k1.item()), float(model.k2.item()), float(model.k3.item()), float(model.k4.item()),
        #                  float(model.k5.item())])
        # param_true.append([model.k_a, model.k_t, model.k_tn, model.k_an, model.k_atn])
        if epoch % epoch_step == 0 or epoch == args.epoch:
            now_time = time.time()
            myprint("NUM_SUB: {};----------------------------".format(model.num_sub), args.log_path)
            loss_print_part = " ".join(["Loss_{0:d}:{1:.6f}".format(i + 1, loss_part.item()) for i, loss_part in enumerate(loss_list)])
            myprint("Epoch [{0:05d}/{1:05d}] Loss:{2:.6f} {3} Lr:{4:.6f} Time:{5:.6f}s ({6:.2f}min in total, {7:.2f}min remains)".format(epoch, args.epoch, loss.item(), loss_print_part, optimizer.param_groups[0]["lr"], now_time - start_time, (now_time - start_time_0) / 60.0, (now_time - start_time_0) / 60.0 / epoch * (args.epoch - epoch)), args.log_path)
            # myprint("True: {};  estimated: {};   relative error: {}".format(model.k_a, model.k_a_nn.item(),abs(((model.k_a_nn.item() - model.k_a)/model.k_a))), args.log_path)
            # myprint("True: {};  estimated: {};   relative error: {}".format(model.k_t, model.k_t_nn.item(), abs(((model.k_t_nn.item() - model.k_t)/model.k_t))), args.log_path)
            # myprint("True: {};  estimated: {};   relative error: {}".format(model.k_tn, model.k_tn_nn.item(),abs(((model.k_tn_nn.item() - model.k_tn)/model.k_tn))), args.log_path)
            # myprint("True: {};  estimated: {};   relative error: {}".format(model.k_an, model.k_an_nn.item(),abs(((model.k_an_nn.item() - model.k_an)/model.k_an))), args.log_path)
            # myprint("True: {};  estimated: {};   relative error: {}".format(model.k_atn, model.k_atn_nn.item(),abs(((model.k_atn_nn.item() - model.k_atn)/model.k_atn))), args.log_path)

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
            test_ad(model, args, config, now_string, True, model.gt, None)
            myprint("[Loss]", args.log_path)
            draw_loss(np.asarray(loss_record), 1.0)
            np.save(loss_save_path, np.asarray(loss_record))
            myprint("NUM_SUB: {}; True parameter : {};".format(model.num_sub, model.true_para), args.log_path)
            myprint("NUM_SUB: {}; Personalized parameter estimation: {};".format(model.num_sub, model.personalized_para), args.log_path)

        if epoch == args.epoch:
            para_pred = model.personalized_para;
            # print(model.personalized_para)
            # np.save(loss_save_path, np.asarray(loss_record))


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
    # print("tensor board path: {}".format(board_save_path))
    # print("%load_ext tensorboard")
    # print("%tensorboard --logdir={}".format(board_save_path.replace(" ", "\ ")))
    # # return [num_parameter, best_loss, time_cost, loss_record]
    return model, res_dic, para_pred


def draw_loss(loss_list, last_rate=1.0):
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
        save_flag=False,
        save_path=None
    )


def test_ad(model, args, config, now_string, show_flag=True, gt=None, loss_2_details=None):
    # print("--------------------------------------------------call test ad--------------------------------------------------")
    model.eval()
    myprint("Testing & drawing...", args.log_path)
    t = model.x
    y = []
    # for i in range(184):
        # print(model(t,i).shape)
    y.append(model(t).cpu().detach().numpy())
    y = np.asarray(y)
    y = y.reshape([1630,3])

    x = [item[0] for item in model.decode_t(t).cpu().detach().numpy()]

    figure_save_path_folder = f"{args.main_path}/figure/{args.name}_id={args.seed}_{args.overall_start}/"
    myprint("Test: save figure in {}".format(figure_save_path_folder), args.log_path)
    if not os.path.exists(figure_save_path_folder):
        os.makedirs(figure_save_path_folder)

    colorlist = ['r', 'g', 'b']
    labels = ["A", "T", "N"]

    m = MultiSubplotDraw(row=2, col=3, fig_size=(39, 20), tight_layout_flag=True, show_flag=False, save_flag=True,
                         save_path="{}/{}".format(figure_save_path_folder,
                                                  f"{get_now_string()}_{args.name}_id={args.seed}_{args.epoch}_{args.lr}_{now_string}_sub={model.num_sub}.png"),
                         save_dpi=100)
    for i in range(3):
        ax =m.add_subplot(
                y_lists=[y[:, i].reshape(1630) for j in range(184)],  # y_lists=[y[:,1:3]]
                x_list=x,
                color_list=colorlist[i] * 184,
                line_style_list=["solid"] * 184,
                line_width=6, fig_title_size=40,x_label_size=30,y_label_size=30, number_label_size=25,
                fig_title= "Subject {}: {}".format(model.num_sub, labels[i]),
            )

        ax.scatter(x = model.gt_ytrue_month.cpu().detach().numpy(), y = model.gt_ytrue[:,i].cpu().detach().numpy(), color = colorlist[i], marker = 'x', s = 400, linewidths= 6)
    param_ls = model.personalized_para.cpu().detach().numpy()
    param_true = model.true_para.cpu().detach().numpy()
    param_ls.reshape(11)
    param_true.reshape(11)
    # param_true = np.asarray(param_true)
    # labels = ["k_a", "k_t", "k_tn", "k_an", "k_atn"]

    A = y[:, 0:1]
    T = y[:, 1:2]
    N = y[:, 2:3]


    A_TonA = (param_ls[1] * np.power(T, 2)) / (np.power((param_ls[2]), 2) + np.power(T,2))
    T_AonT = (param_ls[4] * np.power(A, 2)) / (np.power((param_ls[5]), 2) + np.power(A,2))
    N_AonN = (param_ls[8] * np.power(A, 2)) / (np.power((param_ls[9]), 2) + np.power(A, 2))
    N_TonN = (param_ls[6] * np.power(T, 2)) / (np.power((param_ls[7]), 2) + np.power(T, 2))
    N_ATonN = param_ls[10] * A * T

    A_prod = param_ls[0] * A * (1 - A)
    T_prod = param_ls[3] * T * (1 - T)

    # matlab
    A_TonA_matlab = (param_true[1] * np.power(T, 2)) / (np.power((param_true[2]), 2) + np.power(T, 2))
    T_AonT_matlab = (param_true[4] * np.power(A, 2)) / (np.power((param_true[5]), 2) + np.power(A, 2))
    N_AonN_matlab = (param_true[8] * np.power(A, 2)) / (np.power((param_true[9]), 2) + np.power(A, 2))
    N_TonN_matlab = (param_true[6] * np.power(T, 2)) / (np.power((param_true[7]), 2) + np.power(T, 2))
    N_ATonN_matlab = param_true[10] * A * T

    A_prod_matlab = param_true[0] * A * (1 - A)
    T_prod_matlab = param_true[3] * T * (1 - T)



    m.add_subplot(x_list=x, y_lists=[A_TonA, A_prod,A_TonA_matlab,A_prod_matlab ],
                color_list=['b', 'r', 'b', 'r'], fig_title= "Euqation A", line_style_list=["solid", "solid","dashed", "dashed"],
                legend_list=["A_TonA", "A_prod","A_TonA_matlab", "A_prod_matlab"],
                fig_title_size=40,x_label_size=30,y_label_size=30, number_label_size=25,)

    m.add_subplot(x_list=x, y_lists=[T_AonT, T_prod, T_AonT_matlab, T_prod_matlab],
                  color_list=['b', 'r', 'b', 'r'], fig_title="Euqation T",
                  line_style_list=["solid", "solid", "dashed", "dashed"],
                  legend_list=["T_AonT", "T_prod", "T_AonT_matlab", "T_prod_matlab"],
                  fig_title_size=40,x_label_size=30,y_label_size=30, number_label_size=25,)

    m.add_subplot(x_list=x, y_lists=[N_AonN, N_TonN, N_ATonN, N_AonN_matlab, N_TonN_matlab, N_ATonN_matlab],
                  color_list=['b', 'r', 'g', 'b', 'r', 'g'], fig_title="Euqation N (linear)",
                  line_style_list=["solid", "solid", "solid", "dashed", "dashed", "dashed"],
                  legend_list=["N_AonN", "N_TonN", "N_ATonN", "N_AonN_matlab", "N_TonN_matlab", "N_ATonN_matlab"],
                  fig_title_size=40,x_label_size=30,y_label_size=30, number_label_size=25, fig_x_label="time",
                  fig_y_label="influence")

    m.draw()

    pred_save_path_folder = f"{args.main_path}/saves/{args.name}_id={args.seed}_{args.overall_start}_sub={model.num_sub}/"
    myprint("Test: save pred in {}".format(pred_save_path_folder), args.log_path)
    if not os.path.exists(pred_save_path_folder):
        os.makedirs(pred_save_path_folder)
    np.save("{}/{}".format(pred_save_path_folder,f"{get_now_string()}_{args.name}_id={args.seed}_{args.epoch}_{args.lr}_{now_string}_pred"),y)
    np.save("{}/{}".format(pred_save_path_folder,f"{get_now_string()}_{args.name}_id={args.seed}_{args.epoch}_{args.lr}_{now_string}_para"),param_ls)


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
    personalized_para = [];
    models = []
    normal_pt_idx = np.load("./Data/20221030normal_pt_idx.npy")
    for i in normal_pt_idx:
        config = ConfigAD()
        model = SimpleNetworkAD(config,i).to(device)
        model, tmp, para = train_ad(model, args, config, now_string)
        personalized_para.append(para.cpu().detach().numpy());
        models.append(model)

    print("{}/Personalized_para_pred/{}".format(args.main_path,
                           f"{args.name}_id={args.seed}_{args.epoch}_{args.lr}_{now_string}_pred"));
    np.save("{}/Personalized_para_pred/{}".format(args.main_path,
                           f"{get_now_string()}_{args.name}_id={args.seed}_{args.epoch}_{args.lr}_{now_string}_pred"),personalized_para)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100, help="epoch")
    parser.add_argument("--log_path", type=str, default="logs/1.txt", help="log path")
    parser.add_argument("--mode", type=str, default="origin", help="continue or origin")
    parser.add_argument("--epoch_step", type=int, default=10, help="epoch_step")
    parser.add_argument("--name", type=str, default="test", help="name")
    parser.add_argument("--python", type=str, default="ModelBYCC.py", help="python file name")
    parser.add_argument("--id", type=str, default="whyy", help="id")
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



    
    
