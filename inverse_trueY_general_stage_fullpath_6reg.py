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

        mat = scipy.io.loadmat('./Data/20221114_label_6reg_avg.mat')
        self.y_true = torch.tensor(mat['label_6reg_avg']).float()

        self.y_true_month = torch.tensor([0, 406, 812, 1218, 1629])
        # print("------------------------------GROUND_TRUTH_AD--------------------------------------------", args.log_path)


class ConfigAD:
    def __init__(self):
        # myprint("--------------------------------------------------call init--------------------------------------------------", args.log_path)
        self.T_all = 500.0
        self.T = 500.0
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

        self.general_para = nn.Parameter(torch.abs(torch.rand(16)))
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
        print("output y" ,y.size())
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

        f_a = A_t - (self.general_para[0]*A*(1 - A) #   A metabolism
                     + (self.general_para[1]*torch.pow(T,self.theta)) / (torch.pow((self.general_para[2]),self.theta) + torch.pow(T,self.theta)) # T on A
                     + (self.general_para[3]*torch.pow(N,self.theta)) / (torch.pow((self.general_para[4]),self.theta) + torch.pow(N,self.theta)))  # N on A
                     #- self.config.d_a*torch.matmul(A,self.Laplacian))

        f_t = T_t - (self.general_para[5]*T*(1 - T)
                     + (self.general_para[6]*torch.pow(A,self.theta)) / (torch.pow((self.general_para[7]),self.delta) + torch.pow(A,self.theta))
                     + (self.general_para[8]*torch.pow(N, self.theta)) / (torch.pow((self.general_para[9]), self.theta) + torch.pow(N, self.theta))) #- self.config.d_t*torch.matmul(T,self.Laplacian))


        f_n = N_t - ((self.general_para[10]*torch.pow(T,self.gamma)) / (torch.pow((self.general_para[11]),self.gamma) + torch.pow(T,self.gamma))
                    + (self.general_para[12]*torch.pow(A,self.beta)) / (torch.pow((self.general_para[13]),self.beta) + torch.pow(A,self.beta)) #+self.general_para[10]*A*T
                    + (self.general_para[14]*torch.pow(A*T,self.beta)) / (torch.pow((self.general_para[15]),self.beta) + torch.pow(A*T,self.beta)))

        # f_n = N_t - ((self.general_para[6] * torch.pow(T, self.gamma)) / (
        #             torch.pow((self.general_para[7]), self.gamma) + torch.pow(T, self.gamma)) +
        #              (self.general_para[8] * torch.pow(A, self.beta)) / (
        #                          torch.pow((self.general_para[9]), self.beta) + torch.pow(A, self.beta))
        #              # +self.general_para[10]*A*T
        #              )



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

        # loss_4 = (self.loss_norm(torch.abs(self.k1), self.k1) + \
        #     self.loss_norm(torch.abs(self.k2), self.k2) + \
        #     self.loss_norm(torch.abs(self.k3), self.k3) +\
        #     self.loss_norm(torch.abs(self.k4), self.k4) +\
        #     self.loss_norm(torch.abs(self.k5), self.k5) ) *1e5

        loss = loss_1 + loss_2 + loss_3 + loss_4  # + loss_3)#+ loss_4 + loss_5) / 1e5
        all_loss += loss
        all_loss1 += loss_1
        all_loss2 += loss_2
        all_loss4 += loss_4
        all_loss4 += loss_3

            
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
            test_ad(model, args, config, now_string, model.general_para, model.true_para, True, model.gt, None)
            myprint("[Loss]", args.log_path)
            myprint("True parameter : {};".format( model.true_para), args.log_path)
            myprint("General parameter estimation: {};".format(model.general_para),args.log_path)

            draw_loss(np.asarray(loss_record), 1.0)
            np.save(loss_save_path, np.asarray(loss_record))
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
    return model, res_dic


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

def test_ad(model, args, config, now_string, param_ls, param_true, show_flag=True, gt=None, loss_2_details=None):
    # print("--------------------------------------------------call test ad--------------------------------------------------")
    model.eval()
    myprint("Testing & drawing...", args.log_path)
    t = model.x
    y = []
    # for i in range(184):
    # print(model(t,i).shape)
    y.append(model(t).cpu().detach().numpy())
    y = np.asarray(y)
    y = y.reshape([5000, 3])

    x = [item[0] for item in model.decode_t(t).cpu().detach().numpy()]

    figure_save_path_folder = f"{args.main_path}/figure/{args.name}_id={args.seed}_{args.overall_start}/"
    myprint("Test: save figure in {}".format(figure_save_path_folder), args.log_path)
    if not os.path.exists(figure_save_path_folder):
        os.makedirs(figure_save_path_folder)

    colorlist = ['r', 'g', 'b']
    labels = ["A", "T", "N"]
    colorlist_scatter = ['ro', 'go', 'bo']

    m = MultiSubplotDraw(row=2, col=3, fig_size=(39, 20), tight_layout_flag=True, show_flag=False, save_flag=True,
                         save_path="{}/{}".format(figure_save_path_folder,
                                                  f"{get_now_string()}_{args.name}_id={args.seed}_{args.epoch}_{args.lr}_{now_string}_general.png"),
                         save_dpi=100)
    for i in range(3):
        ax = m.add_subplot(
            y_lists=[y[:, i].reshape(5000)],  # y_lists=[y[:,1:3]]
            x_list=x,
            color_list=colorlist[i],
            line_width=6, fig_title_size=40,x_label_size=20,y_label_size=20,
            line_style_list=["solid"],
            fig_title=labels[i],
            fig_x_label="time",
            fig_y_label="concentration",
        )

        ax.scatter(x = (model.gt_ytrue_month.cpu().detach().numpy())/10, y = model.gt_ytrue[:,i].cpu().detach().numpy(), color = colorlist[i], marker = 'x', s = 400, linewidths= 6)

    param_ls = model.general_para.cpu().detach().numpy()
    param_true = model.true_para.cpu().detach().numpy()
    param_ls.reshape(16)
    param_true.reshape(11)

    A = y[:, 0:1]
    T = y[:, 1:2]
    N = y[:, 2:3]

    equA_metabolism = param_ls[0] * A * (1 - A)
    equA_T = (param_ls[1] * np.power(T, 2)) / (np.power((param_ls[2]), 2) + np.power(T, 2))
    equA_N = (param_ls[3] * np.power(T, 2)) / (np.power((param_ls[4]), 2) + np.power(T, 2))

    equT_metabolism = param_ls[5] * T * (1 - T)
    equT_A = (param_ls[6] * np.power(A, 2)) / (np.power((param_ls[7]), 2) + np.power(A, 2))
    equT_N = (param_ls[8] * np.power(A, 2)) / (np.power((param_ls[9]), 2) + np.power(A, 2))

    equN_A = (param_ls[10] * np.power(A, 2)) / (np.power((param_ls[11]), 2) + np.power(A, 2))
    equN_T = (param_ls[12] * np.power(T, 2)) / (np.power((param_ls[13]), 2) + np.power(T, 2))
    equN_AT = (param_ls[14] * np.power(T, 2)) / (np.power((param_ls[15]), 2) + np.power(T, 2))

    equA = np.array([equA_metabolism.reshape(5000), equA_T.reshape(5000), equA_N.reshape(5000)])
    equT = np.array([equT_metabolism.reshape(5000), equT_A.reshape(5000), equT_N.reshape(5000)])
    equN = np.array([equN_A.reshape(5000), equN_T.reshape(5000), equN_AT.reshape(5000)])

    equA = equA / np.sum(equA, axis=0)
    equT = equT / np.sum(equT, axis=0)
    equN = equN / np.sum(equN, axis=0)

    ax1 = m.add_subplot(x_list=x, y_lists=[equT_metabolism],  color_list=['w'], fig_title="Equation A",line_style_list=["solid"],line_width=0.1, fig_title_size=40, x_label_size=20,y_label_size=20, fig_x_label = "time", fig_y_label = "% influence")

    ax1.stackplot(x, equA, labels=["equA_metabolism", "equA_T", "equA_N"],
                  colors=["palegoldenrod", "lightblue", "cadetblue"])  # cornflowerblue palegoldenrod mediumaquamarine
    ax1.legend(loc='upper left', fontsize=20)

    ax2 = m.add_subplot(x_list=x, y_lists=[equT_metabolism], color_list=['w'], fig_title="Equation T",line_style_list=["solid"], line_width=0.1, fig_title_size=40, x_label_size=20, y_label_size=20, fig_x_label="time", fig_y_label="% influence")

    ax2.stackplot(x, equT, labels=["equT_metabolism", "equA_T", "equT_N"],
                  colors=["palegoldenrod", "lightblue", "cadetblue"])  # cornflowerblue palegoldenrod mediumaquamarine
    ax2.legend(loc='upper left', fontsize=20)

    ax3 = m.add_subplot(x_list=x, y_lists=[equT_metabolism],color_list=['w'], fig_title="Equation N",line_style_list=["solid"],line_width=0.1, fig_title_size=40, x_label_size=20,y_label_size=20, fig_x_label = "time", fig_y_label = "% influence")

    ax3.stackplot(x, equN, labels=["equN_A", "equN_T", "equN_AT"],
                  colors=["palegoldenrod", "lightblue", "cadetblue"])  # cornflowerblue palegoldenrod mediumaquamarine
    ax3.legend(loc='upper left', fontsize=20)

    # m.add_subplot(x_list=x, y_lists=[equN_A, equN_T, equN_AT],
    #               color_list=['b', 'r', 'g'], fig_title="Equation N",
    #               line_style_list=["solid", "solid", "solid"],
    #               legend_list=["equN_A", "equN_T", "equN_AT"], line_width=6, fig_title_size=40, x_label_size=20,
    #               y_label_size=20, fig_x_label="time",
    #               fig_y_label="influence")


    # m.add_subplot(x_list=x, y_lists=[equA_metabolism, equA_T, equA_N],
    #               color_list=['b', 'r', 'g'], fig_title="Equation A",
    #               line_style_list=["solid", "solid", "solid"],
    #               legend_list=["equA_metabolism", "equA_T", "equA_N"], line_width=6, fig_title_size=40,x_label_size=20,
    # y_label_size=20, fig_x_label = "time",
    # fig_y_label = "influence")
    #
    # m.add_subplot(x_list=x, y_lists=[equT_metabolism, equT_A, equT_N],
    #               color_list=['b', 'r', 'g'], fig_title="Equation T",
    #               line_style_list=["solid", "solid", "solid"],
    #               legend_list=["equT_metabolism", "equA_T", "equT_N"], line_width=6, fig_title_size=40, x_label_size=20,
    # y_label_size=20, fig_x_label = "time",
    # fig_y_label = "influence")


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



    
    
