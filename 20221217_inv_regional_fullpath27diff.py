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
import pylab
from utils import *
import matplotlib.colors as mcolors


class GroundTruthAD:
    def __init__(self, t_max, length):
        mat = scipy.io.loadmat('./Data/label_nodal_avg_20221106.mat')
        self.y_true = torch.tensor(mat['label_nodal_avg'].reshape(5,480)).float()
        stepsize = int((length)/9)
        self.y_true_month = torch.tensor([0, stepsize*3,  stepsize*6,  stepsize*8,  stepsize*9])


class ConfigAD:
    def __init__(self):
        self.T_all = 163.0
        self.T = 163.0
        self.T_unit = 0.1
        self.T_N = int(self.T / self.T_unit)
        self.N = int(self.T / self.T_unit)  # 184 #int(self.T / self.T_unit)

        self.Node = 480
        np.random.seed(0)

        self.ub = self.T
        self.lb = 0.0

        self.only_truth_flag = False  # True means only using loss_1 for truth_rate = 1.00, no other loss
        self.truth_rate = 1  # 0.0034 # 0.0034 * 300 = 1 point(s) as truth #1.00 #0.25 #0.0005
        self.truth_length = int(self.truth_rate * self.T / self.T_unit)

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
        self.model_name = "SimpleNetworkAD"
        self.gt = GroundTruthAD(self.config.T, self.config.T_N)
        self.gt_ytrue = self.gt.y_true.to(self.device)
        self.gt_ytrue_month = self.gt.y_true_month.to(self.device)

        # parameters
        mat = scipy.io.loadmat('./Data/20220923YOUT.mat')
        self.Laplacian = torch.tensor(mat['avgNet']).float().to(self.device)[0:160, 0:160]

        self.general_para = nn.Parameter(torch.abs(torch.rand(28,160))) #nn.Parameter(torch.abs(torch.rand(13)))#
        # [self.k_a_nn, self.k_ta_nn, self.k_mt_nn, self.d_a,
        #   self.k_t_nn, self.k_at_nn, self.k_ma_nn, self.d_t,
        #   self.k_tn_nn, self.k_mtn_nn, self.k_an_nn,
        #   self.k_man_nn, self.k_atn_nn]

        # [self.k_a_nn, self.k_ta_nn, self.k_mt_nn, self.d_a, self.theta,
        #   self.k_t_nn, self.k_at_nn, self.k_ma_nn, self.d_t, self.delta,
        #   self.k_r, self.k_tn_nn, self.k_mtn_nn, self.gamma, self.k_an_nn,
        #   self.k_man_nn, self.beta, self.k_atn_nn]

        self.sig = nn.Tanh()
        self.network_unit = 20

        A_blocks = [block_design_a(self.network_unit, self.sig) for i in range(self.config.Node // 3)]
        T_blocks = [block_design_a(self.network_unit, self.sig) for i in range(self.config.Node // 3)]
        N_blocks = [block_design_a(self.network_unit, self.sig) for i in range(self.config.Node // 3)]

        self.sequences_A = nn.Sequential(*A_blocks)
        self.sequences_T = nn.Sequential(*T_blocks)
        self.sequences_N = nn.Sequential(*N_blocks)

    def forward(self, inputs):
        # print("--------------------------------------------------call forward--------------------------------------------------")
        # print("input",inputs.size())
        A_input = inputs[:, 0:160]
        T_input = inputs[:, 160:320]
        N_input = inputs[:, 320:480]

        A_output = []
        T_output = []
        N_output = []
        # print(A_input.shape)
        for i in range(self.config.Node // 3):
            A_output.append(self.sequences_A[i](A_input[:, i:i + 1]))
            T_output.append(self.sequences_T[i](T_input[:, i:i + 1]))
            N_output.append(self.sequences_N[i](N_input[:, i:i + 1]))

        A_output = torch.cat(tuple(item for item in A_output), 1)
        T_output = torch.cat(tuple(item for item in T_output), 1)
        N_output = torch.cat(tuple(item for item in N_output), 1)

        outputs = torch.cat((A_output, T_output, N_output), 1)
        return outputs

    def generate_x(self):
        x = [[i * self.config.T_unit for j in range(self.config.Node)] for i in range(self.config.T_N)]  # toy
        x = np.asarray(x)
        x = self.encode_t(x)
        self.x = torch.tensor(x).float().to(self.device)

    def encode_t(self, num):
        return num / self.config.T_all * 2.0

    def decode_t(self, num):
        return (num) / 2.0 * self.config.T_all

    def loss(self):
        torch.autograd.set_detect_anomaly(True)
        # print("--------------------------------------------------call loss --------------------------------------------------")
        self.eval()
        all_loss, all_loss1, all_loss2, all_loss3, all_loss4 = torch.tensor([0.0]).to(self.device), torch.tensor(
            [0.0]).to(self.device), torch.tensor([0.0]).to(self.device), torch.tensor([0.0]).to(
            self.device), torch.tensor([0.0]).to(self.device)

        y = self.forward(self.x)
        A = y[:, 0:160]
        T = y[:, 160:320]
        N = y[:, 320:480]

        A_t_collection, T_t_collection, N_t_collection, C_t_collection = [], [], [], []
        for ii in range(self.config.Node // 3):
            A_t_collection.append(torch.gradient(A[:, ii:ii + 1].reshape([self.config.T_N]), spacing=(self.encode_t(self.x)[:, 0:1].reshape([self.config.T_N]),))[0].reshape([self.config.T_N, 1]))  # u_t = y_t[:, 0:1]
            T_t_collection.append(torch.gradient(T[:, ii:ii + 1].reshape([self.config.T_N]), spacing=(self.encode_t(self.x)[:, 0:1].reshape([self.config.T_N]),))[0].reshape([self.config.T_N, 1]))  # u_t = y_t[:, 0:1]
            N_t_collection.append(torch.gradient(N[:, ii:ii + 1].reshape([self.config.T_N]), spacing=(self.encode_t(self.x)[:, 0:1].reshape([self.config.T_N]),))[0].reshape([self.config.T_N, 1]))  # u_t = y_t[:, 0:1]


        A_t = torch.cat(A_t_collection, 1)
        T_t = torch.cat(T_t_collection, 1)
        N_t = torch.cat(N_t_collection, 1)

        # fixed hill efficient
        f_a = A_t - (self.general_para[0,:] - self.general_para[1,:]*A  #   A metabolism
                     + (self.general_para[2,:]*torch.pow(T,2)) / (torch.pow((self.general_para[3,:]),2) + torch.pow(T,2)) # T on A
                     + (self.general_para[5,:]*torch.pow(N,2)) / (torch.pow((self.general_para[6,:]),2) + torch.pow(N,2)))  # N on A
                    # - self.general_para[8,:]*torch.matmul(A,self.Laplacian))

        f_t = T_t - (self.general_para[9,:] - self.general_para[10,:]*T
                + (self.general_para[11,:]*torch.pow(T,2)) / (torch.pow((self.general_para[12,:]),2) + torch.pow(T,2)) # T on A
                + (self.general_para[14,:]*torch.pow(N,2)) / (torch.pow((self.general_para[15,:]), 2) + torch.pow(N,2))) # N on A
                #- self.general_para[17,:]*torch.matmul(A,self.Laplacian))

        f_n = N_t - (((self.general_para[18,:]*torch.pow(A,2)) / (torch.pow((self.general_para[19,:]),2) + torch.pow(A,2)) # T on A
                + (self.general_para[21,:]*torch.pow(T,2)) / (torch.pow((self.general_para[22,:]),2) + torch.pow(T,2)) # T on A
                + (self.general_para[24,:]*(torch.pow(A,2)+ torch.pow(T,2))) / (torch.pow((self.general_para[25,:]),2) + torch.pow(A,2) + torch.pow(T,2))))  # N on A

        # no diffusion
        # f_a = A_t - (self.general_para[0,:] - self.general_para[1,:]*A  #   A metabolism
        #              + (self.general_para[2,:]*torch.pow(T,self.general_para[4,:])) / (torch.pow((self.general_para[3,:]),self.general_para[4,:]) + torch.pow(T,self.general_para[4,:])) # T on A
        #              + (self.general_para[5,:]*torch.pow(N,self.general_para[7,:])) / (torch.pow((self.general_para[6,:]),self.general_para[7,:]) + torch.pow(N,self.general_para[7,:]))  # N on A
        #              ) #- self.general_para[8,:]*torch.matmul(A,self.Laplacian))
        #
        # f_t = T_t - (self.general_para[9,:] - self.general_para[10,:]*T
        #         + (self.general_para[11,:]*torch.pow(T,self.general_para[13,:])) / (torch.pow((self.general_para[12,:]),self.general_para[13,:]) + torch.pow(T,self.general_para[13,:])) # T on A
        #         + (self.general_para[14,:]*torch.pow(N,self.general_para[16,:])) / (torch.pow((self.general_para[15,:]), self.general_para[16,:]) + torch.pow(N,self.general_para[16,:])) # N on A
        #         ) #- self.general_para[17,:]*torch.matmul(A,self.Laplacian))
        #
        # f_n = N_t - (((self.general_para[18,:]*torch.pow(A,self.general_para[20,:])) / (torch.pow((self.general_para[19,:]),self.general_para[20,:]) + torch.pow(A,self.general_para[20,:])) # T on A
        #         + (self.general_para[21,:]*torch.pow(T,self.general_para[23,:])) / (torch.pow((self.general_para[22,:]),self.general_para[23,:]) + torch.pow(T,self.general_para[23,:])) # T on A
        #         + (self.general_para[24,:]*(torch.pow(A,self.general_para[26,:])+ torch.pow(T,self.general_para[27,:]))) / (torch.pow((self.general_para[25,:]),2) + torch.pow(A,self.general_para[26,:]) + torch.pow(T,self.general_para[27,:]))))  # N on A

        #full
        # f_a = A_t - (self.general_para[0,:] - self.general_para[1,:]*A  #   A metabolism
        #              + (self.general_para[2,:]*torch.pow(T,self.general_para[4,:])) / (torch.pow((self.general_para[3,:]),self.general_para[4,:]) + torch.pow(T,self.general_para[4,:])) # T on A
        #              + (self.general_para[5,:]*torch.pow(N,self.general_para[7,:])) / (torch.pow((self.general_para[6,:]),self.general_para[7,:]) + torch.pow(N,self.general_para[7,:]))  # N on A
        #              - self.general_para[8,:]*torch.matmul(A,self.Laplacian))
        #
        # f_t = T_t - (self.general_para[9,:] - self.general_para[10,:]*T
        #         + (self.general_para[11,:]*torch.pow(T,self.general_para[13,:])) / (torch.pow((self.general_para[12,:]),self.general_para[13,:]) + torch.pow(T,self.general_para[13,:])) # T on A
        #         + (self.general_para[14,:]*torch.pow(N,self.general_para[16,:])) / (torch.pow((self.general_para[15,:]), self.general_para[16,:]) + torch.pow(N,self.general_para[16,:])) # N on A
        #         - self.general_para[17,:]*torch.matmul(A,self.Laplacian))
        #
        # f_n = N_t - (((self.general_para[18,:]*torch.pow(A,self.general_para[20,:])) / (torch.pow((self.general_para[19,:]),self.general_para[20,:]) + torch.pow(A,self.general_para[20,:])) # T on A
        #         + (self.general_para[21,:]*torch.pow(T,self.general_para[23,:])) / (torch.pow((self.general_para[22,:]),self.general_para[23,:]) + torch.pow(T,self.general_para[23,:])) # T on A
        #         + (self.general_para[24,:]*(torch.pow(A,self.general_para[26,:])+ torch.pow(T,self.general_para[27,:]))) / (torch.pow((self.general_para[25,:]),2) + torch.pow(A,self.general_para[26,:]) + torch.pow(T,self.general_para[27,:]))))  # N on A




        f_y = torch.cat((f_a, f_t, f_n), 1)


        # L2 norm
        self.loss_norm = torch.nn.MSELoss().to(self.device)
        # zeros_1D = torch.tensor([[0.0]] * self.config.N).to(self.device)

        zeros_2D = torch.tensor([[0.0 for i in range(self.config.Node)] for j in range(self.config.N)]).to(self.device)
        y_totrain = torch.index_select(y, 0, self.gt_ytrue_month)
        loss_1 = self.loss_norm(y_totrain, self.gt_ytrue)



        if self.config.loss2_partial_flag:
            new_period = int(self.config.continue_period * self.config.T_all / self.config.T_unit)
            loss_2 = self.loss_norm(f_y[-new_period:, :], zeros_2D[-new_period:, :])
        else:
            loss_2 = self.loss_norm(f_y, zeros_2D)  # + torch.var(torch.square(f_y))

        loss_3 = self.loss_norm(torch.abs(y), y) * 1e5

        loss_4 = self.loss_norm(torch.abs(self.general_para), self.general_para) * 1e5

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
    # print("bug :", args.main_path,"/train/",args.name, "}_{", args.epoch, "}_{", args.epoch_step, "}_{", args.lr, "}_{", now_string, "}_{last.pt")
    model_save_path_last = f"{args.main_path}/train/{args.name}_{args.epoch}_{args.epoch_step}_{args.lr}_{now_string}_last.pt"
    model_save_path_best = f"{args.main_path}/train/{args.name}_{args.epoch}_{args.epoch_step}_{args.lr}_{now_string}_best.pt"
    loss_save_path = f"{args.main_path}/loss/{args.name}_{args.epoch}_{args.epoch_step}_{args.lr}_{now_string}_loss_{args.epoch}.npy"
    board_save_path = f"{args.main_path}/board/{args.name}_{args.epoch}_{args.epoch_step}_{args.lr}_{now_string}_board"
    myprint("using {}".format(str(device)), args.log_path)
    myprint("epoch = {}".format(args.epoch), args.log_path)
    myprint("epoch_step = {}".format(args.epoch_step), args.log_path)
    myprint("model_name = {}".format(args.name), args.log_path)
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

            loss_print_part = " ".join(
                ["Loss_{0:d}:{1:.6f}".format(i + 1, loss_part.item()) for i, loss_part in enumerate(loss_list)])
            myprint(
                "Epoch [{0:05d}/{1:05d}] Loss:{2:.6f} {3} Lr:{4:.6f} Time:{5:.6f}s ({6:.2f}min in total, {7:.2f}min remains)".format(
                    epoch, args.epoch, loss.item(), loss_print_part, optimizer.param_groups[0]["lr"],
                    now_time - start_time, (now_time - start_time_0) / 60.0,
                    (now_time - start_time_0) / 60.0 / epoch * (args.epoch - epoch)), args.log_path)
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
            test_ad(np.asarray(loss_record),model, args, config, now_string, model.general_para, True, model.gt, None)
            myprint("[Loss]", args.log_path)
            myprint("avg General parameter estimation: {};".format(torch.mean(model.general_para,1)), args.log_path)

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


def test_ad(loss_list, model, args, config, now_string, param_ls, show_flag=True, gt=None, loss_2_details=None):
    # print("--------------------------------------------------call test ad--------------------------------------------------")
    t_all = 1630
    model.eval()
    myprint("Testing & drawing...", args.log_path)
    t = model.x
    y = []
    # for i in range(184):
    # print(model(t,i).shape)
    y.append(model(t).cpu().detach().numpy())
    y = np.asarray(y)
    y = y.reshape([t_all, 480])

    x = [item[0] for item in model.decode_t(t).cpu().detach().numpy()]

    figure_save_path_folder = f"{args.main_path}/figure/{args.name}_id={args.seed}_{args.overall_start}/"
    myprint("Test: save figure in {}".format(figure_save_path_folder), args.log_path)
    if not os.path.exists(figure_save_path_folder):
        os.makedirs(figure_save_path_folder)

    colorlist = ['r', 'g', 'b']
    labels = ["A", "T", "N"]
    labels_true = ["A truth", "T truth", "N truth"]
    colorlist_scatter = ['ro', 'go', 'bo']

    mat = scipy.io.loadmat('./Data/20220923YOUT.mat')
    Laplacian = mat['avgNet'][0:160, 0:160]

    m = MultiSubplotDraw(row=3, col=3, fig_size=(39, 30), tight_layout_flag=True, show_flag=False, save_flag=True,
                         save_path="{}/{}".format(figure_save_path_folder,
                                                  f"{get_now_string()}_{args.name}_id={args.seed}_{args.epoch}_{args.lr}_{now_string}_general.png"),
                         save_dpi=100)

    for i in range(3):
        print(i,y[:, i:(i+1)*160].shape)
        ax = m.add_subplot(
            y_lists=[y[:, i*160:(i+1)*160].reshape(t_all,160)],  # y_lists=[y[:,1:3]]
            x_list=x,
            color_list=colorlist[i],
            line_style_list=["solid"],
            fig_title=labels[i],
        )

    for i in range(3):
        ax = m.add_subplot(
            y_lists=[[0.5] *1630],  # y_lists=[y[:,1:3]]
            x_list=x,
            color_list=colorlist[i],
            line_style_list=["dashed"],
            line_width= 0.001,
            fig_title=labels_true[i])
        for j in range(160):
            # print(j+i*160)
            # if j+i*160 == 479:
            #     print((model.gt_ytrue_month.cpu().detach().numpy())/10)
            #     print(model.gt_ytrue[:,j+i*160].cpu().detach().numpy())
            ax.plot((model.gt_ytrue_month.cpu().detach().numpy())/10, model.gt_ytrue[:,j+i*160].cpu().detach().numpy(), color = colorlist[i], linestyle='dashed', marker='s')

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

    print("RU&NNNNN")
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





