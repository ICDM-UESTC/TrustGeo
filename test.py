# -*- coding: utf-8 -*-

"""
    load checkpoint and then test
"""

import torch.nn

from lib.utils import *
import argparse
import numpy as np
import random
from lib.model import *
import copy
import pandas as pd



parser = argparse.ArgumentParser()
# parameters of initializing
parser.add_argument('--seed', type=int, default=2022, help='manual seed')
parser.add_argument('--model_name', type=str, default='TrustGeo')
parser.add_argument('--dataset', type=str, default='New_York', choices=["Shanghai", "New_York", "Los_Angeles"],
                    help='which dataset to use')

# parameters of training
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--lambda1', type=float, default=7e-3)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--harved_epoch', type=int, default=5) 
parser.add_argument('--early_stop_epoch', type=int, default=50)
parser.add_argument('--saved_epoch', type=int, default=5)  
parser.add_argument('--load_epoch', type=int, default=5)

# parameters of model
parser.add_argument('--dim_in', type=int, default=30, choices=[51, 30], help="51 if Shanghai / 30 else")

opt = parser.parse_args()
print("Learning rate: ", opt.lr)
print("Dataset: ", opt.dataset)

if opt.seed:
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
torch.set_printoptions(threshold=float('inf'))

warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

'''load data'''
train_data = np.load("./datasets/{}/Clustering_s1234_lm70_train.npz".format(opt.dataset),
                     allow_pickle=True)
test_data = np.load("./datasets/{}/Clustering_s1234_lm70_test.npz".format(opt.dataset),
                    allow_pickle=True)
train_data, test_data = train_data["data"], test_data["data"]
print("data loaded.")


if __name__ == '__main__':
    train_data, test_data = get_data_generator(opt, train_data, test_data, normal=2)

    losses = [np.inf]

    checkpoint = torch.load(f"asset/model/{opt.dataset}_{opt.load_epoch}.pth")
    print(f"Load model asset/model/{opt.dataset}_{opt.load_epoch}.pth")
    model = eval("TrustGeo")(opt.dim_in)
    model.load_state_dict(checkpoint['model_state_dict'])
    if cuda:
        model.cuda() 

    # test
    total_mse, total_mae, test_num = 0, 0, 0
    dislist = []

    model.eval()
    distance_all = []  
    macs_list = []
    params_list = []

    with torch.no_grad():

        for i in range(len(test_data)):
            lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay, y_max, y_min = test_data[i]["lm_X"], test_data[i]["lm_Y"], \
                                                                           test_data[i][
                                                                               "tg_X"], test_data[i]["tg_Y"], \
                                                                           test_data[i][
                                                                               "lm_delay"], test_data[i]["tg_delay"], \
                                                                           test_data[i]["y_max"], test_data[i]["y_min"]

            y_pred_g, v_g, alpha_g, beta_g, y_pred_a, v_a, alpha_a, beta_a = model(Tensor(lm_X), Tensor(lm_Y), Tensor(tg_X),
                                                                                                                    Tensor(tg_Y), Tensor(lm_delay),Tensor(tg_delay))
            
            # fuse multi views
            y_pred_f, v_f, alpha_f, beta_f = fuse_nig(y_pred_g, v_g, alpha_g, beta_g, y_pred_a, v_a, alpha_a, beta_a)
               
            distance = dis_loss(Tensor(tg_Y), y_pred_f, y_max, y_min)
            for i in range(len(distance.cpu().detach().numpy())):
                dislist.append(distance.cpu().detach().numpy()[i])
                distance_all.append(distance.cpu().detach().numpy()[i])
                
            test_num += len(tg_Y)
            total_mse += (distance * distance).sum()
            total_mae += distance.sum()
            
        total_mse = total_mse / test_num
        total_mae = total_mae / test_num
    
        print("test: mse: {:.3f}  mae: {:.3f}".format(total_mse, total_mae))
        dislist_sorted = sorted(dislist)
        print('test median: {:.3f}'.format(dislist_sorted[int(len(dislist_sorted) / 2)]))


   
