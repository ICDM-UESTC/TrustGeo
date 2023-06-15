# -*- coding: utf-8 -*-
import torch.nn

from lib.utils import *
import argparse, os
import numpy as np
import random
from lib.model import *
import copy
from thop import profile
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
parser.add_argument('--saved_epoch', type=int, default=200)  

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

'''initiate model'''
model = TrustGeo(dim_in=opt.dim_in)
model.apply(init_network_weights)
if cuda:
    model.cuda()

'''initiate criteria and optimizer'''
lr = opt.lr
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(opt.beta1, opt.beta2))
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

if __name__ == '__main__':
    train_data, test_data = get_data_generator(opt, train_data, test_data, normal=2)

    log_path = f"asset/log"
    if not os.path.exists(log_path):
        os.mkdir(log_path)
        
    f = open(f"asset/log/{opt.dataset}.txt", 'a')
    f.write(f"*********{opt.dataset}*********\n")
    f.write("dim_in="+str(opt.dim_in)+", ")
    f.write("early_stop_epoch="+str(opt.early_stop_epoch)+", ")
    f.write("harved_epoch="+str(opt.harved_epoch)+", ")
    f.write("saved_epoch="+str(opt.saved_epoch)+", ")
    f.write("lambda="+str(opt.lambda1)+", ")
    f.write("lr="+str(opt.lr)+", ")
    f.write("model_name="+opt.model_name+", ")
    f.write("seed="+str(opt.seed)+",")
    f.write("\n")
    f.close()

    # train
    losses = [np.inf]
    no_better_epoch = 0
    early_stop_epoch = 0

    for epoch in range(2000):
        print("epoch {}.    ".format(epoch))
        beta = min([(epoch * 1.) / max([100, 1.]), 1.])
        total_loss, total_mae, train_num, total_data_perturb_loss = 0, 0, 0, 0
        model.train()
        for i in range(len(train_data)):
            lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay, y_max, y_min = train_data[i]["lm_X"], \
                                                                       train_data[i]["lm_Y"], \
                                                                       train_data[i]["tg_X"], \
                                                                       train_data[i]["tg_Y"], \
                                                                       train_data[i]["lm_delay"], \
                                                                       train_data[i]["tg_delay"], \
                                                                       train_data[i]["y_max"], \
                                                                       train_data[i]["y_min"]
            optimizer.zero_grad()
            y_pred_g, v_g, alpha_g, beta_g, y_pred_a, v_a, alpha_a, beta_a = model(Tensor(lm_X), Tensor(lm_Y), Tensor(tg_X),
                                                                                    Tensor(tg_Y), Tensor(lm_delay),Tensor(tg_delay))
            
            # fuse multi views
            y_pred_f, v_f, alpha_f, beta_f = fuse_nig(y_pred_g, v_g, alpha_g, beta_g, y_pred_a, v_a, alpha_a, beta_a)
            

            #loss function
            distance = dis_loss(Tensor(tg_Y), y_pred_f, y_max, y_min)
            mse_loss = distance * distance  # mse loss
          
            loss = NIG_loss(y_pred_g, v_g, alpha_g, beta_g, mse_loss, coeffi=opt.lambda1) + \
                   NIG_loss(y_pred_a, v_a, alpha_a, beta_a, mse_loss, coeffi=opt.lambda1) + \
                   NIG_loss(y_pred_f, v_f, alpha_f, beta_f, mse_loss, coeffi=opt.lambda1)


            loss.backward()
            optimizer.step()
            mse_loss = mse_loss.sum()
            
            total_loss += mse_loss
            total_mae += distance.sum()
            train_num += len(tg_Y)

        total_loss = total_loss / train_num
        total_mae = total_mae / train_num

        print("train: loss: {:.4f} mae: {:.4f}".format(total_loss, total_mae))

        # test
        total_mse, total_mae, test_num = 0, 0, 0
        dislist = []

        model.eval()
        distance_all = []
        
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

            print("test: mse: {:.4f}  mae: {:.4f}".format(total_mse, total_mae))
            dislist_sorted = sorted(dislist)
            print('test median:', dislist_sorted[int(len(dislist_sorted) / 2)])
            
            # save checkpoint fo each 200 epoch
            if epoch >0 and epoch % opt.saved_epoch ==0 and epoch<1000:
                savepath = f"asset/model/{opt.dataset}_{epoch}.pth"
                save_cpt(model, optimizer, epoch, savepath)
                print("Save checkpoint!")

                f = open(f"asset/log/{opt.dataset}.txt", 'a')
                f.write(f"\n*********epoch={epoch}*********\n")
                f.write("test: mse: {:.3f}\tmae: {:.3f}".format(total_mse, total_mae))
                f.write("\ttest median: {:.3f}".format(dislist_sorted[int(len(dislist_sorted) / 2)]))
                f.close()

            batch_metric = total_mae.cpu().numpy()
            if batch_metric <= np.min(losses):
                no_better_epoch = 0 
                early_stop_epoch = 0
                print("Better MAE in epoch {}: {:.4f}".format(epoch, batch_metric))
            else:
                no_better_epoch = no_better_epoch + 1
                early_stop_epoch = early_stop_epoch + 1

            losses.append(batch_metric)

        # halve the learning rate
        if no_better_epoch % opt.harved_epoch == 0 and no_better_epoch != 0:
            lr /= 2
            print("learning rate changes to {}!\n".format(lr))
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(opt.beta1, opt.beta2))
            no_better_epoch = 0

        if early_stop_epoch == opt.early_stop_epoch:
            break
