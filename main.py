from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test
from tqdm import tqdm
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.DLinear import DLinear
from models.lora import apply_lora, freeze_model, LinearLoRA
from torch.optim.lr_scheduler import StepLR

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random

warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='GPT4TS')

parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

# decomp
parser.add_argument('--input_dim', type=int, default=1)
parser.add_argument('--kernel_sizes', type=int, default=5)
parser.add_argument('--steps', type=int, default=3)
parser.add_argument('--ld_pretrained_path', type=str, default='results/decomp.pth')
parser.add_argument('--freeze_decomp', action='store_false', help="Use technical indicators")

parser.add_argument('--root_path', type=str, default='./dataset/traffic/')
parser.add_argument('--data_path', type=str, default='traffic.csv')
parser.add_argument('--lora_path', type=str, default='aapl_ohclv_08.csv')
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--freq', type=int, default=1)
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--percent', type=int, default=10)
parser.add_argument('--technical_indicators', action='store_true', help="Use technical indicators")

parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)

parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--patience', type=int, default=3)

parser.add_argument('--gpt_layers', type=int, default=3)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--enc_in', type=int, default=862)
parser.add_argument('--c_out', type=int, default=862)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--kernel_size', type=int, default=25)
parser.add_argument('--use_lora', action='store_true', help="Use LoRA for finetuning")

parser.add_argument('--loss_func', type=str, default='mse')
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--tmax', type=int, default=10)

parser.add_argument('--itr', type=int, default=1)
parser.add_argument('--cos', type=int, default=0)

def main(args):
    args = parser.parse_args(args)

    SEASONALITY_MAP = {
    "minutely": 1440,
    "10_minutes": 144,
    "half_hourly": 48,
    "hourly": 24,
    "daily": 7,
    "weekly": 1,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1
    }

    mses = []
    maes = []

    for ii in range(args.itr):
        print('ii', ii)
        setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, 336, args.label_len, args.pred_len,
                                                                        args.d_model, args.n_heads, args.e_layers, args.gpt_layers, 
                                                                        args.d_ff, args.embed, ii)
        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        if args.freq == 0:
            args.freq = 'h'

        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')

        # after you build train_loader…
        print("len:", len(train_loader))

        if args.freq != 'h':
            args.freq = SEASONALITY_MAP[test_data.freq]
            print("freq = {}".format(args.freq))

        device = torch.device('cuda:0')

        time_now = time.time()
        train_steps = len(train_loader)

        if args.model == 'PatchTST':
            model = PatchTST(args, device)
            model.to(device)
        elif args.model == 'DLinear':
            model = DLinear(args, device)
            model.to(device)
        else:
            model = GPT4TS(args, device)
        # mse, mae = test(model, test_data, test_loader, args, device, ii)

        params = model.parameters()
        model_optim = torch.optim.Adam(params, lr=args.learning_rate)
        
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        if args.loss_func == 'mse':
            criterion = nn.MSELoss()
        elif args.loss_func == 'smape':
            class SMAPE(nn.Module):
                def __init__(self):
                    super(SMAPE, self).__init__()
                def forward(self, pred, true):
                    return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
            criterion = SMAPE()
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)
        print(">>> Entered train()", flush=True)

        print("Number of training batches:", len(train_loader))

        for epoch in range(args.train_epochs):
            print(f">>> Starting epoch {epoch}", flush=True)

            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                # print('i',i)
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(device)

                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                
                outputs = model(batch_x, ii)

                outputs = outputs[:, -args.pred_len:, :]
                # print('outputs shape train', outputs.shape) # outputs shape train torch.Size([64, 48, 1])
                batch_y = batch_y[:, -args.pred_len:, :].to(device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                model_optim.step()

            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)
            vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
            # test_loss = vali(model, test_data, test_loader, criterion, args, device, ii)
            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}, Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            if args.cos:
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, args)
            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))
        print("------------------------------------")
        mae, mse, rmse, mape, mspe, smape, nd = test(model, test_data, test_loader, args, device, ii)
        mses.append(mse)
        maes.append(mae)

    mses = np.array(mses)
    maes = np.array(maes)
    print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
    print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))
    print(f'mae:{mae}, mse:{mse}, rmse:{rmse}, mape:{mape}, mspe:{mspe}, smape:{smape}, nd:{nd}')

    if args.use_lora:
        lora_data, lora_loader = data_provider(args, 'lora')

        model = freeze_model(model)

        # freeze_model(model)
        # print('lora-injected',model)
        
        model = apply_lora(model, rank=64, alpha=256, target_modules=("Linear",))

        # Unfreeze input embeddings
        if hasattr(model, "wte"):
            for p in model.wte.parameters():
                p.requires_grad = True

        if hasattr(model, "wpe"):
            for p in model.wpe.parameters():
                p.requires_grad = True

        # for name, param in model.named_parameters():
        #     print(name, param.requires_grad)
        model.to(device)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print("LoRA trainable parameters count: ", sum(p.numel() for p in trainable_params))
        # total_params = sum(p.numel() for p in model.parameters())
        # print("Total trainable parameters count: ", total_params)

        lora_optimizer = torch.optim.AdamW(trainable_params, lr=1e-6, weight_decay=5e-6)
        scheduler = StepLR(lora_optimizer, step_size=4, gamma=0.5)
        loss_fn = nn.MSELoss()
        lora_train_loss = []

        for epoch in range(10):
            # print(epoch)
            for i, (batch_x, batch_y) in enumerate(lora_loader):
                model.train()
                lora_optimizer.zero_grad()

                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)

                outputs = model(batch_x, i)

                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :].to(device)

                lora_loss = loss_fn(outputs, batch_y)
                lora_train_loss.append(lora_loss.item())

                lora_loss.backward()
                lora_optimizer.step()

            scheduler.step()

        # testing the LoRA trained model
        mae, mse, rmse, mape, mspe, smape, nd = test(model, test_data, test_loader, args, device, ii)
        print(f'mae:{mae}, mse:{mse}, rmse:{rmse}, mape:{mape}, mspe:{mspe}, smape:{smape}, nd:{nd}')
