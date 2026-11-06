from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import (
DLinear_Filterbank,
DLinear_Filterbank_deep,
DLinear_Filterbank_matrix,
)
from utils.tools import EarlyStopping, visual, test_params_flop, lr_scheduler, Dlin_adjust_learning_rate
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import optuna
# import nni 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'DLinear_Filterbank': DLinear_Filterbank,
            'DLinear_Filterbank_deep': DLinear_Filterbank_deep,
            'DLinear_Filterbank_matrix': DLinear_Filterbank_matrix,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, noise_var=None):
        data_set, data_loader = data_provider(self.args, flag, noise_var=noise_var)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.95)
        # from optimizer.adopt import ADOPT
        # model_optim = ADOPT(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(s in self.args.model for s in ["Linear","TiDBITS"]):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    
                        if any(s in self.args.model for s in ["Linear","TiDBITS"]):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting, trial=None, fine_tune = False):
        train_data, train_loader = self._get_data(flag='train', noise_var=self.args.noise_variance)
        vali_data, vali_loader = self._get_data(flag='val', noise_var=self.args.noise_variance)
        test_data, test_loader = self._get_data(flag='test', noise_var=self.args.noise_variance)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        # scheduler=lr_scheduler(model_optim,self.args.train_epochs,self.args)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                  steps_per_epoch=train_steps,
                                                  pct_start=0.3,
                                                  epochs=self.args.train_epochs,
                                                  max_lr=self.args.learning_rate)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss_sum = 0.0
            train_loss_count = 0
            nan_encountered = False

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                #batch_y = batch_y.float().to(self.device)
                batch_y = batch_y.float().to(self.device)[:,-self.args.pred_len:,:]
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                
                # used in training on L+H (reconstruction and prediction)
                batch_xy = torch.cat([batch_x, batch_y[:,-self.args.pred_len:,:]], dim=1)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(s in self.args.model for s in ["Linear","TiDBITS"]):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                        f_dim = -1 if self.args.features == 'MS' else 0
                        
                        # training on L+H or L+H + fine tuning on H
                        if self.args.train_mode == 'LH' or (self.args.train_mode == 'LHH'and not fine_tune):
                            outputs = outputs[:, :, f_dim:]
                            loss = criterion(outputs, batch_xy)
                        else:
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                        train_loss_sum += loss.item()
                        train_loss_count += 1
                else:
                    if any(s in self.args.model for s in ["Linear","TiDBITS"]):
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0

                    if self.args.train_mode == 'LH' or (self.args.train_mode == 'LHH'and not fine_tune):
                        outputs = outputs[:, :, f_dim:]
                        loss = criterion(outputs, batch_xy)
                    else:
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                    train_loss_sum += loss.item()
                    train_loss_count += 1
                    if torch.isnan(loss):
                        nan_encountered = True
                        break
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()  
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                
                # Clear intermediate variables to free memory
                del outputs, loss
                if 'batch_xy' in locals():
                    del batch_xy

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = train_loss_sum / train_loss_count if train_loss_count > 0 else 0.0
            # Force garbage collection before validation to free GPU memory
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            
            # Report intermediate results to Optuna and prune the trial if necessary
            if trial is not None:
                trial.report(vali_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # nni.report_intermediate_result(test_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            if nan_encountered:
                print("NaN encountered, stopping training")
                break
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            Dlin_adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args.learning_rate, self.args.lradj)
            # Print the adjusted learning rate from model_optim after Dlin_adjust_learning_rate
            current_lr = model_optim.param_groups[0]['lr']
            if current_lr < 1e-6:
                print("Learning rate too low, stopping training")
                break
            
            # scheduler.adjust_learning_rate(epoch + 1, train_loss)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # print('Doing inference [%f%%]\r' % (i * 100 / len(test_loader)), end='')
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(s in self.args.model for s in ["Linear","TiDBITS"]):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(s in self.args.model for s in ["Linear","TiDBITS"]):
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())



                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        #preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        #trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        #inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        # print('mse:{}'.format(mse))
        # f = open("result.csv", 'a')
        # f.write(setting + "  \n")
        # # f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        # f.write('setting:{}, mse:{}, mae:{}, rse:{}'.format(setting, mse, mae, rse))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        return mse

    def train_on_test(self, setting):
        """Function to train a model using test data and evaluate on test data. Used to estimate incapacity."""
        train_data, train_loader = self._get_data(flag='test')
        vali_data, vali_loader = (train_data, train_loader)
        test_data, test_loader = (train_data, train_loader)

        path = os.path.join(self.args.checkpoints, setting+'_test') # save in new 
        print('###############test model save path:', path)
        if not os.path.exists(path):
            os.makedirs(path)

        ###------ Rest of the function is same as train() ------###
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler=lr_scheduler(model_optim,self.args.train_epochs,self.args)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss_sum = 0.0
            train_loss_count = 0

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                #batch_y = batch_y.float().to(self.device)
                batch_y = batch_y.float().to(self.device)[:,-self.args.pred_len:,:]
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # used in training on L+H ( reconstruction and prediction)
                batch_xy = torch.cat([batch_x, batch_y], dim=1)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        # training on L+H
                        if self.args.train_mode == 'LH':
                            outputs = outputs[:, :, f_dim:]
                            loss = criterion(outputs, batch_xy)
                        else:
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                        train_loss_sum += loss.item()
                        train_loss_count += 1
                else:
                    if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    # training on L+H
                    if self.args.train_mode == 'LH':
                        outputs = outputs[:, :, f_dim:]
                        loss = criterion(outputs, batch_xy)
                    else:
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                    train_loss_sum += loss.item()
                    train_loss_count += 1

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                
                # Clear intermediate variables to free memory
                del outputs, loss
                if 'batch_xy' in locals():
                    del batch_xy

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = train_loss_sum / train_loss_count if train_loss_count > 0 else 0.0
            # Force garbage collection before validation to free GPU memory
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # nni.report_intermediate_result(test_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # scheduler.adjust_learning_rate(epoch + 1, train_loss)
            Dlin_adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args.learning_rate, self.args.lradj)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model, early_stopping.val_loss_min
    
    def evaluate_cap_rob_metrics(self, model_tr, model_te, setting_tr, setting_te):
        # Load train and test data
        train_data, train_loader = self._get_data(flag='train')
        # vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        criterion = self._select_criterion()
        
        # Load pretrained models
        path_tr = os.path.join(self.args.checkpoints, setting_tr)
        model_tr.load_state_dict(torch.load(path_tr + '/' + 'checkpoint.pth'))
        if model_te is None:
            self.model = model_tr
            L_tr_te = self.vali(test_data, test_loader, criterion)
            return None, None, None, L_tr_te
        else:
            path_te = os.path.join(self.args.checkpoints, setting_te+'_test') 
            model_te.load_state_dict(torch.load(path_te + '/' + 'checkpoint.pth'))

        # evaluate model_te loss on test data (incapacity)
        self.model = model_te
        L_te_te = self.vali(test_data, test_loader, criterion)

        # evaluate model_tr loss on train data (incapacity)
        self.model = model_tr
        L_tr_tr = self.vali(train_data, train_loader, criterion)
        L_tr_te = self.vali(test_data, test_loader, criterion)

        # evaluate model_tr loss against model_te output on test data L_w_diff (non robustness)
        # evaluate model_tr loss on test data L_tr_te (generalization)
        # the following code is a modification of the self.vali() function
        model_tr.eval()
        model_te.eval()
        #total_loss_w_diff = []
        #total_loss_tr_te = []
        preds_tr_te = []
        preds_te_te = []
        trues = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs_tr = model_tr(batch_x)
                            outputs_te = model_te(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs_tr = model_tr(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                outputs_te = model_te(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs_tr = model_tr(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                outputs_te = model_te(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs_tr = model_tr(batch_x)
                        outputs_te = model_te(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs_tr = model_tr(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            outputs_te = model_te(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs_tr = model_tr(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            outputs_te = model_te(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs_tr = outputs_tr[:, -self.args.pred_len:, f_dim:]
                outputs_te = outputs_te[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred_tr_te = outputs_tr.detach().cpu()
                pred_te_te = outputs_te.detach().cpu()
                true = batch_y.detach().cpu()

                preds_tr_te.append(pred_tr_te)
                preds_te_te.append(pred_te_te)
                trues.append(true)
                #loss_w_diff = criterion(pred_tr_te, pred_te_te)
                #loss_tr_te = criterion(pred_tr_te, true)
                #total_loss_tr_te =
                #total_loss_w_diff.append(loss_w_diff)
                #total_loss_tr_te.append(loss_tr_te)

        preds_tr_te = np.concatenate(preds_tr_te, axis=0)
        preds_te_te = np.concatenate(preds_te_te, axis=0)
        trues = np.concatenate(trues, axis=0)
        #breakpoint()
        #print("tr_te: ",preds_tr_te.shape)
        #print("te_te",preds_te_te.shape)

        preds_tr_te= torch.tensor(preds_tr_te, dtype=torch.float32)
        preds_te_te = torch.tensor(preds_te_te, dtype=torch.float32)
        trues = torch.tensor(trues, dtype=torch.float32)

        loss_w_diff = criterion(preds_tr_te, preds_te_te).item()
        loss_tr_te = criterion(preds_tr_te, trues).item()
        ## This loss should be the same as L_tr_te
        print("Test loss computed using vali() function: ", L_tr_te)
        print("Test loss computed using criterion: ", loss_tr_te)
        
        #return L_tr_tr, L_te_te, L_w_diff, L_tr_te
        return L_tr_tr, L_te_te, loss_w_diff, L_tr_te

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(s in self.args.model for s in ["Linear","TiDBITS"]):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(s in self.args.model for s in ["Linear","TiDBITS"]):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
