import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau,MultiplicativeLR, ExponentialLR, CosineAnnealingLR #,LinearLR
plt.switch_backend('agg')
import torch.nn as nn
import math

class embedding(nn.Module):
    """
    High frequency embedding block similar to NeRF positional encoding
    Returns input signal with multiple frequency shifted components
    num_freq_bands frequencies are sampled from a linear space of [2^0, 2^max_freq]
    Eg. [1, 16.5, 32] when num_freq_bands=3, max_freq=5
    output_dim = dim(input) * (2*num_freq_bands + 1)
    """
    def __init__(self, num_freq_bands=5, max_freq=5):
        super(embedding, self).__init__()
        self.freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=num_freq_bands)
        # print(self.freq_bands)

    def forward(self, x):
        x_new = torch.tensor([]).to(x.device)
        for freq in self.freq_bands:
            s = torch.sin(math.pi * freq * x)
            c = torch.cos(math.pi * freq * x)
            x_new = torch.cat([x_new, s, c], dim=2)
        x_new = torch.cat([x, x_new], dim=2)
        return x_new


class embedding2(nn.Module):
    """
    Append periodic signals of different frequencies to input signal
    num_freq_bands frequencies are sampled from a linear space of [2^0, 2^max_freq]
    Eg. [1, 16.5, 32] when num_freq_bands=3, max_freq=5
    output_dim = dim(input) * (num_freq_bands + 1)
    """
    def __init__(self, num_freq_bands=5, max_freq=5):
        super(embedding2, self).__init__()
        self.freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=num_freq_bands)
        # print(self.freq_bands)

    def forward(self, x):
        t = torch.linspace(0,1,x.size()[-1]).to(x.device)
        t = t.expand(x.size())
        x_new = torch.tensor([]).to(x.device)
        for freq in self.freq_bands:
            s = torch.sin(math.pi * freq * t)
            x_new = torch.cat([x_new, s], dim=2)
        x_new = torch.cat([x, x_new], dim=2)
        return x_new
    
class embedding3(nn.Module):
    """
    Append periodic signal with multiple phases to input signal
    num_freq_bands frequencies are sampled from a linear space of [2^0, 2^max_freq]
    Eg. [1, 16.5, 32] when num_freq_bands=3, max_freq=5
    output_dim = dim(input) * (num_freq_bands + 1)
    """
    def __init__(self, num_freq_bands=5, max_freq=5):
        super(embedding2, self).__init__()
        self.freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=num_freq_bands)
        # print(self.freq_bands)

    def forward(self, x):
        t = torch.linspace(0,1,x.size()[-1]).to(x.device)
        t = t.expand(x.size())
        x_new = torch.tensor([]).to(x.device)
        for freq in self.freq_bands:
            s = torch.sin(math.pi * freq * t)
            x_new = torch.cat([x_new, s], dim=2)
        x_new = torch.cat([x, x_new], dim=2)
        return x_new

def Dlin_adjust_learning_rate(optimizer, scheduler, epoch, learning_rate, lradj):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if lradj == '1':
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif lradj == '2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif lradj == '3':
        lr_adjust = {epoch: learning_rate if epoch < 10 else learning_rate*0.1}
    elif lradj == '4':
        lr_adjust = {epoch: learning_rate if epoch < 15 else learning_rate*0.1}
    elif lradj == '5':
        lr_adjust = {epoch: learning_rate if epoch < 25 else learning_rate*0.1}
    elif lradj == '6':
        lr_adjust = {epoch: learning_rate if epoch < 5 else learning_rate*0.1}
    elif lradj == 'type3':
        lr_adjust = {epoch: learning_rate if epoch < 3 else learning_rate * (0.8 ** ((epoch - 3) // 1))}
    elif lradj == 'type4':
        lr_adjust = {epoch: learning_rate if epoch < 3 else learning_rate * (0.95 ** ((epoch - 3) // 1))}
    elif lradj =='constant':
        lr_adjust = {epoch: learning_rate}
    elif lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
        
    # elif lradj == 'FixLR':
    #     lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
   
        

class lr_scheduler():
    def __init__(self, optimizer, epochs, args):
        self.lradj=args.lradj
        self.learning_rate=args.learning_rate
        self.epochs=epochs
        self.lmda=lambda epoch:0.95/epoch   # Decays faster than exponential
        self.optimizer=optimizer
        if (self.lradj in ['1','2','3','4','5','6','FixLR','type3','type4']):
            self.mode='Dlin'
        elif (self.lradj in ['MLR', 'LLR','ExLR', 'CosLR', 'PlatLR']): #,'RpLR'
            if(self.lradj=='MLR'):
                self.scheduler=MultiplicativeLR(self.optimizer,lr_lambda=self.lmda) 
            elif(self.lradj=='LLR'):
                print("Can't import LinearLR")
                #self.scheduler=LinearLR(self.optimizer,start_factor=0.1,total_iters=self.epochs)
            elif(self.lradj=='ExLR'):
                self.scheduler=ExponentialLR(self.optimizer, gamma=0.95)
            elif(self.lradj=='CosLR'):
                self.scheduler=CosineAnnealingLR(self.optimizer, gamma=0.95)
            elif(self.lradj=='PlatLR'):
                self.scheduler=ReduceLROnPlateau(self.optimizer,mode='min',factor=0.5,patience=3,verbose=True)
            self.mode='Optim'
    def adjust_learning_rate(self, epoch, train_loss):
        if(self.mode=='Dlin'):
            Dlin_adjust_learning_rate(self.optimizer, self.scheduler, epoch, self.learning_rate, self.lradj)
        elif(self.mode=='Optim' and self.lradj=='PlatLR'):
            self.scheduler.step(train_loss)
            print('Updating learning rate to {}'.format(self.scheduler._last_lr))
        else:
            self.scheduler.step()
            print('Updating learning rate to {}'.format(self.scheduler.get_last_lr()))
        
        



class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# self.encoders = nn.Sequential(
#             _ResidualBlock(
#                 input_dim=encoder_dim,
#                 output_dim=hidden_size,
#                 hidden_size=hidden_size,
#                 use_layer_norm=use_layer_norm,
#                 dropout=dropout,
#             ),
#             *[
#                 _ResidualBlock(
#                     input_dim=hidden_size,
#                     output_dim=hidden_size,
#                     hidden_size=hidden_size,
#                     use_layer_norm=use_layer_norm,
#                     dropout=dropout,
#                 )
#                 for _ in range(num_encoder_layers - 1)
#             ],
#         )