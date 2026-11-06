import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='DLinearConv_Electricity', help='model id')
parser.add_argument('--model', type=str, required=True, default='DlinearConv',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='electricity.csv', help='data file')
parser.add_argument('--features', type=str, default='M',help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model `checkpoints`')

# forecasting task
parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--noise_variance', type=float, default=0, help='variance of Gaussian noise to be added to input before prediction')

# DLinear
parser.add_argument('--individual', type=str_to_bool, default=False, help='DLinear: a linear layer for each variate(channel) individually')
parser.add_argument('--depth', type=int, default=2, help='DLinearConv: Depth of the conv tree')
# CoCoNet 
parser.add_argument('--use_revIN', type=str_to_bool, default=True, help='use reversible input normalization')
parser.add_argument('--hidden_size', type=int, default=512, help='hidden size of the residual block')
parser.add_argument('--use_layer_norm', type=str_to_bool, default=True, help='use layer normalization')
parser.add_argument('--sampling_fraction', type=float, default=0.2, help='sampling fraction after decomposition before processing with linear layer')
parser.add_argument('--filter_len', type=int, default=25, help='Filter length of the convolution kernel')
parser.add_argument('--n_blocks', type=int, default=2, help='Number of deep decomposition blocks in the model')
# Sparse weight sharing
parser.add_argument('--period_len', type=int, default=1, help='Period length for downsampling and stacking before processing with linear layer. 1 means no downsampling')

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=321, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=6, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--train_mode', type=str, default='H', help='train on prediction only (H) or prediction and reconstruction (LH)')


# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() else False

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_nv{}_dm{}_dd{}_sf{}_fl{}_pl{}_lr{}_sc{}_do{}_hs{}_ln{}_IN{}'.format(
        args.data_path[:-4],
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.noise_variance,
        args.d_model,
        args.depth,
        args.sampling_fraction,
        args.filter_len,
        args.period_len,
        args.learning_rate,
        args.lradj,
        args.dropout,
        args.hidden_size,
        args.use_layer_norm,
        args.use_revIN)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print(f'>>>>>>>testing on val data: {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        vali_data, vali_loader = exp._get_data(flag='val') # Note: original used 'test' for validation
        val_mse = exp.vali(vali_data, vali_loader, exp._select_criterion())
        print(f'Validation MSE: {val_mse}')
        
        print(f'>>>>>>>testing on test data: {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        test_data, test_loader = exp._get_data(flag='test') # Note: original used 'test' for validation
        test_mse = exp.vali(test_data, test_loader, exp._select_criterion())
        print(f'Test MSE: {test_mse}')


        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                  args.model,
                                                                                                  args.data,
                                                                                                  args.features,
                                                                                                  args.seq_len,
                                                                                                  args.label_len,
                                                                                                  args.pred_len,
                                                                                                  args.d_model,
                                                                                                  args.n_heads,
                                                                                                  args.e_layers,
                                                                                                  args.d_layers,
                                                                                                  args.d_ff,
                                                                                                  args.factor,
                                                                                                  args.embed,
                                                                                                  args.distil,
                                                                                                  args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
