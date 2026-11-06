#TODO: add train mode to checkpoint savefilename
import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
from itertools import product
import optuna
from optuna.trial import TrialState
import logging
import sys, copy

# Define configuration for various experiments
DEEP_NETWORK = True
LRADJ_LIST = ["ExLR"]   # fixed_params dictionary overrides this if 'lradj' key is set
SEQ_LEN_LIST = [200, 336, 450, 600, 720]
MAX_TRAIN_EPOCHS = '100'
TRAIN_MODE = "H"
BATCH_SIZE_DEAFULT = "32"
BATCH_SIZE_TRAFFIC = "32"
BATCH_SIZE_ELEC = "32"

def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_exp_configs():
    model_list = ['DLinear_Filterbank']
    # model_list = ['DLinear_Filterbank_deep']
    # dataset_list = [1,2,3,4,5]
    dataset_list = [8, 9]
    pred_len_list_default = ['96', '192', '336', '720']
    # pred_len_list_default = ['336'] # <- Default
    pred_len_list_ili = ['24', '36', '48', '60']
    # pred_len_list_ili = ['24']
    dataset_dict = {1: ['weather.csv', 21, 'custom'], 2: ['ETTh1.csv', 7, 'ETTh1'],
                        3: ['ETTh2.csv', 7, 'ETTh2'], 4: ['ETTm1.csv', 7, 'ETTm1'], 
                        5: ['ETTm2.csv', 7, 'ETTm2'], 6: ['exchange_rate.csv', 8, 'custom'], 
                        7: ['national_illness.csv', 7, 'custom'], 
                        8: ['traffic.csv', 862, 'custom'],
                        9: ['electricity.csv', 321, 'custom']}

    exp_config_dict = {}
    for model_name in model_list:
        for dataset_id in dataset_list:
            if dataset_id == 7:
                pred_len_list = pred_len_list_ili
                seq_len = 104
            else:
                pred_len_list = pred_len_list_default
                seq_len = 720
            for pred_len in pred_len_list:
                exp_config_dict[model_name+"_"+dataset_dict[dataset_id][0][:-4]+"_"+str(seq_len)+"_"+str(pred_len)] = {'model_name': model_name, 
                                                                                                                    'data_info': dataset_dict[dataset_id], 
                                                                                                                    'seq_len': seq_len, 
                                                                                                                    'pred_len': pred_len, 
                                                                                                                    'indiv': False}
    return exp_config_dict                                                                     


def create_parser():
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
    
    return parser

fixed_params = {    # Should only contain parameters not to be modified during HPO. 
                # 'depth': 2,
                'use_revIN': True, 
                'sampling_fraction': 0, 
                'dropout': 0, 
                'hidden_size': 512, 
                'use_layer_norm': False,
                # 'filter_len': 720,
                # 'period_len': 24,
                'seq_len': 720,
                'noise_variance': 0,
                # 'lr': 0.02,
                'lradj': 'type3',
                }
if DEEP_NETWORK:    # fractional sampling needed to take advantage of resblocks
    if 'sampling_fraction' in fixed_params:
        del fixed_params['sampling_fraction']
    

def create_hpo_params(trial, best_hparams=None):
    params = {}
    params.update(fixed_params)

    def get_param(name, suggest_func, *args):
        if name in fixed_params:
            return fixed_params[name]
        if best_hparams and name in best_hparams:
            return best_hparams[name]
        return suggest_func(name, *args)

    params['lr'] = get_param('lr', trial.suggest_loguniform, 0.001, 0.03)
    params['lradj'] = get_param('lradj', trial.suggest_categorical, LRADJ_LIST)
    if DEEP_NETWORK:    # High depth is costly to compute for deep architectures
        params['depth'] = get_param('depth', trial.suggest_int, 1, 10)  
    else:
        params['depth'] = get_param('depth', trial.suggest_int, 1, 80)
    params['use_revIN'] = get_param('use_revIN', trial.suggest_categorical, [True, False])
    params['sampling_fraction'] = get_param('sampling_fraction', trial.suggest_uniform, 0, 1)
    
    seq_len_for_filter = params.get('seq_len', fixed_params.get('seq_len'))
    if seq_len_for_filter:
        params['filter_len'] = get_param('filter_len', trial.suggest_int, 10, seq_len_for_filter - 1)

    params['period_len'] = get_param('period_len', trial.suggest_categorical, [4, 12, 24, 48])
    params['seq_len'] = get_param('seq_len', trial.suggest_categorical, SEQ_LEN_LIST)
    params['noise_variance'] = get_param('noise_variance', trial.suggest_uniform, 0, 1)

    if DEEP_NETWORK:
        params['dropout'] = get_param('dropout', trial.suggest_uniform, 0, 0.5)
        params['hidden_size'] = get_param('hidden_size', trial.suggest_categorical, [256, 512, 1024])
        params['use_layer_norm'] = get_param('use_layer_norm', trial.suggest_categorical, [True, False])
    else:
        params['dropout'] = fixed_params.get('dropout', 0)
        params['hidden_size'] = fixed_params.get('hidden_size', 512)
        params['use_layer_norm'] = fixed_params.get('use_layer_norm', False)

    return params


def get_setting_string(args):
    return '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_nv{}_dm{}_dd{}_sf{}_fl{}_pl{}_lr{}_sc{}_do{}_hs{}_ln{}_IN{}'.format(
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

def get_experiment_args(exp_config, params):
    parser = create_parser()

    if exp_config['data_info'][0] == 'national_illness.csv':
        seq_len = exp_config['seq_len']
    else:
        seq_len = params.get('seq_len', exp_config['seq_len'])

    if exp_config['data_info'][0] == 'traffic.csv':
        batch_size = BATCH_SIZE_TRAFFIC
    elif exp_config['data_info'][0] == 'electricity.csv':
        batch_size = BATCH_SIZE_ELEC
    else:
        batch_size = BATCH_SIZE_DEAFULT
    
    model_id = f"{exp_config['model_name']}_{exp_config['data_info'][0][:-4]}_{seq_len}_{exp_config['pred_len']}"

    args_list = [
        "--is_training", "1",
        "--root_path", "./dataset/",
        "--data_path", exp_config['data_info'][0],
        "--model_id", model_id,
        "--model", exp_config['model_name'],
        "--individual", str(exp_config['indiv']),
        "--data", exp_config['data_info'][2],
        "--features", "M",
        "--seq_len", str(seq_len),
        "--pred_len", str(exp_config['pred_len']),
        "--noise_variance", str(params.get('noise_variance', 0)),
        "--enc_in", str(exp_config['data_info'][1]),
        "--des", "Exp1",
        "--itr", "1",
        "--batch_size", batch_size,
        "--learning_rate", str(params['lr']),
        "--lradj", str(params['lradj']),
        "--depth", str(params.get('depth', 2)),
        '--train_epochs', MAX_TRAIN_EPOCHS,
        "--train_mode", TRAIN_MODE,
        "--use_revIN", str(params.get('use_revIN', True)),
        "--hidden_size", str(params.get('hidden_size', 512)),
        "--use_layer_norm", str(params.get('use_layer_norm', True)),
        "--dropout", str(params.get('dropout', 0.05)),
        "--sampling_fraction", str(params.get('sampling_fraction', 0.2)),
        "--filter_len", str(params.get('filter_len', 25)),
        "--period_len", str(params.get('period_len', 1)),
    ]
    args = parser.parse_args(args_list)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    return args

def set_random_seed(seed=2021):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def create_objective(exp_config):
    """Create an optuna objective for HPO on the validation set."""
    def objective(trial):
        params = create_hpo_params(trial)
        print("Starting trial with params: ", params)
        set_random_seed()

        args = get_experiment_args(exp_config, params)
        setting = get_setting_string(args)
        
        exp = Exp_Main(args)
        print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.train(setting, trial)

        print(f'>>>>>>>testing on val data: {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        vali_data, vali_loader = exp._get_data(flag='val') # Note: original used 'test' for validation
        mse = exp.vali(vali_data, vali_loader, exp._select_criterion())
        return mse
    return objective

def create_test_objective(exp_config, best_trial_hparams):
    """Create an optuna objective for HPO on the test set, using best validation hyperparameters."""
    def objective(trial):
        params = create_hpo_params(trial, best_hparams=best_trial_hparams)
        print("Starting test trial with params: ", params)
        set_random_seed()
        
        args = get_experiment_args(exp_config, params)
        setting = get_setting_string(args)

        exp = Exp_Main(args) 
        print(f'>>>>>>>start training on test dataset : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
        _, val_loss = exp.train_on_test(setting)
        return val_loss
    return objective


def create_finaleval_objective(exp_config, best_train_trial, best_test_trial):
    """Creates a new objective that uses the best models trained on the train data and test 
    data to compute detailed metrics such as incapacity, non robustness, etc. 
    """
    def final_evaluation():
        set_random_seed()
        
        train_params = dict(best_train_trial.params)
        train_params.update({k: v for k, v in fixed_params.items() if k not in train_params})
        
        if 'period_len' not in train_params:
            train_params['period_len'] = train_params.get('filter_len', 1)

        print("Starting detailed evaluation.")
        print("Best train model configuration: ", train_params)

        # Prepare train experiment
        train_args = get_experiment_args(exp_config, train_params)
        setting_tr = get_setting_string(train_args)
        exp_tr = Exp_Main(train_args) 
        model_tr = exp_tr.model
        
        model_te = None
        setting_te = None
        
        if best_test_trial:
            test_params = dict(train_params)
            test_params.update(best_test_trial.params)
            if 'period_len' not in test_params:
                test_params['period_len'] = test_params.get('filter_len', 1)
            print("Best test model configuration: ", test_params)

            test_args = get_experiment_args(exp_config, test_params)
            setting_te = get_setting_string(test_args)
            exp_te = Exp_Main(test_args) 
            model_te = exp_te.model
        
        print(f'>>>>>>>Evaluating detailed metrics train model saved at {setting_tr} and test model saved at {setting_te or "None"} <<<<<<<<<<<<<<<<<<<<<<<<')
        return exp_tr.evaluate_cap_rob_metrics(model_tr, model_te, setting_tr, setting_te)
        
    return final_evaluation


if __name__ == '__main__':
    COMPUTE_CAP_ROB_METRICS = False
    parser = argparse.ArgumentParser(description='Hyperparameter search')
    parser.add_argument('--exp_idx', default=-1, type=int, help='experiment index from enumerated exp_config_dict to tune hyperparameters for')
    main_args = parser.parse_args()

    exp_config_dict = create_exp_configs()
    if main_args.exp_idx != -1:
        try:
            exp_name = list(exp_config_dict.keys())[main_args.exp_idx]
            exp_config = exp_config_dict[exp_name]
            exp_config_dict = {exp_name: exp_config}
        except IndexError:
            print(f"Error: exp_idx {main_args.exp_idx} is out of range.")
            sys.exit(1)

    study_folder = "optuna_studies/filterbank_sparse_exp2"
    os.makedirs(study_folder, exist_ok=True)

    for exp_name, exp_config in exp_config_dict.items():
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        
        storage_name = f"sqlite:///{study_folder}/{exp_name}.db"
        study = optuna.create_study(study_name=exp_name, storage=storage_name, load_if_exists=True,
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1))
        
        objective = create_objective(exp_config)
        study.optimize(objective, n_trials=4, n_jobs=1)

        best_train_trial = study.best_trial
        best_test_trial = None

        if COMPUTE_CAP_ROB_METRICS:
            test_study_name = f'{exp_name}_test'
            test_storage_name = f"sqlite:///{study_folder}/{test_study_name}.db"
            
            if os.path.exists(test_storage_name[10:]):
                print('Deleting previously existing test study...')
                os.remove(test_storage_name[10:])

            test_study = optuna.create_study(study_name=test_study_name, storage=test_storage_name)
            test_objective = create_test_objective(exp_config, best_train_trial.params)
            test_study.optimize(test_objective, n_trials=10, n_jobs=1)
            best_test_trial = test_study.best_trial
        
        eval_objective = create_finaleval_objective(exp_config, best_train_trial, best_test_trial)
        study_results = eval_objective()
        
        print("Final evaluation results:", study_results)
        study.set_user_attr("model", exp_config['model_name'])
        study.set_user_attr("train_mse", str(study_results[0]))
        study.set_user_attr("test_mse", str(study_results[1]))
        study.set_user_attr("w_diff", str(study_results[2]))
        study.set_user_attr("gen_mse", str(study_results[3]))
        study.set_user_attr("train_params", str(best_train_trial.params))
        study.set_user_attr("test_params", str(best_test_trial.params if best_test_trial else "None"))