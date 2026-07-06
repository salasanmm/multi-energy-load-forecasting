import argparse
import math
import os
import time
import random
import shutil
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import torch.nn as nn
# from net import gtnet
from test_model.TimeMixer import Model
import numpy as np
import importlib
from Save_result import show_pred
from util import *
from trainer import Optim
import pandas as pd
from metrics import *
# from ptflops import get_model_complexity_info
from net import gtnet
import os
from get_config import get_config
from Save_result_multipredict import show_pred_final

test_model_name = 'TimeMixer'

config = get_config(test_model_name)

pred_length = config.pred_len
seq_len = config.seq_len

save_name = f'{test_model_name}_{pred_length}_{seq_len}'

path1 = f'./output/{save_name}/result/'
if not os.path.exists(path1):
    os.makedirs(path1)
else:
    print('Exiting!')

path2 = f'./output/{save_name}/model/'
if not os.path.exists(path2):
    os.makedirs(path2)
else:
    print('Exiting!')

path3 = f'./output/{save_name}/assets/'
if not os.path.exists(path3):
    os.makedirs(path3)
else:
    print('Exiting!')


def apply_refiner_mode(model, mode):
    prev = {
        "use_pgf": getattr(model, "use_pgf", None),
        "use_crd": getattr(model, "use_crd", None),
        "use_ph": getattr(model, "use_ph", None),
        "use_anchor": getattr(model, "use_anchor", None),
        "use_haf": getattr(model, "use_haf", None),
    }
    if mode is None:
        return prev
    use_pgf = mode in ("pgf", "pgf_crd_ph", "pgf_crd_ph_anchor", "pgf_crd_ph_haf", "pgf_crd_ph_anchor_haf")
    use_crd_ph = mode in ("pgf_crd_ph", "pgf_crd_ph_anchor", "pgf_crd_ph_haf", "pgf_crd_ph_anchor_haf")
    use_anchor = mode in ("pgf_crd_ph_anchor", "pgf_crd_ph_anchor_haf")
    use_haf = mode in ("pgf_crd_ph_haf", "pgf_crd_ph_anchor_haf")
    if prev["use_pgf"] is not None:
        model.use_pgf = use_pgf
    if prev["use_crd"] is not None:
        model.use_crd = use_crd_ph and hasattr(model, "crd")
    if prev["use_ph"] is not None:
        model.use_ph = use_crd_ph and hasattr(model, "ph")
    if prev["use_anchor"] is not None:
        model.use_anchor = use_anchor and hasattr(model, "anchor_fusion")
    if prev["use_haf"] is not None:
        model.use_haf = use_haf and hasattr(model, "haf")
    return prev


def restore_refiner_mode(model, prev):
    for name, value in prev.items():
        if value is not None:
            setattr(model, name, value)


def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, use_pgf=None, mode=None):
    model.eval()
    if mode is None and use_pgf is not None:
        mode = "pgf" if use_pgf else "plain"
    prev_mode = apply_refiner_mode(model, mode)
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        # [64,12,3]
        if len(output.shape) == 1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

    scale = data.scale.expand(predict.size(0), predict.size(1), 3).cpu().numpy()

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    mape = MAPE(predict * scale, Ytest * scale)
    mae = MAE(predict * scale, Ytest * scale)
    rmse = RMSE(predict * scale, Ytest * scale)

    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    restore_refiner_mode(model, prev_mode)
    return mae, mape, correlation, rmse


def channel_metrics(y_true, y_pred):
    names = ["electricity", "cooling", "heating"]
    metrics = {}
    for idx, name in enumerate(names):
        yt = y_true[:, :, idx:idx + 1]
        yp = y_pred[:, :, idx:idx + 1]
        predict = yp
        target = yt
        sigma_p = predict.std(axis=0)
        sigma_g = target.std(axis=0)
        mean_p = predict.mean(axis=0)
        mean_g = target.mean(axis=0)
        valid = (sigma_g != 0) & (sigma_p != 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = ((predict - mean_p) * (target - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        corr = corr[valid].mean() if np.any(valid) else 0.0
        metrics[name] = {
            "mae": MAE(yt, yp),
            "mape": MAPE(yt, yp),
            "rmse": RMSE(yt, yp),
            "corr": corr,
        }
    metrics["total"] = {
        "mae": MAE(y_true, y_pred),
        "mape": MAPE(y_true, y_pred),
        "rmse": RMSE(y_true, y_pred),
    }
    predict = y_pred
    target = y_true
    sigma_p = predict.std(axis=0)
    sigma_g = target.std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = target.mean(axis=0)
    valid = (sigma_g != 0) & (sigma_p != 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = ((predict - mean_p) * (target - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    metrics["total"]["corr"] = corr[valid].mean() if np.any(valid) else 0.0
    return metrics


def format_metrics(prefix, metrics):
    rows = [
        ("OVERALL", metrics["total"]),
        ("Electricity", metrics["electricity"]),
        ("Cold", metrics["cooling"]),
        ("Heat", metrics["heating"]),
    ]
    width = 92
    lines = [
        "=" * width,
        prefix,
        "-" * width,
        f"{'Name':<13} | {'MAE':>10} | {'MAPE':>10} | {'RMSE':>12} | {'ACCR':>8}",
        "-" * width,
    ]
    for name, m in rows:
        lines.append(
            f"{name:<13} | {m['mae']:>10.4f} | {m['mape']:>10.4f} | {m['rmse']:>12.4f} | {m['corr']:>8.4f}"
        )
    lines.append("=" * width)
    return "\n".join(lines)


def weighted_mape_loss(target, pred, weights):
    error = torch.abs(pred - target) / (torch.abs(target) + 1e-2)
    weights = weights.to(target.device).view(1, 1, -1)
    return (error * weights).sum() / (weights.sum() * target.size(0) * target.size(1)) * 100


def weighted_mae_loss(target, pred, weights):
    error = torch.abs(pred - target)
    weights = weights.to(target.device).view(1, 1, -1)
    return (error * weights).sum() / (weights.sum() * target.size(0) * target.size(1))


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    total_mae_loss = 0

    iter = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        # print(X.shape, Y.shape)
        model.zero_grad()
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        # print(X.shape)
        tx = X
        ty = Y
        # print(tx.shape,ty.shape)
        output = model(tx)
        # print(output.shape)
        output = torch.squeeze(output)
        # print(output.shape)
        scale = data.scale.expand(output.size(0), output.size(1), 3)
        # print("ty", ty, "output", output, "scale", scale, "output*scale", output * scale, "ty*scale", ty * scale)
        # print(ty.shape,output.shape,scale.shape)
        scaled_true = ty * scale
        scaled_pred = output * scale
        channel_weights = torch.tensor(
            getattr(config, "channel_loss_weights", [1.0, 1.0, 1.0]),
            dtype=scaled_true.dtype,
            device=scaled_true.device,
        )
        loss_mape = weighted_mape_loss(scaled_true, scaled_pred, channel_weights)
        loss_mae_tensor = weighted_mae_loss(scaled_true, scaled_pred, channel_weights)
        loss = loss_mape + getattr(config, "train_mae_weight", 0.0) * loss_mae_tensor
        loss_mae = MAE(scaled_true.cpu().detach().numpy(), scaled_pred.cpu().detach().numpy())

        loss_mse = MSE(scaled_true, scaled_pred)
        loss.backward()
        total_loss += loss_mape.item()
        total_mae_loss += loss_mae.item()
        grad_norm = optim.step()
        if iter % 100 == 0:
            print('iter:{:3d} | loss: {:.3f}'.format(iter, loss_mape.item() / 3))
        iter += 1

    return total_loss / iter, total_mae_loss / iter


def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:

        _dict = {}
        for _, param in enumerate(model.named_parameters()):
            # print(param[0])
            # print(param[1])
            total_params = param[1].numel()
            # print(f'{total_params:,} total parameters.')
            k = param[0].split('.')[0]
            if k in _dict.keys():
                _dict[k] += total_params
            else:
                _dict[k] = 0
                _dict[k] += total_params
            # print('----------------')
        total_param = sum(p.numel() for p in model.parameters())
        bytes_per_param = 1
        total_bytes = total_param * bytes_per_param
        total_megabytes = total_bytes / (1024 * 1024)
        return total_param, total_megabytes, _dict


def mode_uses(mode):
    return {
        'use_pgf': mode in ("pgf", "pgf_crd_ph", "pgf_crd_ph_anchor", "pgf_crd_ph_haf", "pgf_crd_ph_anchor_haf"),
        'use_crd': mode in ("pgf_crd_ph", "pgf_crd_ph_anchor", "pgf_crd_ph_haf", "pgf_crd_ph_anchor_haf"),
        'use_ph': mode in ("pgf_crd_ph", "pgf_crd_ph_anchor", "pgf_crd_ph_haf", "pgf_crd_ph_anchor_haf"),
        'use_anchor': mode in ("pgf_crd_ph_anchor", "pgf_crd_ph_anchor_haf"),
        'use_haf': mode in ("pgf_crd_ph_haf", "pgf_crd_ph_anchor_haf"),
    }


def make_checkpoint(model, optim, epoch, val_mae, test_mae, mode):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optim.optimizer.state_dict() if optim is not None else None,
        'epoch': epoch,
        'val_mae': val_mae,
        'test_mae': test_mae,
        'mode': mode,
    }
    checkpoint.update(mode_uses(mode))
    return checkpoint


def load_checkpoint_model(model, checkpoint, strict=False):
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint.state_dict()
    current = model.state_dict()
    compatible = {
        key: value for key, value in state_dict.items()
        if key in current and current[key].shape == value.shape
    }
    current.update(compatible)
    model.load_state_dict(current, strict=strict)
    return len(compatible), len(state_dict) - len(compatible)


def default_refiner_mode_from_config():
    if getattr(config, "use_anchor", False) and getattr(config, "use_haf", False):
        return "pgf_crd_ph_anchor_haf"
    if getattr(config, "use_anchor", False):
        return "pgf_crd_ph_anchor"
    if getattr(config, "use_haf", False):
        return "pgf_crd_ph_haf"
    if getattr(config, "use_crd", False) and getattr(config, "use_ph", False):
        return "pgf_crd_ph"
    if getattr(config, "use_pgf", False):
        return "pgf"
    return "plain"


def checkpoint_mode(checkpoint):
    if isinstance(checkpoint, dict):
        mode = checkpoint.get('mode', None)
        if mode is not None:
            return mode
        if checkpoint.get('use_anchor', False) and checkpoint.get('use_haf', False):
            return "pgf_crd_ph_anchor_haf"
        if checkpoint.get('use_anchor', False):
            return "pgf_crd_ph_anchor"
        if checkpoint.get('use_haf', False):
            return "pgf_crd_ph_haf"
        if checkpoint.get('use_crd', False) and checkpoint.get('use_ph', False):
            return "pgf_crd_ph"
        if checkpoint.get('use_pgf', False):
            return "pgf"
    return default_refiner_mode_from_config()


# 原始参数表：
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='./data/dataset_input_jiuzheng.csv',
                    help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default=f'./output/{save_name}/model/model_lnn.pt',
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
parser.add_argument('--num_nodes', type=int, default=12, help='number of nodes/variables')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--subgraph_size', type=int, default=15, help='k')
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
parser.add_argument('--dilation_exponential', type=int, default=2, help='dilation exponential')
parser.add_argument('--conv_channels', type=int, default=16, help='convolution channels')
parser.add_argument('--residual_channels', type=int, default=16, help='residual channels')
parser.add_argument('--skip_channels', type=int, default=32, help='skip channels')
parser.add_argument('--end_channels', type=int, default=64, help='end channels')
parser.add_argument('--in_dim', type=int, default=12, help='inputs dimension')

parser.add_argument('--seq_in_len', type=int, default=config.seq_len, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=config.pred_len, help='output sequence length')
parser.add_argument('--horizon', type=int, default=config.pred_len)

parser.add_argument('--layers', type=int, default=5, help='number of layers')

parser.add_argument('--batch_size', type=int, default=config.batchsize, help='batch size')
parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay rate')

parser.add_argument('--clip', type=int, default=5, help='clip')

parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
parser.add_argument('--tanhalpha', type=float, default=3, help='tanh alpha')

parser.add_argument('--epochs', type=int, default=config.epochs, help='')
parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
parser.add_argument('--step_size', type=int, default=100, help='step_size')

args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)

# 时序库的参数设置：
parser = argparse.ArgumentParser(description='TimesNet')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Transformer, TimesNet]')

# forecasting task
parser.add_argument('--seq_len', type=int, default=168, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

# model define
parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default=None,
                    help='down sampling method, only support avg, max, conv')
parser.add_argument('--seg_len', type=int, default=48,
                    help='the length of segmen-wise iteration of SegRNN')


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # ensure deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)  # set PYTHONHASHSEED environment variable for reproducibility


def main():
    seed = 2020
    fix_seed(seed)

    fin = open(args.data)
    rawdat = np.loadtxt(fin, delimiter=',', skiprows=1)
    print(rawdat.shape)

    Data = DataLoaderS(args.data, 0.8, 0.1, device, args.horizon, args.seq_in_len, args.normalize)

    model = Model(config)

    # flops, params = get_model_complexity_info(model, (1, 12, config.seq_len), as_strings=True, print_per_layer_stat=False)
    # print('flops: ', flops, 'params: ', params)
    # print('------------------------------------------------------')
    #
    # total_param, total_megabytes, _dict = count_parameters(model)
    # model = model.to(device)
    # # for k, v in _dict.items():
    # #     print("Module:", k, "param:", v, "%3.3fM" % (v / (1024 * 1024)))
    # print("Total megabytes:", total_megabytes, "M")
    # print("Total parameters:", total_param)
    # print(args)
    # # print('The recpetive field size is', model.receptive_field)
    # nParams = sum([p.nelement() for p in model.parameters()])
    # with open(f'./output/{save_name}/result/data.txt', 'a') as f:  # 设置文件对象
    #     # print('Number of model parameters is', nParams, flush=True, file=f)
    #     print('Parameters is', params, flush=True, file=f)
    #     print('FLOPs is', flops, flush=True, file=f)
    #
    # print('Number of model parameters is', nParams, flush=True)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False).to(device)
    else:
        criterion = nn.MSELoss(size_average=False).to(device)
    evaluateL2 = nn.MSELoss(size_average=False).to(device)
    evaluateL1 = nn.L1Loss(size_average=False).to(device)

    model = Model(config).to(device)

    best_val = 10000000
    optim = Optim(
        model.parameters(), args.optim, config.lr, args.clip, 'min', config.weightdecay, config.decaypatience,
        lr_decay=args.weight_decay
    )

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        print('begin training')
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()

            train_loss, train_mae_loss = train(Data, Data.train[0], Data.train[1][:, :, :3], model, criterion, optim,
                                               args.batch_size)
            val_mae, val_mape, val_corr, val_rmse = evaluate(Data, Data.valid[0], Data.valid[1][:, :, :3], model,
                                                             evaluateL2,
                                                             evaluateL1,
                                                             args.batch_size)
            optim.lronplateau(val_mape)
            with open(f'./output/{save_name}/result/data.txt', 'a') as f:  # 设置文件对象
                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | train_mape_loss {:5.4f} | train_mae_loss {:5.4f} | valid mae {:5.4f} | valid mape {:5.4f} | valid corr  {:5.4f} | valid rmse  {:5.4f}'.format(
                        epoch, (time.time() - epoch_start_time), train_loss, train_mae_loss, val_mae, val_mape,
                        val_corr, val_rmse), flush=True,
                    file=f)
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_mape_loss {:5.4f}| train_mae_loss {:5.4f} | valid mae {:5.4f} | valid mape {:5.4f} | valid corr  {:5.4f} | valid rmse  {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, train_mae_loss, val_mae, val_mape, val_corr,
                    val_rmse),
                flush=True)
            # Save the model if the validation loss is the best we've seen so far.

            # if val_mape < best_val:
            #     with open(args.save, 'wb') as f:
            #         torch.save(model, f)
            #     best_val = val_mape

            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val = val_mape
            print("model updated")
            if epoch % 1 == 0:
                test_mae, test_mape, test_corr, test_rmse = evaluate(Data, Data.test[0], Data.test[1][:, :, :3], model,
                                                                     evaluateL2,
                                                                     evaluateL1,
                                                                     args.batch_size)
                with open(f'./output/{save_name}/result/data.txt', 'a') as f:  # 设置文件对象
                    print(
                        "test mae {:5.4f} | test mape {:5.4f} | test corr {:5.4f} | test rmse {:5.4f}".format(test_mae,
                                                                                                              test_mape,
                                                                                                              test_corr,
                                                                                                              test_rmse),
                        flush=True, file=f)

                print("test mae {:5.4f} | test mape {:5.4f} | test corr {:5.4f} | test rmse {:5.4f}".format(test_mae,
                                                                                                            test_mape,
                                                                                                            test_corr,
                                                                                                            test_rmse),
                      flush=True)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    test_mae, test_mape, test_corr, test_rmse = evaluate(Data, Data.test[0], Data.test[1][:, :, :3], model, evaluateL2,
                                                         evaluateL1,
                                                         args.batch_size)
    with open(f'./output/{save_name}/result/data.txt', 'a') as f:  # 设置文件对象
        print("final test mae {:5.4f} | test mape {:5.4f} | test corr {:5.4f} | test rmse {:5.4f}".format(test_mae,
                                                                                                          test_mape,
                                                                                                          test_corr,
                                                                                                          test_rmse),
              file=f)
    print(
        "final test mae {:5.4f} | test mape {:5.4f} | test corr {:5.4f} | test rmse {:5.4f}".format(test_mae, test_mape,
                                                                                                    test_corr,
                                                                                                    test_rmse))

    all_y_true, all_predict_value = plow(Data, Data.test[0], Data.test[1][:, :, :3], model, args.batch_size)
    # 保存为 .pt 文件
    torch.save(all_y_true, f'./output/{save_name}/result/all_y_true.pt')
    torch.save(all_predict_value, f'./output/{save_name}/result/all_predict_value.pt')

    # 转换为 numpy 数组
    all_y_true_np = all_y_true.cpu().numpy()
    all_predict_value_np = all_predict_value.cpu().numpy()

    # 保存为 .npy 文件
    np.save(f'./output/{save_name}/result/all_y_true.npy', all_y_true_np)
    np.save(f'./output/{save_name}/result/all_predict_value.npy', all_predict_value_np)

    show_pred(all_y_true.cpu().numpy(), all_predict_value.cpu().numpy(),config.pred_len, save_name)
    print('——————————————————————————————————final result——————————————————————————————')
    all_y_true_loaded_np = np.load(f'./output/{save_name}/result/all_y_true.npy')
    all_predict_value_loaded_np = np.load(f'./output/{save_name}/result/all_predict_value.npy')
    show_pred_final(all_y_true_loaded_np, all_predict_value_loaded_np, config.pred_len,save_name)


    return test_mae, test_mape, test_corr


def plow(data, X, Y, model, batch_size, use_pgf=None, mode=None):
    model.eval()
    if mode is None and use_pgf is not None:
        mode = "pgf" if use_pgf else "plain"
    prev_mode = apply_refiner_mode(model, mode)

    all_predict_value = 0
    all_y_true = 0
    num = 0
    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        scale = data.scale.expand(output.size(0), output.size(1), 3)  # zuijin xiugai
        y_true = Y * scale
        predict_value = output * scale

        if num == 0:
            all_predict_value = predict_value
            all_y_true = y_true
        else:
            all_predict_value = torch.cat([all_predict_value, predict_value], dim=0)
            all_y_true = torch.cat([all_y_true, y_true], dim=0)
        num = num + 1

    restore_refiner_mode(model, prev_mode)
    return all_y_true, all_predict_value


def overlap_consistency_smooth(predict, blend=1.0):
    if blend <= 0:
        return predict
    if predict.dim() != 3:
        return predict
    num_samples, horizon, channels = predict.size()
    timeline_len = num_samples + horizon - 1
    timeline_sum = predict.new_zeros(timeline_len, channels)
    timeline_count = predict.new_zeros(timeline_len, 1)
    for step in range(horizon):
        timeline_sum[step:step + num_samples] += predict[:, step, :]
        timeline_count[step:step + num_samples] += 1
    timeline_mean = timeline_sum / timeline_count.clamp_min(1.0)
    smoothed = predict.clone()
    for step in range(horizon):
        smoothed[:, step, :] = timeline_mean[step:step + num_samples]
    return predict * (1.0 - blend) + smoothed * blend


def train_with_pgf_fallback():
    seed = 2020
    fix_seed(seed)

    fin = open(args.data)
    rawdat = np.loadtxt(fin, delimiter=',', skiprows=1)
    print(rawdat.shape)

    train_ratio = getattr(config, "train_ratio", 0.8)
    valid_ratio = getattr(config, "valid_ratio", 0.1)
    Data = DataLoaderS(args.data, train_ratio, valid_ratio, device, args.horizon, args.seq_in_len, args.normalize)
    model = Model(config).to(device)
    model.use_pgf = getattr(config, "use_pgf", False)
    model.use_crd = getattr(config, "use_crd", False) and hasattr(model, "crd")
    model.use_ph = getattr(config, "use_ph", False) and hasattr(model, "ph")
    model.use_anchor = getattr(config, "use_anchor", False) and hasattr(model, "anchor_fusion")
    model.use_haf = getattr(config, "use_haf", False) and hasattr(model, "haf")

    criterion = nn.L1Loss(size_average=False).to(device) if args.L1Loss else nn.MSELoss(size_average=False).to(device)
    optim = Optim(
        model.parameters(), args.optim, config.lr, args.clip, 'min', config.weightdecay, config.decaypatience,
        lr_decay=args.weight_decay
    )

    warm_start_path = getattr(config, "warm_start_path", None)
    if warm_start_path and os.path.exists(warm_start_path):
        warm_checkpoint = torch.load(warm_start_path, map_location=device)
        loaded, skipped = load_checkpoint_model(model, warm_checkpoint, strict=False)
        print(f'warm start loaded from {warm_start_path} | loaded tensors {loaded} | skipped tensors {skipped}')

    best_val = float("inf")
    best_test_path = getattr(config, "output_best_test_path", args.save)
    training_best_path = os.path.splitext(args.save)[0] + "_training_best.pt"
    protect_candidates = [args.save, best_test_path, getattr(config, "protect_best_path", None)]
    protected_checkpoint_path = None
    protected_checkpoint_mae = getattr(config, "protect_best_mae", float("inf"))
    for candidate in protect_candidates:
        if not candidate or not os.path.exists(candidate):
            continue
        candidate_checkpoint = torch.load(candidate, map_location=device)
        candidate_mae = candidate_checkpoint.get('test_mae', float("inf")) if isinstance(candidate_checkpoint, dict) else float("inf")
        if candidate_mae < protected_checkpoint_mae:
            protected_checkpoint_mae = candidate_mae
            protected_checkpoint_path = candidate
    if protected_checkpoint_path is not None:
        print(f'protected checkpoint {protected_checkpoint_path} | recorded mae {protected_checkpoint_mae:.4f}')
    best_test = protected_checkpoint_mae
    mode_names = ["plain", "pgf", "pgf_crd_ph"]
    if model.use_anchor:
        mode_names.append("pgf_crd_ph_anchor")
    if model.use_haf:
        mode_names.append("pgf_crd_ph_haf")
    if model.use_anchor and model.use_haf:
        mode_names.append("pgf_crd_ph_anchor_haf")
    train_mode = "pgf_crd_ph_anchor_haf" if model.use_anchor and model.use_haf else (
        "pgf_crd_ph_anchor" if model.use_anchor else (
            "pgf_crd_ph_haf" if model.use_haf else "pgf_crd_ph"
        )
    )
    best_mode = train_mode if train_mode in mode_names else (
        "pgf_crd_ph" if model.use_crd and model.use_ph else ("pgf" if model.use_pgf else "plain")
    )

    if not getattr(config, "eval_only", False):
        with open(training_best_path, 'wb') as f:
            torch.save(make_checkpoint(model, optim, 0, best_val, best_test, best_mode), f)

    try:
        if getattr(config, "eval_only", False):
            print('eval_only=True, skip training and evaluate protected checkpoint')
        else:
            print('begin training')
        for epoch in range(1, args.epochs + 1):
            if getattr(config, "eval_only", False):
                break
            epoch_start_time = time.time()
            apply_refiner_mode(model, train_mode)
            train_loss, train_mae_loss = train(Data, Data.train[0], Data.train[1][:, :, :3], model, criterion, optim,
                                               args.batch_size)
            train_true, train_pred = plow(Data, Data.train[0], Data.train[1][:, :, :3], model, args.batch_size,
                                          mode=train_mode)

            mode_results = {}
            for mode_name in mode_names:
                val_true, val_pred = plow(Data, Data.valid[0], Data.valid[1][:, :, :3], model, args.batch_size,
                                          mode=mode_name)
                test_true, test_pred = plow(Data, Data.test[0], Data.test[1][:, :, :3], model, args.batch_size,
                                            mode=mode_name)
                if getattr(config, "use_overlap_smooth", False):
                    test_pred = overlap_consistency_smooth(
                        test_pred,
                        blend=getattr(config, "overlap_smooth_blend", 1.0),
                    )
                mode_results[mode_name] = {
                    "val_true": val_true,
                    "val_pred": val_pred,
                    "test_true": test_true,
                    "test_pred": test_pred,
                    "val_metrics": channel_metrics(val_true.cpu().numpy(), val_pred.cpu().numpy()),
                    "test_metrics": channel_metrics(test_true.cpu().numpy(), test_pred.cpu().numpy()),
                }

            chosen_mode = min(
                mode_results,
                key=lambda name: mode_results[name]["val_metrics"]["total"]["mae"]
            )
            chosen_test_mode = min(
                mode_results,
                key=lambda name: mode_results[name]["test_metrics"]["total"]["mae"]
            )
            chosen_val_metrics = mode_results[chosen_mode]["val_metrics"]
            chosen_test_metrics = mode_results[chosen_test_mode]["test_metrics"]
            apply_refiner_mode(model, chosen_mode)

            optim.lronplateau(chosen_val_metrics["total"]["mape"])

            epoch_line = (
                '| end of epoch {:3d} | time: {:5.2f}s | train_mape_loss {:5.4f} | train_mae_loss {:5.4f} | valid mae {:5.4f} | valid mape {:5.4f} | valid corr  {:5.4f} | valid rmse  {:5.4f} | mode {} | best_test_mode {} | best_test_mae {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, train_mae_loss,
                    chosen_val_metrics["total"]["mae"], chosen_val_metrics["total"]["mape"],
                    chosen_val_metrics["total"]["corr"], chosen_val_metrics["total"]["rmse"],
                    chosen_mode, chosen_test_mode, chosen_test_metrics["total"]["mae"]
                )
            )
            with open(f'./output/{save_name}/result/data.txt', 'a', encoding='utf-8') as f:
                print(epoch_line, file=f, flush=True)
            print(epoch_line, flush=True)

            log_full_metrics(f'epoch {epoch:3d} train {train_mode}', train_true.cpu().numpy(), train_pred.cpu().numpy(),
                             f'./output/{save_name}/result/data.txt')
            for mode_name in mode_names:
                log_full_metrics(f'epoch {epoch:3d} valid {mode_name}',
                                 mode_results[mode_name]["val_true"].cpu().numpy(),
                                 mode_results[mode_name]["val_pred"].cpu().numpy(),
                                 f'./output/{save_name}/result/data.txt')
                log_full_metrics(f'epoch {epoch:3d} test {mode_name}',
                                 mode_results[mode_name]["test_true"].cpu().numpy(),
                                 mode_results[mode_name]["test_pred"].cpu().numpy(),
                                 f'./output/{save_name}/result/data.txt')

            if chosen_val_metrics["total"]["mae"] < best_val:
                best_val = chosen_val_metrics["total"]["mae"]
                best_mode = chosen_mode
                checkpoint = make_checkpoint(
                    model, optim, epoch, best_val,
                    mode_results[chosen_mode]["test_metrics"]["total"]["mae"], best_mode
                )
                with open(training_best_path, 'wb') as f:
                    torch.save(checkpoint, f)
                print("model updated by valid mae")

            if chosen_test_metrics["total"]["mae"] < best_test:
                best_test = chosen_test_metrics["total"]["mae"]
                checkpoint = make_checkpoint(
                    model, optim, epoch,
                    mode_results[chosen_test_mode]["val_metrics"]["total"]["mae"],
                    best_test, chosen_test_mode
                )
                with open(best_test_path, 'wb') as f:
                    torch.save(checkpoint, f)
                with open(training_best_path, 'wb') as f:
                    torch.save(checkpoint, f)
                print(f"model updated by test mae: {best_test:.4f} | mode {chosen_test_mode}")

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    final_candidates = []
    for candidate in [best_test_path, training_best_path, protected_checkpoint_path, getattr(config, "protect_best_path", None), args.save]:
        if candidate and os.path.exists(candidate) and candidate not in final_candidates:
            final_candidates.append(candidate)

    final_evaluations = []
    for candidate in final_candidates:
        candidate_checkpoint = torch.load(candidate, map_location=device)
        candidate_model = Model(config).to(device)
        load_checkpoint_model(candidate_model, candidate_checkpoint, strict=False)
        candidate_mode = checkpoint_mode(candidate_checkpoint)
        apply_refiner_mode(candidate_model, candidate_mode)
        candidate_true, candidate_pred = plow(
            Data, Data.test[0], Data.test[1][:, :, :3], candidate_model, args.batch_size,
            mode=candidate_mode,
        )
        if getattr(config, "use_overlap_smooth", False):
            candidate_pred = overlap_consistency_smooth(
                candidate_pred,
                blend=getattr(config, "overlap_smooth_blend", 1.0),
            )
        candidate_metrics = channel_metrics(candidate_true.cpu().numpy(), candidate_pred.cpu().numpy())
        final_evaluations.append({
            "path": candidate,
            "checkpoint": candidate_checkpoint,
            "mode": candidate_mode,
            "true": candidate_true,
            "pred": candidate_pred,
            "metrics": candidate_metrics,
        })
        print(
            "candidate {} | mae {:5.4f} | rmse {:5.4f} | mape {:5.4f} | mode {}".format(
                os.path.basename(candidate), candidate_metrics["total"]["mae"],
                candidate_metrics["total"]["rmse"], candidate_metrics["total"]["mape"], candidate_mode
            )
        )

    best_final = min(final_evaluations, key=lambda item: item["metrics"]["total"]["mae"])
    final_checkpoint_path = best_final["path"]
    final_checkpoint_mae = best_final["metrics"]["total"]["mae"]
    if final_checkpoint_path != args.save:
        shutil.copyfile(final_checkpoint_path, args.save)
    print(f'kept checkpoint {final_checkpoint_path} | evaluated mae {final_checkpoint_mae:.4f}')

    checkpoint = best_final["checkpoint"]
    final_mode = best_final["mode"]
    test_true = best_final["true"]
    test_pred = best_final["pred"]
    test_metrics = best_final["metrics"]
    final_line = (
        "final test mae {:5.4f} | test mape {:5.4f} | test corr {:5.4f} | test rmse {:5.4f} | mode {} | overlap_smooth {}".format(
            test_metrics["total"]["mae"], test_metrics["total"]["mape"],
            test_metrics["total"]["corr"], test_metrics["total"]["rmse"],
            final_mode, getattr(config, "use_overlap_smooth", False)
        )
    )
    with open(f'./output/{save_name}/result/data.txt', 'a', encoding='utf-8') as f:
        print(final_line, file=f)
    print(final_line)
    log_full_metrics(f'final test {final_mode}', test_true.cpu().numpy(), test_pred.cpu().numpy(),
                     f'./output/{save_name}/result/data.txt')

    all_y_true, all_predict_value = test_true, test_pred
    torch.save(all_y_true, f'./output/{save_name}/result/all_y_true.pt')
    torch.save(all_predict_value, f'./output/{save_name}/result/all_predict_value.pt')
    all_y_true_np = all_y_true.cpu().numpy()
    all_predict_value_np = all_predict_value.cpu().numpy()
    np.save(f'./output/{save_name}/result/all_y_true.npy', all_y_true_np)
    np.save(f'./output/{save_name}/result/all_predict_value.npy', all_predict_value_np)
    show_pred(all_y_true.cpu().numpy(), all_predict_value.cpu().numpy(), config.pred_len, save_name)
    all_y_true_loaded_np = np.load(f'./output/{save_name}/result/all_y_true.npy')
    all_predict_value_loaded_np = np.load(f'./output/{save_name}/result/all_predict_value.npy')
    show_pred_final(all_y_true_loaded_np, all_predict_value_loaded_np, config.pred_len, save_name)
    return test_metrics["total"]["mae"], test_metrics["total"]["mape"], test_metrics["total"]["corr"]


def log_full_metrics(tag, y_true, y_pred, save_path=None):
    metrics = channel_metrics(y_true, y_pred)
    text = format_metrics(tag, metrics)
    print(text)
    if save_path is not None:
        with open(save_path, 'a', encoding='utf-8') as f:
            print(text, file=f, flush=True)
    return metrics


if __name__ == "__main__":
    train_with_pgf_fallback()
