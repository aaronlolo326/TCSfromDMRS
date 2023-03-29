import argparse
import collections
from json import decoder
import datetime
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

import numpy as np
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
# from trainer.trainer_thread import Trainer
from utils import prepare_device, get_transformed_info

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

from pprint import pprint
import os
from collections import Counter, defaultdict
from itertools import chain
import copy

from scipy.spatial import distance
import numpy as np

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_dtype(torch.float64)
np.random.seed(SEED)

def get_trainer_args(config):

    logger = config.get_logger('train')

    # declare decoders
    MIN_PRED_FUNC_FREQ = config["data_loader"]["args"]["min_pred_func_freq"]
    MIN_CONTENT_PRED_FREQ = config["data_loader"]["args"]["min_content_pred_freq"]

    log_dir = os.path.join("saved/log", config["name"])
    run_dir = os.path.join("saved/run", config["name"])

    transformed_dir  = config["data_loader"]["args"]["transformed_dir"]
    transformed_info_dir = os.path.join(transformed_dir, "info")

    pred_func2cnt, content_pred2cnt, pred2ix, content_predarg2ix, pred_func2ix = get_transformed_info(transformed_info_dir)

    sorted_pred_func2ix = sorted(pred_func2ix.items(), key = lambda x: x[1])
    pred_funcs = []
    for pred_func, ix in sorted_pred_func2ix:
        pred_funcs.append(pred_func)

    # if use generative model loss, we need indices:
    if not config['decoder_arch']['args']['use_truth']:
        max_pred_ix = max(pred2ix.values())
        max_pred_func_ix = max(pred_func2ix.values())
        pred2cnt = defaultdict()
        pred_func_ix2arg_num = [-1 for i in range(max_pred_func_ix + 1)]
        pred_func_ix2cnt = [-1 for i in range(max_pred_func_ix + 1)]
        pred_ix2non_arg0_num2pred_func_ix = [[-1, -1, -1, -1] for i in range(max_pred_ix + 1)]
        pred_ix2arg0_pred_func_ix = [[-1] for i in range(max_pred_ix + 1)]
        pred_ix2non_arg0_num2cnt = [[0, 0, 0, 0] for i in range(max_pred_ix + 1)]


        for pred, pred_ix in pred2ix.items():
            # pred2cnt
            if pred2ix[pred] in content_pred2cnt:
                pred2cnt[pred2ix[pred]] = content_pred2cnt[pred2ix[pred]]
            else:
                pred2cnt[pred2ix[pred]] = MIN_CONTENT_PRED_FREQ
        
        pred_ix2arg_num2pred_func_ix = [[-1, -1, -1, -1, -1] for i in range(max_pred_ix + 1)]
        for pred_func, pred_func_ix in pred_func2ix.items():
            pred, arg = pred_func.rsplit("@", 1)
            arg_num = int(arg[-1])
            pred_ix2arg_num2pred_func_ix[pred2ix[pred]][arg_num] = pred_func_ix
            pred_func_ix2arg_num[pred_func_ix] = arg_num
        pred_func_ix2arg_num = torch.tensor(pred_func_ix2arg_num)
        pred_ix2arg_num2pred_func_ix = torch.tensor(pred_ix2arg_num2pred_func_ix)

        pred_ix2arg_num_sum = [0 for i in range(max_pred_ix + 1)]
        for pred_ix, arg_num2pred_func_ix in enumerate(pred_ix2arg_num2pred_func_ix):
            for arg_num, pred_func_ix in enumerate(arg_num2pred_func_ix):
                if pred_func_ix != -1:
                    pred_ix2arg_num_sum[pred_ix] += 2 ** arg_num

        arg_num_sum_set = set(pred_ix2arg_num_sum)
        arg_num_sum2subset = defaultdict(set)
        arg_num_sum2subset_new = defaultdict(set)
        arg_num_sum2add = defaultdict(list)
        for i in [1,2,4,8,16]:
            arg_num_sum2subset_new[i] = set([i])
            arg_num_sum2add[i] = [j for j in [1,2,4,8,16] if j != i]
        # arg_num_sum2add = defaultdict(list, {1: [2,4,8,16], 2: [1,4,8,16], 4: [1,2,8,16], 8: [1,2,4,16], 16: [1,2,4,8]})
        # has_add = True

        while arg_num_sum2subset_new != arg_num_sum2subset:
            arg_num_sum2subset = copy.deepcopy(arg_num_sum2subset_new)
            for arg_num_sum in arg_num_sum2subset:
                arg_num_sum2subset_new[arg_num_sum].add(arg_num_sum)
                for add in arg_num_sum2add[arg_num_sum]:
                    arg_num_sum2subset_new[arg_num_sum + add] = arg_num_sum2subset_new[arg_num_sum + add].union(arg_num_sum2subset_new[arg_num_sum])
                    arg_num_sum2add[arg_num_sum + add] = [a for a in arg_num_sum2add[arg_num_sum] if a != add]

        arg_num_sum2preds_ix = defaultdict(list)
        for pred_ix, arg_num_sum in enumerate(pred_ix2arg_num_sum):
            for sub_arg_num_sum in arg_num_sum2subset[arg_num_sum]:
                arg_num_sum2preds_ix[sub_arg_num_sum].append(pred_ix)

    print ("Initializing encoder ...")
    # content_preds = set(content_pred2cnt)
    if config['encoder_arch']['type'] == 'MyEncoder':
        num_embs = len(pred2ix)
    elif config['encoder_arch']['type'] == 'PASEncoder':
        num_embs = len(content_predarg2ix)

    encoder = config.init_obj('encoder_arch', module_arch, num_embs = num_embs)
    for p in encoder.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
    # logger.info(decoder)

    print ("Initializing decoder ...")
    if not config['decoder_arch']['args']['use_truth']:
        decoder = config.init_obj('decoder_arch', module_arch, num_sem_funcs = len(pred_funcs),
            pred2cnt = pred2cnt, pred_ix2arg_num2pred_func_ix = pred_ix2arg_num2pred_func_ix, arg_num_sum2preds_ix = arg_num_sum2preds_ix, pred_func_ix2arg_num = pred_func_ix2arg_num
        )
    else:
        decoder = config.init_obj('decoder_arch', module_arch, num_sem_funcs = len(pred_funcs), pred_func2cnt = pred_func2cnt, pred_funcs = pred_funcs) 

        
    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    criterion = None
    metric_ftns = [getattr(module_metric, met) for met in config['metrics']]

    trainer_args = {
        "encoder": encoder,
        "pred2ix": pred2ix,
        "pred_func2ix": pred_func2ix,
        "decoder": decoder,
        "criterion": criterion,
        "metric_ftns": metric_ftns,
        "config": config
    }

    return trainer_args

def main(config, pth):

    device = 'cpu'
    world_size = 1
    ddp = False
    
    trainer_args = get_trainer_args(config)
    trainer = Trainer(world_size = world_size, device = device, rank = -1, ddp = ddp, **trainer_args)

    trainer.encoder_opt = config.init_obj('encoder_optimizer', torch.optim, trainer.encoder.parameters())
    trainer.decoder_opt = config.init_obj('decoder_optimizer', torch.optim, trainer.decoder.parameters())

    trainer._resume_checkpoint(pth)


    # # encoder
    # print (trainer.encoder.embeds.weight)
    # # print (trainer.encoder.fc_pair.weight)
    # print (trainer.encoder.fc_mu.weight)
    # print (trainer.encoder.fc_sigma.weight)

    # dog = trainer.pred_func2ix['_dog_n_1@ARG0']
    # cat = trainer.pred_func2ix['_cat_n_1@ARG0']

    # ix2pred = {ix: pred for pred, ix in trainer.pred2ix.items()}
    # ws = []

    # for w_idx, w in enumerate(trainer.encoder.embeds.weight):
    #     w = w.detach().numpy()
    #     ws.append(w)

    # ws = np.stack(ws, axis=0)

    # ix2norm = {ix: np.linalg.norm(w) for ix, w in enumerate(ws)}
    # ranked_norm = sorted(ix2norm.items(), key = lambda x: x[1])
    # ranked_norm_r = sorted(ix2norm.items(), key = lambda x: x[1], reverse = True)

    # k = 50
    # print ("Top {} large embeds:".format(k))
    # for rank, (ix, norm) in enumerate(ranked_norm_r):
    #     if rank >= k:
    #         break
    #     print (ix2pred[ix], "\t", norm)
    # print ()
    # print ("Top {} small embeds:".format(k))
    # for rank, (ix, norm) in enumerate(ranked_norm):
    #     if rank >= k:
    #         break
    #     print (ix2pred[ix], "\t", norm)
    # print ()

    # k = 30
    # q = input("Enter pred: ")
    # while q != "0":
    #     try:
    #         q_pf_ix = trainer.pred2ix[q]
    #     except:
    #         print ("{} not found".format(q))
    #     else:
    #         cos_sim = []
    #         q_w = ws[q_pf_ix]

    #         ix2cos_dis = {ix: distance.cosine(q_w, w) for ix, w in enumerate(ws)}
    #         ix2euc_dis = {ix: distance.euclidean(q_w, w) for ix, w in enumerate(ws)}
    #         ix2dot = {ix: np.dot(q_w, w) for ix, w in enumerate(ws)}

    #         ranked_cos_dist = sorted(ix2cos_dis.items(), key = lambda x: x[1])
    #         ranked_euc_dist = sorted(ix2euc_dis.items(), key = lambda x: x[1])
    #         ranked_dot = sorted(ix2dot.items(), key = lambda x: x[1], reverse = True)

    #         print ("Top {} cos_dist-close embeds:".format(k))
    #         for rank, (ix, cos_dist) in enumerate(ranked_cos_dist):
    #             if rank >= k:
    #                 break
    #             print (ix2pred[ix], "\t", cos_dist)
    #         print ()

    #         print ("Top {} euc_dist-close embeds:".format(k))
    #         for rank, (ix, euc_dist) in enumerate(ranked_euc_dist):
    #             if rank >= k:
    #                 break
    #             print (ix2pred[ix], "\t", euc_dist)
    #         print ()

    #         print ("Top {} dot-close embeds:".format(k))
    #         for rank, (ix, dot_prod) in enumerate(ranked_dot):
    #             if rank >= k:
    #                 break
    #             print (ix2pred[ix], "\t", dot_prod)
    #         print ()
            
    #     finally:
    #         q = input("Enter pred: ")

    # decoder

    dog = trainer.pred_func2ix['_dog_n_1@ARG0']
    cat = trainer.pred_func2ix['_cat_n_1@ARG0']

    ix2pred_func = {ix: pred_func for pred_func, ix in trainer.pred_func2ix.items()}
    ws = []

    sparse_sem_funcs = config['decoder_arch']['args']['sparse_sem_funcs']
    if not sparse_sem_funcs:
        for w_idx, w in enumerate(trainer.decoder.sem_funcs):
            w = w.detach().numpy()
            ws.append(w)
    else:
        for w_idx, w in enumerate(trainer.decoder.sem_funcs.weight):
            w = w.detach().numpy()
            ws.append(w)

    ws = np.stack(ws, axis=0)
    
    dim = int(config['decoder_arch']['args']['input_dim'] / 2)

    ix2norm = {ix: np.linalg.norm(w[:dim]) for ix, w in enumerate(ws) if ix2pred_func[ix].endswith("ARG0")}
    ranked_norm = sorted(ix2norm.items(), key = lambda x: x[1])
    ranked_norm_r = sorted(ix2norm.items(), key = lambda x: x[1], reverse = True)

    k = 50
    print ("Top {} large sem_funcs:".format(k))
    for rank, (ix, norm) in enumerate(ranked_norm_r):
        if rank >= k:
            break
        print (ix2pred_func[ix], "\t", norm)
    print ()
    print ("Top {} small sem_funcs:".format(k))
    for rank, (ix, norm) in enumerate(ranked_norm):
        if rank >= k:
            break
        print (ix2pred_func[ix], "\t", norm)
    print ()

    k = 30
    q = input("Enter pred (ARG0): ")
    while q != "0":
        try:
            q_pf_ix = trainer.pred_func2ix['{}@ARG0'.format(q)]
        except:
            print ("{}@ARG0 not found".format(q))
        else:
            cos_sim = []
            q_w = ws[q_pf_ix][:dim]

            # all [:dim]
            ix2cos_dis_dim1 = {ix: distance.cosine(q_w, w[:dim]) for ix, w in enumerate(ws)}
            ix2euc_dis_dim1 = {ix: distance.euclidean(q_w, w[:dim]) for ix, w in enumerate(ws)}
            ix2dot_dim1 = {ix: np.dot(q_w, w[:dim]) for ix, w in enumerate(ws)} # np.abs(np.abs(q_w[-1]) - np.abs(w[-1]))

            ranked_cos_dist_dim1 = sorted(ix2cos_dis_dim1.items(), key = lambda x: x[1])
            ranked_euc_dist_dim1 = sorted(ix2euc_dis_dim1.items(), key = lambda x: x[1])
            ranked_dot_dim1 = sorted(ix2dot_dim1.items(), key = lambda x: x[1], reverse = True)
            
            # all [dim:dim*2] minus one-place's
            ix2cos_dis_dim2 = {ix: distance.cosine(q_w, w[dim:dim * 2]) for ix, w in enumerate(ws) if not ix2pred_func[ix].endswith("ARG0")}
            ix2euc_dis_dim2 = {ix: distance.euclidean(q_w, w[dim:dim * 2]) for ix, w in enumerate(ws) if not ix2pred_func[ix].endswith("ARG0")}
            ix2dot_dim2 = {ix: np.dot(q_w, w[dim:dim * 2]) for ix, w in enumerate(ws) if not ix2pred_func[ix].endswith("ARG0")} # np.abs(np.abs(q_w[-1]) - np.abs(w[-1]))

            ranked_cos_dist_dim2 = sorted(ix2cos_dis_dim2.items(), key = lambda x: x[1])
            ranked_euc_dist_dim2 = sorted(ix2euc_dis_dim2.items(), key = lambda x: x[1])
            ranked_dot_dim2 = sorted(ix2dot_dim2.items(), key = lambda x: x[1], reverse = True)

            print ("Top {} [:dim]cos_dist-close sem_funcs:".format(k))
            for rank, (ix, cos_dist) in enumerate(ranked_cos_dist_dim1):
                if rank >= k:
                    break
                print (ix2pred_func[ix], "\t", cos_dist)
            print ("Top {} [dim:dim*2]cos_dist-close sem_funcs:".format(k))
            for rank, (ix, cos_dist) in enumerate(ranked_cos_dist_dim2):
                if rank >= k:
                    break
                print (ix2pred_func[ix], "\t", cos_dist)
            print ()

            print ("Top {} [:dim]euc_dist-close sem_funcs:".format(k))
            for rank, (ix, euc_dist) in enumerate(ranked_euc_dist_dim1):
                if rank >= k:
                    break
                print (ix2pred_func[ix], "\t", euc_dist)
            print ("Top {} [dim:dim*2]euc_dist-close sem_funcs:".format(k))
            for rank, (ix, euc_dist) in enumerate(ranked_euc_dist_dim2):
                if rank >= k:
                    break
                print (ix2pred_func[ix], "\t", euc_dist)
            print ()

            print ("Top {} [:dim]dot-close sem_funcs:".format(k))
            for rank, (ix, dot_prod) in enumerate(ranked_dot_dim1):
                if rank >= k:
                    break
                print (ix2pred_func[ix], "\t", dot_prod)
            print ("Top {} [dim:dim*2]dot-close sem_funcs:".format(k))
            for rank, (ix, dot_prod) in enumerate(ranked_dot_dim2):
                if rank >= k:
                    break
                print (ix2pred_func[ix], "\t", dot_prod)
            print ()
            
        finally:
            q = input("Enter pred (ARG0): ")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-p', '--pth', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(parser, options)
    main(config, args.pth)