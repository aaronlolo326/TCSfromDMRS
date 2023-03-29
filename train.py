import argparse
import collections
import datetime
import torch
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
from collections import defaultdict
import copy

# custom_seed = 29
torch.use_deterministic_algorithms(mode = True)
torch.autograd.set_detect_anomaly(False) # False

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_dtype(torch.float64)


def setup(rank, world_size, device, sparse_sem_funcs):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'
    if device == 'cpu' or sparse_sem_funcs:
        backend = "gloo" 
    else:
        backend = "nccl"  #nccl
    # initialize the process group
    # If you plan on using this module with a nccl backend or a gloo backend (that uses Infiniband),
    # together with a DataLoader that uses multiple workers, please change the multiprocessing start method to forkserver (Python 3 only) or spawn.
    # Unfortunately Gloo (that uses Infiniband) and NCCL2 are not fork safe, and you will likely experience deadlocks if you don’t change this setting.
    dist.init_process_group(backend, rank = rank, world_size = world_size, timeout=datetime.timedelta(seconds = 1000))

def cleanup():
    dist.destroy_process_group()

def get_indices(content_pred2cnt, pred2ix, pred_func2ix, MIN_CONTENT_PRED_FREQ):
    max_pred_ix = max(pred2ix.values())
    max_pred_func_ix = max(pred_func2ix.values())
    pred2cnt = defaultdict()
    pred_func_ix2arg_num = [-1 for i in range(max_pred_func_ix + 1)]
    # pred_func_ix2cnt = [-1 for i in range(max_pred_func_ix + 1)]
    # pred_ix2non_arg0_num2pred_func_ix = [[-1, -1, -1, -1] for i in range(max_pred_ix + 1)]
    # pred_ix2arg0_pred_func_ix = [[-1] for i in range(max_pred_ix + 1)]
    # pred_ix2non_arg0_num2cnt = [[0, 0, 0, 0] for i in range(max_pred_ix + 1)]

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

    return pred2cnt, pred_ix2arg_num2pred_func_ix, arg_num_sum2preds_ix, pred_func_ix2arg_num

def get_trainer_args(config, train = True):

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
    if not config['decoder_arch']['args']['use_truth'] and train:
        pred2cnt, pred_ix2arg_num2pred_func_ix, arg_num_sum2preds_ix, pred_func_ix2arg_num = get_indices(content_pred2cnt, pred2ix, pred_func2ix, MIN_CONTENT_PRED_FREQ)

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
        if train:
            decoder = config.init_obj('decoder_arch', module_arch, num_sem_funcs = len(pred_funcs), train = train,
                pred2cnt = pred2cnt, pred_ix2arg_num2pred_func_ix = pred_ix2arg_num2pred_func_ix,
                arg_num_sum2preds_ix = arg_num_sum2preds_ix, pred_func_ix2arg_num = pred_func_ix2arg_num
            )
        else:
            decoder = config.init_obj('decoder_arch', module_arch, num_sem_funcs = len(pred_funcs), train = train
            )
    else:
        decoder = config.init_obj('decoder_arch', module_arch, num_sem_funcs = len(pred_funcs), pred_func2cnt = pred_func2cnt, pred_funcs = pred_funcs) 
    if config['decoder_arch']['args']['sparse_sem_funcs']:
        pass
    else:
        for p in decoder.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    # count num_params
    num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print ("num_params of encoder (some are dummy for arg0 sem_funcs):", num_params)
    num_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print ("num_params of decoder (some are dummy for arg0 sem_funcs):", num_params)



    # get function handles of loss and metrics
    # criterion = getattr(module_loss, config['loss'])
    criterion = None
    metric_ftns = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler

    # trainable_params = [{'params': sem_funcs[pregarg].parameters()} for pregarg in sem_funcs]

    ## one parameter group
    # sem_funcs_params = chain.from_iterable([sem_func.parameters() for _, sem_func in sem_funcs.items()])
    # trainable_params = filter(lambda p: p.requires_grad, sem_funcs_params)

    ## each decoder has it's parameter group
    # trainable_params = [{'params': sem_func.parameters()} for _, sem_func in sem_funcs.items()]
    # trainable_params.extend([{'params': encoder.parameters()}])
    # print ("Initializing optimizer for the encoder and semantic functions ...")
    # optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    # print ("Initializing learning rate scheduler for the optimizer ...")
    # lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

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


def ddp_trainer(rank, ddp_args):

    world_size, ddp, device, config = ddp_args
    print(f"Running DDP on rank {rank}.")

    sparse_sem_funcs = config['decoder_arch']['args']['sparse_sem_funcs']
    setup(rank, world_size, device, sparse_sem_funcs)

    trainer_args = get_trainer_args(config)
    trainer = Trainer(world_size = world_size, device = device, rank = rank, ddp = ddp, **trainer_args)
    trainer.train()

    cleanup()


def main(config, seed):

    # fix random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    # device, device_ids = None, []
    ddp = config['ddp']
    device = None
    world_size = None
    if device_ids == []:
        device = 'cpu'
        num_thread = torch.get_num_threads()
        # torch.set_num_threads(int(num_thread))
        print ("using {} with {} threads".format(device, num_thread))
        world_size = min(int(num_thread), 2)
    else:
        device = 'cuda:0'
        print ("using {} with device_ids: {}".format(device, device_ids))
        if len(device_ids) > 1:
            ddp = True
            world_size = min(len(device_ids), 12)
        else:
            ddp = False

    if ddp and world_size > 1:
        mp.spawn(ddp_trainer,
            args = ((world_size, ddp, device, config),),
            nprocs = world_size,
            join = True
        )
    else:
        trainer_args = get_trainer_args(config)
        trainer = Trainer(world_size = world_size, device = device, rank = -1, ddp = ddp, **trainer_args)
        trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-s', '--seed'], type=int, target='seed')
    ]
    config = ConfigParser.from_args(parser, options)
    args = parser.parse_args()
    main(config, args.seed)