import argparse
import collections
from json import decoder
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
from collections import Counter, defaultdict
from itertools import chain

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_dtype(torch.float64)
np.random.seed(SEED)

def setup(rank, world_size, cpu):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'
    if cpu:
        backend = "gloo" 
    else:
        backend = "gloo"  #nccl
    # initialize the process group
    # If you plan on using this module with a nccl backend or a gloo backend (that uses Infiniband),
    # together with a DataLoader that uses multiple workers, please change the multiprocessing start method to forkserver (Python 3 only) or spawn.
    # Unfortunately Gloo (that uses Infiniband) and NCCL2 are not fork safe, and you will likely experience deadlocks if you don’t change this setting.
    dist.init_process_group(backend, rank = rank, world_size = world_size, timeout=datetime.timedelta(seconds=500))

def cleanup():
    dist.destroy_process_group()

def get_trainer_args(config):

    logger = config.get_logger('train')

    # declare decoders
    MIN_PRED_FUNC_FREQ = config["data_loader"]["args"]["min_pred_func_freq"]
    MIN_CONTENT_PRED_FREQ = config["data_loader"]["args"]["min_content_pred_freq"]

    log_dir = os.path.join("saved/log", config["name"])
    run_dir = os.path.join("saved/run", config["name"])

    transformed_dir  = config["data_loader"]["args"]["transformed_dir"]
    transformed_info_dir = os.path.join(transformed_dir, "info")
    # log_dir = config.log_dir
    # pred_func2cnt_file_path = os.path.join(transformed_info_dir, "pred_func2cnt.txt")
    # content_pred2cnt_file_path = os.path.join(transformed_info_dir, "content_pred2cnt.txt")
    # pred2ix_file_path = os.path.join(transformed_info_dir, "pred2ix.txt")
    # pred_func2ix_file_path = os.path.join(transformed_info_dir, "pred_func2ix.txt")

    pred_func2cnt, content_pred2cnt, pred2ix, pred_func2ix = get_transformed_info(transformed_info_dir)

    sorted_pred_func2ix = sorted(pred_func2ix.items(), key = lambda x: x[1])
    pred_funcs = []
    for pred_func, ix in sorted_pred_func2ix:
        pred_funcs.append(pred_func)

    print ("Initializing encoder ...")
    # content_preds = set(content_pred2cnt)
    num_embs = len(pred2ix)

    encoder = config.init_obj('encoder_arch', module_arch, num_embs = num_embs)
    # logger.info(decoder)

    print ("Initializing semantic functions ...")
    sem_funcs = []
    for pred_func in pred_funcs:
        if pred_func.endswith("ARG0"):
            sem_func = config.init_obj('one_place_sem_func', module_arch)
        else:
            sem_func = config.init_obj('two_place_sem_func', module_arch)
        sem_funcs.append(sem_func)
    
    print ("Initializing decoder ...")
    decoder = config.init_obj('decoder_arch', module_arch, sem_funcs = sem_funcs, pred_func2cnt = pred_func2cnt, pred_funcs = pred_funcs) 

    # if len(device_ids) > 1:
    #     decoder = torch.nn.DataParallel(decoder, device_ids=device_ids)
        
    # print (torch.cuda.list_gpu_processes(device = device))
    # print (torch.cuda.memory_summary(device = device))

    # print (torch.cuda.list_gpu_processes(device=None))
    # print (torch.cuda.memory_summary(device=None))

    param_size = 0
    buffer_size = 0
    for sem_func_ix in range(len(sem_funcs)):
        for param in sem_funcs[sem_func_ix].parameters():
            # param.requires_grad = False
            param_size += param.nelement() * param.element_size()
        for buffer in sem_funcs[sem_func_ix].buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print ('#embs in encoder: {}'.format(num_embs))
    print ('#sem_funcs: {}'.format(len(sem_funcs)))
    print('agg. sem_funcs size: {:.3f}MB; param: {}MB; buffer: {}MB'.format(size_all_mb, param_size/ 1024**2, buffer_size/ 1024**2))

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
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

    world_size, ddp, cpu, config = ddp_args
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size, cpu)

    if cpu:
        rank = 'cpu'
    trainer_args = get_trainer_args(config)
    trainer = Trainer(world_size = world_size, device = rank, ddp = ddp, cpu = cpu, **trainer_args)
    trainer.train(rank)

    cleanup()


def main(config):

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    # device, device_ids = None, []
    ddp = config['ddp']
    cpu = False
    world_size = None
    if device_ids == []:
        num_thread = torch.get_num_threads()
        # torch.set_num_threads(int(num_thread))
        print ("using {} with {} threads".format(device, num_thread))
        cpu = True
        world_size = int(num_thread)
    else:
        print ("using {} with device_ids: {}".format(device, device_ids))
        if len(device_ids) > 1:
            ddp = True
            world_size = len(device_ids)
        else:
            ddp = False

    if ddp:
        mp.spawn(ddp_trainer,
            args = ((world_size, ddp, cpu, config),),
            nprocs = world_size,
            join = True
        )
    else:
        trainer_args = get_trainer_args(config)
        trainer = Trainer(world_size = world_size, device = device, ddp = ddp, cpu = cpu, **trainer_args)
        trainer.train(-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
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
    main(config)