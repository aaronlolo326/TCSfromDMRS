import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

from pprint import pprint
import os
from collections import Counter

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):

    logger = config.get_logger('train')

    # declare decoders
    MIN_PRED_FUNC_FREQ = config["data_loader"]["args"]["min_pred_func_freq"]
    MIN_LEX_PRED_FREQ = config["data_loader"]["args"]["min_lex_pred_freq"]

    log_dir = "saved/log/TCS/0809_153834"
    # log_dir = config.log_dir
    pred_func2cnt_file_path = os.path.join(log_dir, "pred_func2cnt.txt")
    lex_pred2cnt_file_path = os.path.join(log_dir, "lex_pred2cnt.txt")

    print ("Initializing decoders ...")
    decoders = collections.defaultdict()
    pred_funcs = set()
    pred_func2cnt = Counter()
    with open(pred_func2cnt_file_path) as f:
        line = f.readline()
        while line:
            pred_func, cnt = line.strip().split("\t")
            if int(cnt) < MIN_PRED_FUNC_FREQ:
                break
            decoder_model = None
            if pred_func.endswith("ARG0"):
                decoder_model = config.init_obj('one_place_decoder_arch', module_arch)
                pred_funcs.add(pred_func.split("@")[0])
                pred_func2cnt[pred_func] = int(cnt)
            else:
                decoder_model = config.init_obj('two_place_decoder_arch', module_arch)
            decoders[pred_func] = decoder_model
            line = f.readline()


    print ("Initializing encoder ...")
    lex_pred2cnt = Counter()
    with open(lex_pred2cnt_file_path) as f:
        line = f.readline()
        while line:
            lex_pred, cnt = line.strip().split("\t")
            if int(cnt) < MIN_LEX_PRED_FREQ:
                break
            lex_pred2cnt[lex_pred] = int(cnt)
            line = f.readline()

    lexical_preds = set(lex_pred2cnt)
    emb_preds = pred_funcs.union(lexical_preds)

    pred2ix = {pred: ix + 1 for ix, pred in enumerate(emb_preds)}
    num_embs = len(pred2ix)

    encoder = config.init_obj('encoder_arch', module_arch, num_embs = num_embs)
    
    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data,
        lex_pred2cnt = lex_pred2cnt, pred_func2cnt = pred_func2cnt)
    valid_data_loader = data_loader.split_validation()

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    
    # build model architecture, then print to console
    ## configure snt_id to be included
    ## configure predicate_args from data
    data_info_dir = config.__getitem__('data_info_dir')

    # logger.info(decoder)
    print ("Sending encoder to device ...")
    encoder = encoder.to(device)
    if len(device_ids) > 1:
        encoder = torch.nn.DataParallel(encoder, device_ids=device_ids)

    print (torch.cuda.list_gpu_processes(device = device))
    print (torch.cuda.memory_summary(device = device))

    print ("Sending decoders to device ...")
    for pred_func in decoders:
    # encoder = config.init_obj('encoder_arch', module_arch)
        decoders[pred_func]  = decoders[pred_func].to(device)
        if len(device_ids) > 1:
            decoders[pred_func] = torch.nn.DataParallel(decoders[pred_func], device_ids=device_ids)
    
    print (torch.cuda.list_gpu_processes(device=None))
    print (torch.cuda.memory_summary(device=None))

    param_size = 0
    buffer_size = 0
    for pred_func in decoders:
        for param in decoders[pred_func].parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in decoders[pred_func].buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print ('#embs in encoder: {}'.format(num_embs))
    print ('#decoders: {}'.format(len(decoders)))
    print('agg. decoders size: {:.3f}MB; param: {}MB; buffer: {}MB'.format(size_all_mb, param_size/ 1024**2, buffer_size/ 1024**2))

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    print ("Getting trainable parameters for the decoders ...")
    trainable_params = [{'params': decoders[pregarg].parameters()} for pregarg in decoders] 
    print ("Initializing optimizer for the decoders ...")
    decoder_optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    print ("Initializing learning rate scheduler for the optimizer ...")
    decoder_lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, decoder_optimizer)

    trainer = Trainer(encoder, pred2ix, decoders, criterion, metrics, decoder_optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=decoder_lr_scheduler)

    trainer.train()


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