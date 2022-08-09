import argparse
from collections import namedtuple, Counter
import torch
import numpy as np
from parse_config import ConfigParser

from transform.tcs_transform import TruthConditions

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

from pprint import pprint
import json
import os

from multiprocessing import Pool
from functools import reduce

from pprint import pprint

from src import util

# import networkx as nx

# fix random seeds for reproducibility
SEED = 123

# def _prepare_train_worker_max(args):

#     data_dir, transform_config, worker_id, sample_only = args

#     num_lex_pred2cnt = Counter()
#     for root, dirs, files in os.walk(data_dir):
#         for file in tqdm(files):
#             if not util.is_data_json(file):
#                 continue
#             if not file.startswith("{}_".format(str(worker_id))):
#                 continue
#             with open(os.path.join(data_dir, file)) as f:
#                 idx2instance = json.load(f)
#                 for idx, instance in idx2instance.items():

#                     trsfm = TruthConditions(transform_config)
#                     transformed = trsfm(instance)
#                     # count pred_func
                    
#                     # print (transformed['encoders']['lexical_preds'])
#                     # _count_pred_func(sub_pred_func2cnt, transformed, worker_id)
#                     num_lex_pred2cnt[len(transformed['encoders']['lexical_preds'])] += 1
#                     if sample_only:
#                         break
#                 if sample_only:
#                     break
#             if sample_only:
#                 break
#             # print ("processed {}".format(file))
#             # pprint (sub_pred_func2cnt)

#     return num_lex_pred2cnt


# def prepare_train_max(data_dir, transform_config, sample_only):

#     workers_args = [(data_dir, transform_config, worker_id, sample_only) for worker_id in range(10)]
#     prepared = []
#     with Pool(10) as p:
#         prepared = list(tqdm(p.imap(_prepare_train_worker_max, workers_args)))
    
#     num_lex_pred2cnt = reduce(lambda x, y: x + y, prepared)
    
#     return num_lex_pred2cnt


def _prepare_train_worker(args):

    def _count_pred_func(pred_func2cnt, logic_expr, worker_id):
        if isinstance(logic_expr, dict):
            pred_func2cnt[logic_expr['pred_func']] += 1
        elif logic_expr:
            root, left, right = logic_expr
            _count_pred_func(pred_func2cnt, left, worker_id)
            _count_pred_func(pred_func2cnt, right, worker_id)

    def _count_lex_pred(lex_pred2cnt, lexical_preds, worker_id):
        lex_pred2cnt.update(lexical_preds)

    data_dir, transform_config, worker_id, sample_only = args

    prepared = {"pred_func2cnt": Counter(), "lex_pred2cnt": Counter()}
    for root, dirs, files in os.walk(data_dir):
        for file in tqdm(files):
            if not util.is_data_json(file):
                continue
            if not file.startswith("{}_".format(str(worker_id))):
                continue
            with open(os.path.join(data_dir, file)) as f:
                idx2instance = json.load(f)
                for idx, instance in idx2instance.items():

                    trsfm = TruthConditions(transform_config, None, None, None, None, False)
                    transformed = trsfm(instance)
                    
                    # count pred_func
                    logic_expr = transformed["decoders"]["logic_expr"]
                    _count_pred_func(prepared["pred_func2cnt"], logic_expr, worker_id)
                    lexical_preds = transformed["encoders"]["lexical_preds"]
                    _count_lex_pred(prepared["lex_pred2cnt"], lexical_preds, worker_id)
                    
                    if sample_only:
                        break
                if sample_only:
                    break
            if sample_only:
                break
            # print ("processed {}".format(file))
            # pprint (sub_pred_func2cnt)

    return prepared

def prepare_train(data_dir, transform_config, sample_only):

    workers_args = [(data_dir, transform_config, worker_id, sample_only) for worker_id in range(10)]
    prepared = []
    with Pool(10) as p:
        prepared_workers = list(tqdm(p.imap(_prepare_train_worker, workers_args)))
    
    prepared = reduce(lambda x, y: {key: x[key] + y[key] for key in x}, prepared_workers)
    return prepared



def main(config, sample_only = "no"):

    if sample_only == "no": sample_only = False
    elif sample_only == "yes": sample_only = True

    data_loader_args = config['data_loader']['args']
    data_dir = data_loader_args['data_dir']
    transform_config_file_path = data_loader_args['transform_config_file_path']
    
    with open(transform_config_file_path) as f:
        transform_config = json.load(f)

    # num_lex_pred2cnt = prepare_train_max(data_dir, transform_config, sample_only)
    # print (num_lex_pred2cnt)

    prepared = prepare_train(data_dir, transform_config, sample_only)

    out_dir = config.log_dir
    print (out_dir)
    pred_func2cnt_file_path = os.path.join(out_dir, "pred_func2cnt.txt")
    with open(pred_func2cnt_file_path, "w") as f:
        for pred_func, cnt in prepared["pred_func2cnt"].most_common():
            f.write("{}\t{}\n".format(pred_func, str(cnt)))
    lex_pred2cnt_file_path = os.path.join(out_dir, "lex_pred2cnt.txt")
    with open(lex_pred2cnt_file_path, "w") as f:
        for lex_pred, cnt in prepared["lex_pred2cnt"].most_common():
            f.write("{}\t{}\n".format(lex_pred, str(cnt)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare_train')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    parser.add_argument('-s', '--sampleOnly', default="no", type=str,
                      help='prepare sample instances only')
    args = parser.parse_args()
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = namedtuple('CustomArgs', 'flags type target')
    options = []
    config = ConfigParser.from_args(parser, options)
    main(config, args.sampleOnly)