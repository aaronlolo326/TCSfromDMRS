from scipy import sparse
import argparse
from collections import namedtuple, Counter, defaultdict
import torch
import numpy as np
from parse_config import ConfigParser

from transform.tcs_transform import TruthConditions, schema

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

from pprint import pprint
import json
import jsonschema
import os
import time
import re

from multiprocessing import Pool
from functools import reduce
import itertools

from pprint import pprint

from src import util, dg_util
from utils import get_transformed_info, draw_logic_expr

import networkx as nx
from networkx.readwrite.json_graph import node_link_data
from networkx.drawing.nx_agraph import to_agraph

# import networkx as nx

NUM_WORKERS = 12

trsfm_key2abbrev = {
    "node2pred": "i",
    "pred_func_nodes": "n",
    "content_preds": "l",
    "logic_expr": "e",
    "pred_func_used": "f"
}

# def _prepare_train_worker_max(args):

#     data_dir, transform_config, worker_id, sample_only = args

#     num_content_pred2cnt = Counter()
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
                    
#                     # print (transformed['encoders']['content_preds'])
#                     # _count_pred_func(sub_pred_func2cnt, transformed, worker_id)
#                     num_content_pred2cnt[len(transformed['encoders']['content_preds'])] += 1
#                     if sample_only:
#                         break
#                 if sample_only:
#                     break
#             if sample_only:
#                 break
#             # print ("processed {}".format(file))
#             # pprint (sub_pred_func2cnt)

#     return num_content_pred2cnt


# def prepare_train_max(data_dir, transform_config, sample_only):

#     workers_args = [(data_dir, transform_config, worker_id, sample_only) for worker_id in range(10)]
#     prepared = []
#     with Pool(10) as p:
#         prepared = list(tqdm(p.imap(_prepare_train_worker_max, workers_args)))
    
#     num_content_pred2cnt = reduce(lambda x, y: x + y, prepared)
    
#     return num_content_pred2cnt

def _prepare_relpron_train_worker(args):

    transformed_file_path = None
    written = False

    data_dir, fig_dir, transformed_dir, transform_config, config, worker_id, draw_tree, print_pred_func, sample_only, q_snt_id, to_ix, data_loader_args, as_json = args
    if transformed_dir:
        transformed_file_path = os.path.join(transformed_dir, "transformed_{}.json".format(worker_id))

    min_pred_func_freq, min_content_pred_freq, filter_min_freq = None, None, False
    if data_loader_args:
        min_pred_func_freq, min_content_pred_freq, filter_min_freq = data_loader_args

    prepared = {"pred_func2cnt": Counter(), "pred2cnt": Counter(), "content_pred2cnt": Counter(), "err2cnt": Counter()}

    relpron_path = {
        "dev": os.path.join(data_dir, "relpron.dev"),
        "test": os.path.join(data_dir, "relpron.test")
    }



    for split, path in relpron_path.items():

        if split not in config['eval_relpron_dataloader']['args']['split']:
            continue

        with open(path) as f:
            line = f.readline().strip()
            while line:
                m = re.match('^([OS]BJ) (\S+)_N: (\S+)_N that (\S+)_[VN] (\S+)_[VN]\s*$', line)
                instance = tuple(m.group(i) for i in range(1,6))
                sub_obj, targ = instance[0], instance[1]
                if sub_obj == 'SBJ':
                    sbj, verb, obj = instance[2:5]
                elif sub_obj == 'OBJ':
                    obj, sbj, verb = instance[2:5]
                sbj = "_" + sbj + "_n_" + "0"
                obj = "_" + obj + "_n_" + "0"
                verb = "_" + verb + "_v_" + "0"
                targ = "_" + targ + "_n_" + "0"
                prepared["pred_func2cnt"] += Counter([
                    sbj + "@ARG0",
                    verb + "@ARG0",
                    obj + "@ARG0",
                    verb + "@ARG1",
                    verb + "@ARG2",
                    targ + "@ARG0"
                ])
                prepared["pred2cnt"] += Counter([
                    sbj,
                    verb,
                    obj,
                    targ
                ])
                line = f.readline().strip()

        prepared["content_pred2cnt"] = prepared["pred2cnt"].copy()

        transformed_dir  = config['data_loader']['args']["transformed_dir"] 
        transformed_info_dir = os.path.join(transformed_dir, "info")
        os.makedirs(transformed_info_dir, exist_ok = True)

        pred_func2ix = defaultdict()
        ix = 0
        for pred_func, cnt in prepared["pred_func2cnt"].most_common():
            if cnt >= min_pred_func_freq:
                if not pred_func.endswith("@ARG0"):
                    if not prepared["pred_func2cnt"][pred_func.rsplit("@", 1)[0] + "@ARG0"] >= min_pred_func_freq:
                        continue
                pred_func2ix[pred_func] = ix
                ix += 1

        pred_func2ix_file_path = os.path.join(transformed_info_dir, "pred_func2ix.txt")
        with open(pred_func2ix_file_path, "w") as f:
            for pred_func, ix in pred_func2ix.items():
                f.write("{}\t{}\n".format(str(ix), pred_func))

        pred_func2cnt_file_path = os.path.join(transformed_info_dir, "pred_func2cnt.txt")
        with open(pred_func2cnt_file_path, "w") as f:
            for pred_func, cnt in prepared["pred_func2cnt"].most_common():
                if pred_func in pred_func2ix:
                    f.write("{}\t{}\n".format(pred_func2ix[pred_func], str(cnt)))

        preds = set()
        for content_pred, cnt in prepared["content_pred2cnt"].most_common():
            if cnt >= min_content_pred_freq:
                preds.add(content_pred)
        for pred_func in pred_func2ix:
            pred = pred_func.rsplit("@", 1)[0]
            preds.add(pred)
        
        sorted_preds = sorted(list(preds))
        print ("saving pred2ix of {} predicates ...".format(len(sorted_preds)))
        pred2ix = defaultdict()
        pred2ix_path = os.path.join(transformed_info_dir, "pred2ix.txt")
        with open(pred2ix_path, "w") as f:
            for ix, pred in enumerate(sorted_preds):
                pred2ix[pred] = ix
                f.write("{}\t{}\n".format(ix, pred))

        content_pred2cnt_file_path = os.path.join(transformed_info_dir, "content_pred2cnt.txt")
        with open(content_pred2cnt_file_path, "w") as f:
            for content_pred, cnt in prepared["content_pred2cnt"].most_common():
                if cnt >= min_content_pred_freq:
                    f.write("{}\t{}\n".format(pred2ix[content_pred], str(cnt)))

        print ("prepared transformed data info saved at: {}".format(transformed_info_dir))

        errd2cnt_file_path = os.path.join(transformed_info_dir, "err2cnt.txt")
        with open(errd2cnt_file_path, "w") as f:
            for err, cnt in prepared["err2cnt"].most_common():
                f.write("{}\t{}\n".format(err, str(cnt)))


        snt_id = -1
        written = False
        transformed_keys = config['transformed_keys']
        with open(path) as f:
            line = f.readline().strip()
            while line:
                m = re.match('^([OS]BJ) (\S+)_N: (\S+)_N that (\S+)_[VN] (\S+)_[VN]\s*$', line)
                line = f.readline().strip()
                
                instance = tuple(m.group(i) for i in range(1,6))
                sub_obj, targ = instance[0], instance[1]
                if sub_obj == 'SBJ':
                    sbj, verb, obj = instance[2:5]
                elif sub_obj == 'OBJ':
                    obj, sbj, verb = instance[2:5]
                sbj = "_" + sbj + "_n_" + "0"
                obj = "_" + obj + "_n_" + "0"
                verb = "_" + verb + "_v_" + "0"
                targ = "_" + targ + "_n_" + "0"
                
                snt_id += 1
                trsfm = TruthConditions(
                   *[None] * 9
                )
                if not config['with_logic']:
                    pred_funcs = [
                        pred_func2ix[sbj + "@ARG0"],
                        pred_func2ix[verb + "@ARG0"],
                        pred_func2ix[obj + "@ARG0"],
                        pred_func2ix[verb + "@ARG1"],
                        pred_func2ix[verb + "@ARG2"],
                        pred_func2ix[targ + "@ARG0"]
                    ]
                    vars_unzipped = [
                        [1, 2, 3, 2, 2, 1 if sub_obj == 'SBJ' else 3],
                        [0, 0, 0, 1, 3, 0]
                    ]
                    # transformed_keys = ["node2pred", "pred_func_nodes", "content_preds", "logic_expr", "pred_func_used"]

                    ## Devices that physicists use are telescopes.
                    # node2pred = {
                    #     0: pred2ix[sbj],
                    #     1: pred2ix[verb],
                    #     2: pred2ix[obj],
                    #     3: pred2ix[targ]
                    # }
                    # content_preds_ix = [
                    #     pred2ix[sbj],
                    #     pred2ix[verb],
                    #     pred2ix[obj]
                    # ]

                    # Telescopes are devices that physicists use.
                    if sub_obj == 'SBJ':
                        node2pred = {
                            0: pred2ix[targ],
                            1: pred2ix[verb],
                            2: pred2ix[obj],
                            3: pred2ix[sbj]
                        }
                    elif sub_obj == 'OBJ':
                        node2pred = {
                            0: pred2ix[sbj],
                            1: pred2ix[verb],
                            2: pred2ix[targ],
                            3: pred2ix[obj]
                        }
                    content_preds_ix = [
                        pred2ix[sbj],
                        pred2ix[verb],
                        pred2ix[obj],
                        pred2ix[targ]
                    ]

                    pred_func_nodes = [0, 1, 2]

                    arg2ix = trsfm.arg2ix
                    if sub_obj == 'SBJ':
                        pred_func_nodes_ctxt_preds = [
                            [pred2ix[sbj], pred2ix[targ], pred2ix[verb], pred2ix[obj]],
                            [pred2ix[verb], pred2ix[sbj], pred2ix[targ], pred2ix[obj]],
                            [pred2ix[obj], pred2ix[sbj], pred2ix[targ], pred2ix[verb]]
                        ]
                        pred_func_nodes_ctxt_args = [
                            [arg2ix["ARG0"], arg2ix["ARG0"], arg2ix["ARG1"], 0],
                            [arg2ix["ARG0"], arg2ix["ARG1-rvrs"], arg2ix["ARG1-rvrs"], arg2ix["ARG2-rvrs"]],
                            [arg2ix["ARG0"], 0, 0, arg2ix["ARG2"]],
                        ]
                    if sub_obj == 'OBJ':
                        pred_func_nodes_ctxt_preds = [
                            [pred2ix[sbj], pred2ix[verb], pred2ix[obj], pred2ix[targ]],
                            [pred2ix[verb], pred2ix[sbj], pred2ix[obj], pred2ix[targ]],
                            [pred2ix[obj], pred2ix[targ], pred2ix[sbj], pred2ix[verb]]
                        ]
                        pred_func_nodes_ctxt_args = [
                            [arg2ix["ARG0"], arg2ix["ARG1"], 0, 0],
                            [arg2ix["ARG0"], arg2ix["ARG1-rvrs"], arg2ix["ARG2-rvrs"], arg2ix["ARG2-rvrs"]],
                            [arg2ix["ARG0"], arg2ix["ARG0"], 0, arg2ix["ARG2"]],
                        ]


                    transformed = {
                        "discarded": None,
                        "discarded_reason": None,
                        "node2pred": node2pred,
                        "pred_func_nodes": pred_func_nodes,
                        "pred_func_nodes_ctxt_preds": pred_func_nodes_ctxt_preds,
                        "pred_func_nodes_ctxt_args": pred_func_nodes_ctxt_args,
                        "content_preds": content_preds_ix,
                        "logic_expr": pred_funcs,
                        "pred_func_used": pred_funcs
                    }

                    transformed_json = [transformed[key] if not key == 'logic_expr' else pred_funcs for key in transformed_keys]
                            # pprint (transformed_json)
                    transformed_json += [vars_unzipped]

                    if not written:
                        delimiter = "["
                        write_mode = 'w'
                        written = True
                    else:
                        delimiter = ", "
                        write_mode = 'a'
                    
                    if as_json:
                        with open(transformed_file_path, write_mode) as wf:
                            wf.write(delimiter)
                            wf.write(json.dumps(transformed_json))
                        
        if as_json:
            with open(transformed_file_path, "a") as wf:
                wf.write("]")

    return {"prepared": prepared}

    
def _prepare_train_worker(args):

    def _get_vars(logic_expr, pred_func_nodes2ix, worker_id):
        if len(logic_expr) == 2 and all([isinstance(logic_expr[1][i], int) for i in range(len(logic_expr[1]))]):
            sem_func_ix, args = logic_expr
            args_ix = [pred_func_nodes2ix[arg] for arg in args]
            if len(args) == 1:
                args_ix = [args_ix[0], 0]
            return [args_ix]
        elif logic_expr:
            root, *dgtrs = logic_expr
            return list(itertools.chain(*[_get_vars(dgtr, pred_func_nodes2ix, worker_id) for dgtr in dgtrs]))

    def _get_pred_funcs(logic_expr, worker_id):
        if len(logic_expr) == 2 and all([isinstance(logic_expr[1][i], int) for i in range(len(logic_expr[1]))]):
            sem_func_ix, args = logic_expr
            return [sem_func_ix]
        elif logic_expr:
            root, *dgtrs = logic_expr
            return list(itertools.chain(*[_get_pred_funcs(dgtr, worker_id) for dgtr in dgtrs]))

    def _get_pred_func_vars(logic_expr, pred_func_nodes2ix, worker_id):
        if len(logic_expr) == 2 and all([isinstance(logic_expr[1][i], int) for i in range(len(logic_expr[1]))]):
            sem_func_ix, args = logic_expr
            args_ix = tuple(pred_func_nodes2ix[arg] for arg in args)
            if len(args) == 1:
                args_ix = (args_ix[0], 0)
            return ((sem_func_ix, args_ix),)
        elif logic_expr:
            root, *dgtrs = logic_expr
            return list(itertools.chain(*[_get_pred_func_vars(dgtr, pred_func_nodes2ix, worker_id) for dgtr in dgtrs]))


    def _count_pred_func(pred_func2cnt, pred2cnt, logic_expr, worker_id):
        if isinstance(logic_expr, dict):
            
            pred_func2cnt[logic_expr['pf']] += 1
            if isinstance(logic_expr['pf'], str):
                pred2cnt[logic_expr['pf'].rsplit("@", 1)[0]] += 1
            else:
                print ("not string instance?")
                # transformed
                pass
        elif logic_expr:
            root, *dgtrs = logic_expr
            for dgtr in dgtrs:
                _count_pred_func(pred_func2cnt, pred2cnt, dgtr, worker_id)

    def _count_content_pred(content_pred2cnt, content_preds, worker_id):
        content_pred2cnt.update(content_preds)

    def _count_content_predarg(content_predarg2cnt, pred_func_nodes_ctxt_pred_args, worker_id):
        for pred_args in pred_func_nodes_ctxt_pred_args:
            # print (pred_args)
            content_predarg2cnt.update(pred_args)

    def validate_json(f):
        try:
            # Read in the JSON document
            data = json.load(f)
            # And validate the result
            jsonschema.validate(data, schema)
        except jsonschema.exceptions.ValidationError as e:
            print("well-formed but invalid JSON:", e)
            return False
        except json.decoder.JSONDecodeError as e:
            print("poorly-formed text, not JSON:", e)
            return False
        return True

    # id2transformed = defaultdict(defaultdict)

    transformed_file_path = None
    written = False

    data_dir, fig_dir, transformed_dir, transform_config, config, worker_id, draw_tree, print_pred_funcs, sample_only, q_snt_id, to_ix, data_loader_args, as_json, relpron_preds = args
    if transformed_dir:
        transformed_file_path = os.path.join(transformed_dir, "_transformed_{}.json".format(worker_id))

    min_pred_func_freq, min_content_pred_freq, filter_min_freq, content_pred2cnt, pred_func2cnt, pred2ix, predarg2ix, pred_func2ix = None, None, False, None, None, None, None, None
    if data_loader_args:
        min_pred_func_freq, min_content_pred_freq, filter_min_freq, content_pred2cnt, pred_func2cnt, pred2ix, predarg2ix, pred_func2ix = data_loader_args

    prepared = {"pred_func2cnt": Counter(), "pred2cnt": Counter(), "content_pred2cnt": Counter(), "content_predarg2cnt": Counter(), "err2cnt": Counter()}
    corpus_stats = {"num_tokens": 0, "num_surf_preds": 0, "num_content_preds": 0, "num_pred_func_nodes": 0}

    found_q = False
    worker_files = [file for file in os.listdir(data_dir) if all([
        os.path.isfile(os.path.join(data_dir, file)),
        # file.startswith("{}_".format(str(worker_id))),
        sum(map(lambda x: int(x),filter(lambda x: str.isnumeric(x), file))) % NUM_WORKERS == worker_id,
        util.is_data_json(file)
        ])
    ]
    no_files = len(worker_files)
    
    transformed_keys = config["transformed_keys"]

    # if print_pred_funcs:
    if pred2ix:
        ix2pred = {ix: pred for pred, ix in pred2ix.items()}
    if pred_func2ix:
        ix2pred_func = {ix: pred_func for pred_func, ix in pred_func2ix.items()}

    for file_idx, file in enumerate(worker_files):
        if not sample_only and int((no_files/100)) != 0 and file_idx%int((no_files/100)) == 0:
            print ("worker {}: {:.2f}% done".format(worker_id, file_idx/no_files * 100))
        # if not file.startswith("{}_".format(str(worker_id))):
        #     continue
        with open(os.path.join(data_dir, file)) as f:
            idx2instance = json.load(f)
            for idx_loop, (idx, instance) in enumerate(idx2instance.items()):
                snt_id = instance['id']
                if q_snt_id != None and q_snt_id != snt_id:
                    continue
                found_q = True
                trsfm = TruthConditions(
                    transform_config, to_ix, min_pred_func_freq, min_content_pred_freq, content_pred2cnt, pred_func2cnt, filter_min_freq,
                    pred2ix, predarg2ix, pred_func2ix, relpron_preds
                )
                transformed = trsfm(instance)
                # transformed = {
                #     "discarded": self.discarded,
                #     "discarded_reason": self.discarded_reason,
                #     "node2pred": self.node2pred_ix,
                #     "pred_func_nodes": self.pred_func_nodes,
                #     "pred_func_nodes_ctxt_preds": self.pred_func_nodes_ctxt_preds,
                #     "pred_func_nodes_ctxt_args": self.pred_func_nodes_ctxt_preds,
                #     "content_preds": self.content_preds,
                #     "logic_expr": self.logic_expr,
                #     "pred_func_used": list(self.pred_func_used)
                # }
                discarded = transformed["discarded"]
                if discarded:
                    prepared["err2cnt"][transformed["discarded_reason"]] += 1
                    if transformed["discarded_reason"] == "not all pred_func_nodes_ctxt_args have same length":
                        pass
                        # print (1, transformed)
                    # if "Recursion" in transformed["discarded_reason"]:
                    # print (snt_id, snt_id, transformed["discarded_reason"])
                else:
                    # count pred_func
                    if not transformed_file_path:
                        _count_pred_func(prepared["pred_func2cnt"], prepared["pred2cnt"], transformed["logic_expr"], worker_id)
                        _count_content_pred(prepared["content_pred2cnt"], transformed["content_preds"], worker_id)
                        _count_content_predarg(prepared["content_predarg2cnt"], transformed["pred_func_nodes_ctxt_pred_args"], worker_id)
                    
                    corpus_stats["num_tokens"] += transformed["num_tokens"]
                    corpus_stats["num_surf_preds"] += transformed["num_surf_preds"]
                    corpus_stats["num_content_preds"] += len(transformed["content_preds"])
                    corpus_stats["num_pred_func_nodes"] += len(transformed["pred_func_nodes"])

                    if not q_snt_id and transformed_file_path:
                        pred_funcs, vars = None, None
                        if not config['with_logic']:
                            pred_func_nodes2ix = {node: ix + 1 for ix, node in enumerate(transformed["pred_func_nodes"])}
                            pred_funcs_vars = _get_pred_func_vars(transformed["logic_expr"], pred_func_nodes2ix, worker_id)
                            pred_funcs_vars = set(pred_funcs_vars)
                            pred_funcs, vars = list(zip(*pred_funcs_vars))
                            vars_unzipped = list(zip(*vars))
                            if not config['decoder_arch']['args']['use_truth']:
                                pred_func_nodes_ctxt_args = transformed['pred_func_nodes_ctxt_args']
                                assert len(transformed['pred_func_nodes_ctxt_preds'][0]) == len(pred_func_nodes_ctxt_args[0])
                                pred_func_nodes_ctxt_args_coo = sparse.csr_matrix(np.array(pred_func_nodes_ctxt_args)).tocoo()
                                pred_func_nodes_ctxt_args_coo_list = [
                                    pred_func_nodes_ctxt_args_coo.data.tolist(),
                                    pred_func_nodes_ctxt_args_coo.col.tolist(),
                                    pred_func_nodes_ctxt_args_coo.row.tolist()
                                ]
                                preds_ix = set()
                                pred_ix2vars = defaultdict(lambda: [])
                                pred_ix2pred_funcs_ix = defaultdict(lambda: [])
                                pred_ix2args_num_sum =  defaultdict(lambda: 0)
                                for pred_func_num, pred_func_ix in enumerate(pred_funcs):
                                    pred_func = ix2pred_func[pred_func_ix]
                                    pred, arg = pred_func.rsplit("@", 1)
                                    arg_num = int(arg[-1])
                                    assert pred in pred2ix
                                    preds_ix.add(pred2ix[pred])
                                    pred_ix2vars[pred2ix[pred]].append(vars[pred_func_num])
                                    pred_ix2pred_funcs_ix[pred2ix[pred]].append(pred_func_ix)
                                for _, pred_func_ix in enumerate(set(pred_funcs)):
                                    pred_func = ix2pred_func[pred_func_ix]
                                    pred, arg = pred_func.rsplit("@", 1)
                                    arg_num = int(arg[-1])
                                    assert pred in pred2ix
                                    pred_ix2args_num_sum[pred2ix[pred]] += (2 ** arg_num)
                                # preds_ix = list(preds_ix)
                                pred_funcs_ix_list = [pred_ix2pred_funcs_ix[pred_ix] for pred_ix in pred_ix2pred_funcs_ix]
                                vars_list = [pred_ix2vars[pred_ix] for pred_ix in pred_ix2pred_funcs_ix]
                                args_num_sum_list = [pred_ix2args_num_sum[pred_ix] for pred_ix in pred_ix2pred_funcs_ix]
                                transformed_json = []
                                for key in transformed_keys:
                                    if key == 'logic_expr':
                                        transformed_data = pred_funcs_ix_list
                                    elif key == 'pred_func_nodes_ctxt_args':
                                        transformed_data = pred_func_nodes_ctxt_args_coo_list
                                    else:
                                        transformed_data = transformed[key]
                                    transformed_json.append(transformed_data)
                                transformed_json += [vars_list, args_num_sum_list]
                            # pred_funcs = _get_pred_funcs(transformed["logic_expr"], worker_id)
                            # vars = _get_vars(transformed["logic_expr"], pred_func_nodes2ix, worker_id)
                            # assert len(pred_funcs) == len(vars)
                            # cleanse so that p(x,y) appears for once only
                            # pf_xy = set()
                            # for pf_idx, pf in enumerate(pred_funcs):
                            #     pf_xy.add(1)
                            # transform vars from (num_pairs, 2) to (2, num_pairs) for easy pytorch processing
                            else:
                                transformed_json = [transformed[key] if not key == 'logic_expr' else pred_funcs for key in transformed_keys]
                                # pprint (transformed_json)
                                transformed_json += [vars_unzipped]
                            if print_pred_funcs:
                                padded_pred_func_nodes = [-1] + transformed["pred_func_nodes"]
                                vars_pred = [
                                    [
                                        ix2pred[transformed["node2pred"][padded_pred_func_nodes[v]]] if v != 0
                                        else -1
                                        for v in v_pair
                                    ]
                                    for v_pair in vars
                                ]
                                pred_funcs_name = [ix2pred_func[pred_func_ix] for pred_func_ix in pred_funcs]
                                assert len(pred_funcs_name) == len(vars_pred)
                                pred_funcs_vars_pred = list(zip(pred_funcs_name, vars_pred))
                                pprint ([
                                    [
                                    (pred_func, var_pair)
                                    for pred_func, var_pair in pred_funcs_vars_pred
                                    if not (pred_func.split("@")[1] == "ARG0" and pred_func.split("@")[0] == var_pair[0])
                                    ],
                                    [
                                        ix2pred[content_pred_ix] for content_pred_ix in transformed["content_preds"]
                                    ]
                                ])
                                print ()

                        else:
                            transformed_json = [transformed[key] for key in transformed_keys]
                        
                        if as_json:
                            if not written:
                                delimiter = "["
                                write_mode = 'w'
                                written = True
                            else:
                                delimiter = ", "
                                write_mode = 'a'
                            with open(transformed_file_path, write_mode) as f:
                                f.write(delimiter)
                                f.write(json.dumps(transformed_json))
                if draw_tree:
                    snt = instance['snt']
                    logic_expr_save_path = os.path.join(fig_dir, "logic_expr_{}.png".format(snt_id))
                    print (snt)
                    draw_logic_expr(transformed["logic_expr"], save_path = logic_expr_save_path)
                    erg_digraphs = dg_util.Erg_DiGraphs()
                    dmrs_nxDG = nx.node_link_graph(instance['dmrs'])
                    erg_digraphs.init_dmrs_from_nxDG(dmrs_nxDG)
                    erg_digraphs.init_snt(snt)

                    dmrs_save_path = os.path.join(fig_dir, "dmrs_{}.png".format(snt_id))#+ time.asctime( time.localtime(time.time()) ).replace(" ", "-") +".png"
                    erg_digraphs.draw_dmrs(save_path = dmrs_save_path)
                if q_snt_id and found_q or sample_only and idx_loop >= 2000:
                    break
            if q_snt_id and found_q or sample_only:
                break
        if q_snt_id and found_q or sample_only:
            break
        # print ("processed {}".format(file))
        # pprint (sub_pred_func2cnt)

    if as_json:
        if not q_snt_id and transformed_file_path and written:
            with open(transformed_file_path, "a") as f:
                f.write("]")
        # with open(transformed_file_path, "r") as f:
        #     valid_json = validate_json(f)
            # if not valid_json:
            #     print ("Invalid json.")
            # else:
            #     print ("Valid json.")
    return {"prepared": prepared, "corpus_stats": corpus_stats}

def prepare_train(data_dir, fig_dir, transformed_dir, transform_config, config, draw_tree, print_pred_funcs, sample_only, q_snt_id, to_ix, data_loader_args, as_json, relpron_preds):

    workers_args = [(data_dir, fig_dir, transformed_dir, transform_config, config, worker_id, draw_tree, print_pred_funcs, sample_only, q_snt_id, to_ix, data_loader_args, as_json, relpron_preds) for worker_id in range(NUM_WORKERS)]
    prepared = []
    if "relpron" in config["name"]:
        worker_func = _prepare_relpron_train_worker
    else:
        worker_func = _prepare_train_worker
    with Pool(NUM_WORKERS) as p:
        prepared_workers = list(p.imap(worker_func, workers_args))
    
    print ("Pool ended. Reducing ...")
    # prepared = reduce(lambda x, y: {key: x[0][key] + y[0][key] for key in x}, prepared_workers)
    id2transformed_list = []
    prepared = defaultdict(Counter)
    corpus_stats = defaultdict(int)
    for prepared_worker in prepared_workers:
        # id2transformed_list.append(prepared_worker["id2transformed"])
        for key, data in prepared_worker['prepared'].items():
            # if key != "content_predargs":
            #     if key not in prepared:
            #         prepared[key] = Counter()
            #     prepared[key] = prepared[key] + data
            # else:
            #     if key not in prepared:
            #         prepared[key] = set()
            prepared[key] = prepared[key] + data
        for key, data in prepared_worker['corpus_stats'].items():
            corpus_stats[key] = corpus_stats[key] + data

    return prepared, corpus_stats

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def write_balanced(args):
    transformed_dir, worker_id, bal_transformed = args
    print (worker_id, len(bal_transformed))
    print ("Saving {}".format(worker_id))
    with open(os.path.join(transformed_dir, "transformed_{}.json".format(worker_id)), "w") as f:
        json.dump(bal_transformed, f)
    return True

def load_transformed(transformed_path):
    transformed = []
    if os.path.exists(transformed_path):
        print ("loading {}".format(transformed_path))
        with open(transformed_path) as f:
            transformed = json.load(f)
        print ("Finished loading {}".format(transformed_path))
    return transformed

def balance_splits(transformed_dir, as_json):

    num_instance = 0

    if as_json:
        transformed_list = []
        transformed_paths =  [os.path.join(transformed_dir, "_transformed_{}.json".format(i)) for i in range(NUM_WORKERS)]

        workers_args = [transformed_paths[worker_id] for worker_id in range(NUM_WORKERS)]
        with Pool(NUM_WORKERS) as p:
            for idx, transformed in enumerate(p.imap(load_transformed, workers_args)):
                if transformed != []:
                    print ("Extending {}".format(idx))
                    transformed_list.extend(transformed)
                    del transformed

        num_instance = len(transformed_list)

        num_instance_per_file = max(1, int(num_instance/NUM_WORKERS))
        print ("total num_instance", num_instance)
        print ("num_instance_per_file", num_instance_per_file)
        for idx, chunk in enumerate(chunks(transformed_list, num_instance_per_file)):
            # transformed_dir, worker_id, bal_transformed = args
            print (idx, len(chunk))
            if len(chunk) == num_instance_per_file:
                print ("Saving {}".format(idx))
                with open(os.path.join(transformed_dir, "transformed_{}.json".format(idx)), "w") as f:
                    json.dump(chunk, f)
            del chunk

    # workers_args = [(transformed_dir, worker_id, bal_transformed_list[worker_id]) for worker_id in range(NUM_WORKERS)]
    # with Pool(NUM_WORKERS) as p:
    #     return_workers = list(p.imap(write_balanced, workers_args))
    
    # if all(return_workers):
    #     print ("All saved. Pool ended.")
    # prepared = reduce(lambda x, y: {key: x[0][key] + y[0][key] for key in x}, prepared_workers)


def main(config, draw_tree = "no", print_pred_funcs = "no", q_snt_id = None, as_json = True, relpron_dir = None):

    mode = None
    global NUM_WORKERS
    if config["name"].endswith("_dummy"):
        mode = "dummy"
    elif "relpron" in config["name"]:
        mode = "relpron"
    if mode in ["dummy", "relpron"]:
        NUM_WORKERS = 1

    sample_only = config["sample_only"]
    data_loader_args = config['data_loader']['args']
    data_dir = data_loader_args['data_dir']
    min_pred_func_freq = data_loader_args["min_pred_func_freq"]
    min_content_pred_freq = data_loader_args["min_content_pred_freq"]
    filter_min_freq = data_loader_args["filter_min_freq"]
    transform_config_file_path = data_loader_args['transform_config_file_path']

    relpron_word2pred = None
    with open(os.path.join(relpron_dir, "word2pred_premap.json")) as f:
        relpron_word2pred = json.load(f)
    relpron_preds = set(relpron_word2pred.values())

    transformed_dir  = data_loader_args["transformed_dir"]
    transformed_info_dir = os.path.join(transformed_dir, "info")
    os.makedirs(transformed_info_dir, exist_ok = True)

    # sample_str = None
    # if sample_only == "no":
    #     sample_only = False
    #     sample_str = ""
    # elif sample_only == "yes":
    #     sample_only = True
    #     sample_str = "sample"
    if q_snt_id != None: sample_only = False
    if draw_tree == "no": draw_tree = False
    elif draw_tree == "yes": draw_tree = True
    if print_pred_funcs == "no": print_pred_funcs = False
    elif print_pred_funcs == "yes": print_pred_funcs = True
    if as_json == "no": as_json = False
    if as_json == "yes": as_json = True

    with open(transform_config_file_path) as f:
        transform_config = json.load(f)

    fig_dir = None
    if draw_tree:
        fig_dir = os.path.join(data_dir, "figures")
        os.makedirs(fig_dir, exist_ok = True)

    if mode == 'relpron':
        to_ix = False
        data_loader_args = (min_pred_func_freq, min_content_pred_freq, filter_min_freq)
        prepared, corpus_stats = prepare_train(
            data_dir, fig_dir, transformed_dir, transform_config, config, draw_tree, print_pred_funcs, sample_only, q_snt_id, to_ix, data_loader_args, as_json, relpron_preds
        )
    else:
        to_ix = False
        prepared, corpus_stats = prepare_train(data_dir, fig_dir, None, transform_config, config, draw_tree, False, sample_only, q_snt_id, to_ix, None, as_json, relpron_preds)

        run_dir = config.run_dir
        log_dir = config.log_dir

        pred_func2ix = defaultdict()
        ix = 0
        for pred_func, cnt in prepared["pred_func2cnt"].most_common():
            if cnt >= min_pred_func_freq or pred_func.rsplit("@")[0] in relpron_preds:
                if not pred_func.endswith("@ARG0"):
                    if not prepared["pred_func2cnt"][pred_func.rsplit("@", 1)[0] + "@ARG0"] >= min_pred_func_freq:
                        continue
                pred_func2ix[pred_func] = ix
                ix += 1

        pred_func2ix_file_path = os.path.join(transformed_info_dir, "pred_func2ix.txt")
        with open(pred_func2ix_file_path, "w") as f:
            for pred_func, ix in pred_func2ix.items():
                f.write("{}\t{}\n".format(str(ix), pred_func))

        pred_func2cnt_file_path = os.path.join(transformed_info_dir, "pred_func2cnt.txt")
        with open(pred_func2cnt_file_path, "w") as f:
            for pred_func, cnt in prepared["pred_func2cnt"].most_common():
                if pred_func in pred_func2ix:
                # if cnt >= min_pred_func_freq:
                    f.write("{}\t{}\n".format(pred_func2ix[pred_func], str(cnt)))

        preds = set()
        for content_pred, cnt in prepared["content_pred2cnt"].most_common():
            if cnt >= min_content_pred_freq or content_pred in relpron_preds:
                preds.add(content_pred)
        for pred_func in pred_func2ix:
            pred = pred_func.rsplit("@", 1)[0]
            preds.add(pred)
        sorted_preds = sorted(list(preds))
        print ("saving pred2ix of {} predicates ...".format(len(sorted_preds)))
        pred2ix = defaultdict()
        pred2ix_path = os.path.join(transformed_info_dir, "pred2ix.txt")
        with open(pred2ix_path, "w") as f:
            for ix, pred in enumerate(sorted_preds):
                pred2ix[pred] = ix
                f.write("{}\t{}\n".format(ix, pred))

        content_pred2cnt_file_path = os.path.join(transformed_info_dir, "content_pred2cnt.txt")
        with open(content_pred2cnt_file_path, "w") as f:
            for content_pred, cnt in prepared["content_pred2cnt"].most_common():
                if cnt >= min_content_pred_freq or content_pred in relpron_preds:
                    f.write("{}\t{}\n".format(pred2ix[content_pred], str(cnt)))

        # For PASEncoder
        # sorted_predargs = sorted(list(prepared["content_predarg2cnt"].keys()))
        print ("saving predarg2ix of {} predicates ...".format(len(sorted_preds)))

        predarg2ix = defaultdict()
        ix = 0
        for _, (predarg, cnt) in enumerate(prepared["content_predarg2cnt"].most_common()):
            if cnt >= min_pred_func_freq and ix < 100000 or predarg.rsplit("@")[0] in relpron_preds:
                predarg2ix[predarg] = ix
                ix += 1
                
        content_predarg2ix_file_path = os.path.join(transformed_info_dir, "content_predarg2ix.txt")
        with open(content_predarg2ix_file_path, "w") as f:
            for predarg, ix in predarg2ix.items():
                f.write("{}\t{}\n".format(str(ix), predarg))

        content_predarg2cnt_file_path = os.path.join(transformed_info_dir, "content_predarg2cnt.txt")
        with open(content_predarg2cnt_file_path, "w") as f:
            for predarg, cnt in prepared["content_predarg2cnt"].most_common():
                if predarg in predarg2ix:
                # if cnt >= min_pred_func_freq:
                    f.write("{}\t{}\n".format(predarg2ix[predarg], str(cnt)))
        
                    

        print ("prepared transformed data info saved at: {}".format(transformed_info_dir))

        errd2cnt_file_path = os.path.join(transformed_info_dir, "err2cnt.txt")
        with open(errd2cnt_file_path, "w") as f:
            for err, cnt in prepared["err2cnt"].most_common():
                f.write("{}\t{}\n".format(err, str(cnt)))

        corpus_stats_file_path = os.path.join(transformed_info_dir, "corpus_stats.txt")
        with open(corpus_stats_file_path, "w") as f:
            f.write("num_tokens\t" + str(corpus_stats["num_tokens"]) + "\n")
            f.write("num_surf_preds\t" + str(corpus_stats["num_surf_preds"]) + "\n")
            f.write("num_content_preds\t" + str(corpus_stats["num_content_preds"]) + "\n")
            f.write("num_pred_func_nodes\t" + str(corpus_stats["num_pred_func_nodes"]) + "\n")


        
        
        pred_func2cnt, content_pred2cnt, pred2ix, predarg2ix, pred_func2ix = get_transformed_info(transformed_info_dir)

        data_loader_args = (min_pred_func_freq, min_content_pred_freq, filter_min_freq, content_pred2cnt, pred_func2cnt, pred2ix, predarg2ix, pred_func2ix)

        print ("retransforming the data given the min. freq. ...")
        
        to_ix = True
        prepared, corpus_stats = prepare_train(
            data_dir, fig_dir, transformed_dir, transform_config, config, False, print_pred_funcs, sample_only, q_snt_id, to_ix, data_loader_args, as_json, relpron_preds
        )   
        corpus_stats_file_path = os.path.join(transformed_info_dir, "corpus_stats_filt.txt")
        with open(corpus_stats_file_path, "w") as f:
            f.write("num_tokens\t" + str(corpus_stats["num_tokens"]) + "\n")
            f.write("num_surf_preds\t" + str(corpus_stats["num_surf_preds"]) + "\n")
            f.write("num_content_preds\t" + str(corpus_stats["num_content_preds"]) + "\n")
            f.write("num_pred_func_nodes\t" + str(corpus_stats["num_pred_func_nodes"]) + "\n")

        print ("all logical expressions are translated.")

        print ("balancing the splits ...")
        balance_splits(transformed_dir, as_json)
            
        print ("balanced transformed data saved at: {}".format(transformed_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare_train')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    parser.add_argument('-t', '--draw_tree', default="no", type=str,
                      help='save the figure of logical expression tree')
    parser.add_argument('-p', '--print_pred_funcs', default="no", type=str,
                      help='print predicate functions set')
    parser.add_argument('-q', '--q_snt_id', default=None, type=str,
                      help='prepare specific query snt_id only')
    parser.add_argument('-j', '--as_json', default="yes", type=str,
                      help='save as json file')
    parser.add_argument('-e', '--relpron_dir', default="eval_data_sets/RELPRON", type=str,
                      help='dir to RELPRON data')
    args = parser.parse_args()
    # custom cli options to modify configuration from default values given in json file.
    # CustomArgs = namedtuple('CustomArgs', 'flags type target')
    options = []
    config = ConfigParser.from_args(parser, options)
    main(config, args.draw_tree, args.print_pred_funcs, args.q_snt_id, args.as_json, args.relpron_dir)