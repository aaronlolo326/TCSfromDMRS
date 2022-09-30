import argparse
from collections import namedtuple, Counter, defaultdict
import torch
import numpy as np
from parse_config import ConfigParser

from transform.ftcs_transform import TruthConditions

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

from multiprocessing import Pool
from functools import reduce

from pprint import pprint

from src import util, dg_util

import networkx as nx
from networkx.readwrite.json_graph import node_link_data
from networkx.drawing.nx_agraph import to_agraph

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
    
def draw_logic_expr(logic_expr, timestamp = False, name = "err", save_path = None):
    
    def _build_tree(logic_expr_tree, sub_logic_expr, curr_node, par_node, edge_lbl):
        if isinstance(sub_logic_expr, str):
            logic_expr_tree.add_node(curr_node, label = sub_logic_expr)
        elif isinstance(sub_logic_expr, dict):
            logic_expr_tree.add_node(curr_node, label = "{} {}".format(sub_logic_expr['pred_func_name'], str(sub_logic_expr['args'])))
        elif sub_logic_expr:
            root, *dgtrs = sub_logic_expr
            logic_expr_tree.add_node(curr_node, label = root)
            for dgtr_idx, dgtr in enumerate(dgtrs):
                _build_tree(logic_expr_tree, dgtr, curr_node*2 + dgtr_idx, curr_node, dgtr_idx)
        if par_node:
            logic_expr_tree.add_edge(par_node, curr_node, label = edge_lbl)

    
    logic_expr_tree = nx.DiGraph()
    # pprint (logic_expr)
    _build_tree(logic_expr_tree, logic_expr, 1, None, None)
            
    time_str = "_" + time.asctime( time.localtime(time.time()) ).replace(" ", "-") if timestamp else ""
    if not save_path:
        save_path = "./figures/logic_expr_{}".format(name) + time_str + ".png"
    ag = to_agraph(logic_expr_tree)
    ag.layout('dot')
    ag.draw(save_path)
    print ("logic expression tree drawn:", save_path)


def _prepare_train_worker(args):

    def _count_pred_func(pred_func2cnt, pred2cnt, logic_expr, worker_id):
        if isinstance(logic_expr, dict):
            pred_func2cnt[logic_expr['pred_func_name']] += 1
            pred2cnt[logic_expr['pred_func_name'].rsplit("@", 1)[0]] += 1
        elif logic_expr:
            root, *dgtrs = logic_expr
            for dgtr in dgtrs:
                _count_pred_func(pred_func2cnt, pred2cnt, dgtr, worker_id)

    def _count_lex_pred(lex_pred2cnt, lexical_preds, worker_id):
        lex_pred2cnt.update(lexical_preds)

    def validate_json(f):
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "snt_id": {
                        "type": "string"
                    },
                    "decoders": {
                        "type": "object",
                        "properties": {
                            "logic_expr": {
                                "type": "array",
                            }
                        },
                        "required": ["logic_expr"]
                    },
                    "encoders": {
                        "type": "object",
                        "properties": {
                            "pred_func_nodes": {
                                "type": "array"
                            },
                            "lexical_preds": {
                                "type": "array"
                            }
                        },
                        "required": ["pred_func_nodes", "lexical_preds"]
                    }
                },
                "required": ["decoders", "encoders", "snt_id"]
            }
        }
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

    id2transformed_file_path = None
    written = False

    data_dir, fig_dir, transformed_dir, transform_config, worker_id, draw_tree, sample_only, q_snt_id, data_loader_args = args
    if transformed_dir:
        id2transformed_file_path = os.path.join(transformed_dir, "transformed_{}.json".format(worker_id))

    min_pred_func_freq, min_lex_pred_freq, lex_pred2cnt, pred_func2cnt, filter_min_freq = None, None, None, None, False
    if data_loader_args:
        min_pred_func_freq, min_lex_pred_freq, lex_pred2cnt, pred_func2cnt = data_loader_args
        filter_min_freq = True

    prepared = {"pred_func2cnt": Counter(), "pred2cnt": Counter(), "lex_pred2cnt": Counter(), "err2cnt": Counter()}
        
    found_q = False
    worker_files = [file for file in os.listdir(data_dir) if all([
        os.path.isfile(os.path.join(data_dir, file)),
        file.startswith("{}_".format(str(worker_id))),
        util.is_data_json(file)
        ])
    ]
    no_files = len(worker_files)
        
    for file_idx, file in enumerate(worker_files):
        if not sample_only: #and file_idx%(no_files/1000) == 0:
            print ("worker {}: {}% done".format(worker_id, file_idx/no_files * 100))
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
                    transform_config, min_pred_func_freq, min_lex_pred_freq, lex_pred2cnt, pred_func2cnt, filter_min_freq
                )
                transformed = trsfm(instance)
                discarded = transformed["discarded"]
                if discarded:
                    prepared["err2cnt"][transformed["discarded_reason"]] += 1
                    # if "Recursion" in transformed["discarded_reason"]:
                    # print (snt_id, snt_id, transformed["discarded_reason"])
                else:
                    # count pred_func
                    logic_expr = transformed["decoders"]["logic_expr"]
                    _count_pred_func(prepared["pred_func2cnt"], prepared["pred2cnt"], logic_expr, worker_id)
                    lexical_preds = transformed["encoders"]["lexical_preds"]
                    _count_lex_pred(prepared["lex_pred2cnt"], lexical_preds, worker_id)
                    # id2transformed[snt_id]['decoders'] = transformed["decoders"]
                    # id2transformed[snt_id]['encoders'] = transformed["encoders"]
                    if not q_snt_id and id2transformed_file_path:
                        transformed_json = defaultdict()
                        transformed_json["snt_id"] = snt_id
                        transformed_json["decoders"] = transformed["decoders"]
                        transformed_json["encoders"] = transformed["encoders"]
                        transformed_json["node2pred"] = transformed["node2pred"]
                        if not written:
                            delimiter = "["
                            write_mode = 'w'
                            written = True
                        else:
                            delimiter = ", "
                            write_mode = 'a'
                        with open(id2transformed_file_path, write_mode) as f:
                            f.write(delimiter)
                            f.write(json.dumps(transformed_json))
                if draw_tree:
                    snt = instance['snt']
                    logic_expr_save_path = os.path.join(fig_dir, "logic_expr_{}.png".format(snt_id))
                    print (snt)
                    draw_logic_expr(logic_expr, save_path = logic_expr_save_path)
                    erg_digraphs = dg_util.Erg_DiGraphs()
                    dmrs_nxDG = nx.node_link_graph(instance['dmrs'])
                    erg_digraphs.init_dmrs_from_nxDG(dmrs_nxDG)
                    erg_digraphs.init_snt(snt)

                    dmrs_save_path = os.path.join(fig_dir, "dmrs_{}.png".format(snt_id))#+ time.asctime( time.localtime(time.time()) ).replace(" ", "-") +".png"
                    erg_digraphs.draw_dmrs(save_path = dmrs_save_path)

                if q_snt_id and found_q or sample_only and idx_loop >= 1000:
                    break
            if q_snt_id and found_q or sample_only:
                break
        if q_snt_id and found_q or sample_only:
            break
        # print ("processed {}".format(file))
        # pprint (sub_pred_func2cnt)
    if not q_snt_id and id2transformed_file_path and written:
        with open(id2transformed_file_path, "a") as f:
            f.write("]")
        with open(id2transformed_file_path, "r") as f:
            valid_json = validate_json(f)
            # if not valid_json:
            #     print ("Invalid json.")
            # else:
            #     print ("Valid json.")

    return {"prepared": prepared}

def prepare_train(data_dir, fig_dir, transformed_dir, transform_config, draw_tree, sample_only, q_snt_id, data_loader_args = None):

    workers_args = [(data_dir, fig_dir, transformed_dir, transform_config, worker_id, draw_tree, sample_only, q_snt_id, data_loader_args) for worker_id in range(10)]
    prepared = []
    with Pool(10) as p:
        prepared_workers = list(p.imap(_prepare_train_worker, workers_args))
    
    print ("Pool ended. Reducing ...")
    # prepared = reduce(lambda x, y: {key: x[0][key] + y[0][key] for key in x}, prepared_workers)
    id2transformed_list = []
    prepared = defaultdict(Counter)
    for prepared_worker in prepared_workers:
        # id2transformed_list.append(prepared_worker["id2transformed"])
        for key, data in prepared_worker['prepared'].items():
            prepared[key] = prepared[key] + data

    return prepared



def main(config, draw_tree = "no", sample_only = "no", q_snt_id = None):

    sample_str = None
    if sample_only == "no":
        sample_only = False
        sample_str = ""
    elif sample_only == "yes":
        sample_only = True
        sample_str = "sample"
    if q_snt_id != None: sample_only = False
    if draw_tree == "no": draw_tree = False
    elif draw_tree == "yes": draw_tree = True

    data_loader_args = config['data_loader']['args']
    data_dir = data_loader_args['data_dir']
    min_pred_func_freq = data_loader_args["min_pred_func_freq"]
    min_lex_pred_freq = data_loader_args["min_lex_pred_freq"]
    transform_config_file_path = data_loader_args['transform_config_file_path']

    with open(transform_config_file_path) as f:
        transform_config = json.load(f)

    fig_dir = None
    if draw_tree:
        fig_dir = os.path.join(data_dir, "figures")
        os.makedirs(fig_dir, exist_ok = True)

    # data_info_dir = os.path.join(data_dir, "info")
    # if snt_id:
    #     with open(os.path.join(data_info_dir, "idx2file_path.json")) as f:
    #         idx2file_path = json.load(f)
    #         file_path = idx2file_path[snt_id]

    # num_lex_pred2cnt = prepare_train_max(data_dir, transform_config, sample_only)
    # print (num_lex_pred2cnt)
    transformed_dir = None
    
    prepared = prepare_train(data_dir, fig_dir, transformed_dir, transform_config, draw_tree, sample_only, q_snt_id)

    run_dir = config.run_dir
    log_dir = config.log_dir

    pred_func2cnt_file_path = os.path.join(run_dir, "pred_func2cnt.txt")
    with open(pred_func2cnt_file_path, "w") as f:
        for pred_func, cnt in prepared["pred_func2cnt"].most_common():
            if cnt >= min_pred_func_freq:
                f.write("{}\t{}\n".format(pred_func, str(cnt)))

    lex_pred2cnt_file_path = os.path.join(run_dir, "lex_pred2cnt.txt")
    with open(lex_pred2cnt_file_path, "w") as f:
        for lex_pred, cnt in prepared["lex_pred2cnt"].most_common():
            if cnt >= min_lex_pred_freq:
                f.write("{}\t{}\n".format(lex_pred, str(cnt)))

    preds = set()
    for pred_func, cnt in prepared["pred_func2cnt"].most_common():
        if cnt >= min_pred_func_freq:
            pred = pred_func.rsplit("@", 1)[0]
            preds.add(pred)
        else:
            break
    sorted_preds = sorted(list(preds))
    print ("saving pred2ix of {} predicates ...".format(len(sorted_preds)))
    pred2ix_path = os.path.join(run_dir, "pred2ix.txt")
    with open(pred2ix_path, "w") as f:
        for ix, pred in enumerate(sorted_preds):
            f.write("{}\t{}\n".format(ix, pred))

    print ("prepared run data saved at: {}".format(run_dir))

    errd2cnt_file_path = os.path.join(run_dir, "err2cnt.txt")
    with open(errd2cnt_file_path, "w") as f:
        for err, cnt in prepared["err2cnt"].most_common():
            f.write("{}\t{}\n".format(err, str(cnt)))

    pred_func2cnt, lex_pred2cnt = prepared["pred_func2cnt"] , prepared["lex_pred2cnt"]

    data_loader_args = (min_pred_func_freq, min_lex_pred_freq, lex_pred2cnt, pred_func2cnt)
    
    transformed_dir =  os.path.join(data_dir, "transformed", config['name']) #"_{}".format(sample_str))
    os.makedirs(transformed_dir, exist_ok = True)

    print ("retransforming the data given the min. freq. ...")
    
    prepared = prepare_train(
        data_dir, fig_dir, transformed_dir, transform_config, False, sample_only, q_snt_id, data_loader_args
    )

    # sorted_preds = sorted(list(prepared["pred2cnt"].keys()))
    # print ("saving pred2ix of {} predicates ...".format(len(sorted_preds)))
    # pred2ix_path = os.path.join(transformed_dir, "pred2ix.txt")
    # with open(pred2ix_path, "w") as f:
    #     for ix, pred in enumerate(sorted_preds):
    #         f.write("{}\t{}\n".format(ix, pred))
        
    print ("transformed data saved at: {}".format(transformed_dir))

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
    parser.add_argument('-s', '--sample_only', default="no", type=str,
                      help='prepare first few sample instances only')
    parser.add_argument('-q', '--q_snt_id', default=None, type=str,
                      help='prepare specific query snt_id only')
    args = parser.parse_args()
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = namedtuple('CustomArgs', 'flags type target')
    options = []
    config = ConfigParser.from_args(parser, options)
    main(config, args.draw_tree, args.sample_only, args.q_snt_id)