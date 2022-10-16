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

from multiprocessing import Pool
from functools import reduce

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



def _prepare_train_worker(args):

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

    data_dir, fig_dir, transformed_dir, transform_config, worker_id, draw_tree, sample_only, q_snt_id, to_ix, data_loader_args = args
    if transformed_dir:
        transformed_file_path = os.path.join(transformed_dir, "_transformed_{}.json".format(worker_id))

    min_pred_func_freq, min_content_pred_freq, filter_min_freq, content_pred2cnt, pred_func2cnt, pred2ix, pred_func2ix = None, None, False, None, None, None, None
    if data_loader_args:
        min_pred_func_freq, min_content_pred_freq, filter_min_freq, content_pred2cnt, pred_func2cnt, pred2ix, pred_func2ix = data_loader_args

    prepared = {"pred_func2cnt": Counter(), "pred2cnt": Counter(), "content_pred2cnt": Counter(), "err2cnt": Counter()}
        
    found_q = False
    worker_files = [file for file in os.listdir(data_dir) if all([
        os.path.isfile(os.path.join(data_dir, file)),
        # file.startswith("{}_".format(str(worker_id))),
        sum(map(lambda x: int(x),filter(lambda x: str.isnumeric(x), file))) % NUM_WORKERS == worker_id,
        util.is_data_json(file)
        ])
    ]
    no_files = len(worker_files)
    
    transformed_keys = transform_config["transformed_keys"]

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
                    pred2ix, pred_func2ix
                )
                transformed = trsfm(instance)
                # transformed = {
                #     "discarded": self.discarded,
                #     "discarded_reason": self.discarded_reason,
                #     "node2pred": self.node2pred,
                #     "pred_func_nodes": list(self.pred_func_nodes),
                #     "content_preds": list(self.content_preds),
                #     "logic_expr": self.logic_expr,
                #     "pred_func_used": list(self.pred_func_used)
                # }
                discarded = transformed["discarded"]
                if discarded:
                    prepared["err2cnt"][transformed["discarded_reason"]] += 1
                    # if "Recursion" in transformed["discarded_reason"]:
                    # print (snt_id, snt_id, transformed["discarded_reason"])
                else:
                    # count pred_func
                    if not transformed_file_path:
                        _count_pred_func(prepared["pred_func2cnt"], prepared["pred2cnt"], transformed["logic_expr"], worker_id)
                        _count_content_pred(prepared["content_pred2cnt"], transformed["content_preds"], worker_id)
                    # id2transformed[snt_id]['decoders'] = transformed["decoders"]
                    # id2transformed[snt_id]['encoders'] = transformed["encoders"]
                    if not q_snt_id and transformed_file_path:
                        # transformed_json = {trsfm_key2abbrev[key]: transformed[key] for key in transformed_keys}
                        # transformed_json["snt_id"] = snt_id
                        transformed_json = [transformed[key] for key in transformed_keys]
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
    if not q_snt_id and transformed_file_path and written:
        with open(transformed_file_path, "a") as f:
            f.write("]")
        # with open(transformed_file_path, "r") as f:
        #     valid_json = validate_json(f)
            # if not valid_json:
            #     print ("Invalid json.")
            # else:
            #     print ("Valid json.")

    return {"prepared": prepared}

def prepare_train(data_dir, fig_dir, transformed_dir, transform_config, draw_tree, sample_only, q_snt_id, to_ix, data_loader_args = None):

    workers_args = [(data_dir, fig_dir, transformed_dir, transform_config, worker_id, draw_tree, sample_only, q_snt_id, to_ix, data_loader_args) for worker_id in range(NUM_WORKERS)]
    prepared = []
    with Pool(NUM_WORKERS) as p:
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
    print ("loading {}".format(transformed_path))
    with open(transformed_path) as f:
        transformed = json.load(f)
    print ("Finished loading {}".format(transformed_path))
    return transformed

def balance_splits(transformed_dir):
    num_instance = 0
    transformed_list = []
    transformed_paths =  [os.path.join(transformed_dir, "_transformed_{}.json".format(i)) for i in range(NUM_WORKERS)]

    workers_args = [transformed_paths[worker_id] for worker_id in range(NUM_WORKERS)]
    with Pool(NUM_WORKERS) as p:
        for idx, transformed in enumerate(p.imap(load_transformed, workers_args)):
            print ("Extending {}".format(idx))
            transformed_list.extend(transformed)
            del transformed

    num_instance = len(transformed_list)

    num_instance_per_file = int(num_instance/NUM_WORKERS)
    print ("total num_instance", num_instance)
    print ("num_instance_per_file", num_instance_per_file)
    for idx, chunk in enumerate(chunks(transformed_list, num_instance_per_file)):
        # transformed_dir, worker_id, bal_transformed = args
        print (idx, len(chunk))
        if len(chunk) == num_instance:
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


def main(config, draw_tree = "no", q_snt_id = None):


    sample_only = config["sample_only"]
    data_loader_args = config['data_loader']['args']
    data_dir = data_loader_args['data_dir']
    min_pred_func_freq = data_loader_args["min_pred_func_freq"]
    min_content_pred_freq = data_loader_args["min_content_pred_freq"]
    filter_min_freq = data_loader_args["filter_min_freq"]
    transform_config_file_path = data_loader_args['transform_config_file_path']

    transformed_dir  = data_loader_args["transformed_dir"]
    transformed_info_dir = os.path.join(transformed_dir, "info")
    os.makedirs(transformed_info_dir, exist_ok = True)

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

    with open(transform_config_file_path) as f:
        transform_config = json.load(f)

    fig_dir = None
    if draw_tree:
        fig_dir = os.path.join(data_dir, "figures")
        os.makedirs(fig_dir, exist_ok = True)



    to_ix = False
    prepared = prepare_train(data_dir, fig_dir, None, transform_config, draw_tree, sample_only, q_snt_id, to_ix)

    run_dir = config.run_dir
    log_dir = config.log_dir

    pred_func2ix = defaultdict()
    for ix, (pred_func, cnt) in enumerate(prepared["pred_func2cnt"].most_common()):
        if cnt >= min_pred_func_freq:
            pred_func2ix[pred_func] = ix
    pred_func2ix_file_path = os.path.join(transformed_info_dir, "pred_func2ix.txt")
    with open(pred_func2ix_file_path, "w") as f:
        for pred_func, ix in pred_func2ix.items():
            f.write("{}\t{}\n".format(str(ix), pred_func))

    pred_func2cnt_file_path = os.path.join(transformed_info_dir, "pred_func2cnt.txt")
    with open(pred_func2cnt_file_path, "w") as f:
        for ix, (pred_func, cnt) in enumerate(prepared["pred_func2cnt"].most_common()):
            if cnt >= min_pred_func_freq:
                f.write("{}\t{}\n".format(pred_func2ix[pred_func], str(cnt)))

    preds = set()
    for content_pred, cnt in prepared["content_pred2cnt"].most_common():
        if cnt >= min_content_pred_freq:
            preds.add(content_pred)
    for pred_func, cnt in prepared["pred_func2cnt"].most_common():
        if cnt >= min_pred_func_freq:
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




    # pred_func2cnt, content_pred2cnt = prepared["pred_func2cnt"] , prepared["content_pred2cnt"]
    pred_func2cnt, content_pred2cnt, pred2ix, pred_func2ix = get_transformed_info(transformed_info_dir)

    data_loader_args = (min_pred_func_freq, min_content_pred_freq, filter_min_freq, content_pred2cnt, pred_func2cnt, pred2ix, pred_func2ix)

    print ("retransforming the data given the min. freq. ...")
    
    to_ix = True
    prepared = prepare_train(
        data_dir, fig_dir, transformed_dir, transform_config, False, sample_only, q_snt_id, to_ix, data_loader_args
    )

    print ("all logical expressions are translated.")

    print ("balancing the splits ...")
    balance_splits(transformed_dir)
        
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
    parser.add_argument('-q', '--q_snt_id', default=None, type=str,
                      help='prepare specific query snt_id only')
    args = parser.parse_args()
    # custom cli options to modify configuration from default values given in json file.
    # CustomArgs = namedtuple('CustomArgs', 'flags type target')
    options = []
    config = ConfigParser.from_args(parser, options)
    main(config, args.draw_tree, args.q_snt_id)