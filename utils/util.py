import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict, Counter, defaultdict
import os
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import time

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def get_transformed_info(transformed_info_dir):

    pred_func2cnt_file_path = os.path.join(transformed_info_dir, "pred_func2cnt.txt")
    content_pred2cnt_file_path = os.path.join(transformed_info_dir, "content_pred2cnt.txt")
    pred2ix_file_path = os.path.join(transformed_info_dir, "pred2ix.txt")
    pred_func2ix_file_path = os.path.join(transformed_info_dir, "pred_func2ix.txt")

    pred_func2cnt = Counter()
    with open(pred_func2cnt_file_path) as f:
        line = f.readline()
        while line:
            pred_func, cnt = line.strip().split("\t")
            pred_func = int(pred_func)
            pred_func2cnt[pred_func] = int(cnt)
            line = f.readline()

    content_pred2cnt = Counter()
    with open(content_pred2cnt_file_path) as f:
        line = f.readline()
        while line:
            content_pred, cnt = line.strip().split("\t")
            content_pred2cnt[int(content_pred)] = int(cnt)
            line = f.readline()

    pred2ix = defaultdict()
    with open(pred2ix_file_path) as f:
        line = f.readline()
        while line:
            ix, pred = line.strip().split("\t")
            # if int(cnt) < MIN_PRED_FUNC_FREQ:
            #     break
            pred2ix[pred] = int(ix)
            line = f.readline()

    pred_func2ix = defaultdict()
    with open(pred_func2ix_file_path) as f:
        line = f.readline()
        while line:
            ix, pred_func = line.strip().split("\t")
            ix = int(ix)
            pred_func2ix[pred_func] = ix
            line = f.readline()

    return pred_func2cnt, content_pred2cnt, pred2ix, pred_func2ix

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def draw_logic_expr(logic_expr, timestamp = False, name = "err", save_path = None):
    
    def _build_tree(logic_expr_tree, sub_logic_expr, curr_node, par_node, edge_lbl):
        if isinstance(sub_logic_expr, str):
            logic_expr_tree.add_node(curr_node, label = sub_logic_expr)
        elif isinstance(sub_logic_expr, dict):
            logic_expr_tree.add_node(curr_node, label = "{} {}".format(sub_logic_expr['pf'], str(sub_logic_expr['args'])))
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