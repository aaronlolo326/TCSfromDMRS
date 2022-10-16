import argparse
from collections import namedtuple, Counter, defaultdict
import torch
from parse_config import ConfigParser

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

from pprint import pprint
import json
import os
from itertools import product

from src import util, dg_util

def map_pred():
    pass

def prepare_hyp(config, hyp_dir, hyp_file, transformed_dir, min_hyp_pred_pair_freq = None):
    
    MIN_CONTENT_PRED_FREQ = config["data_loader"]["args"]["min_content_pred_freq"]
    MIN_PRED_FUNC_FREQ = config["data_loader"]["args"]["min_pred_func_freq"]
    # MIN_HYP_PAIR_FREQ = 10000
    if not min_hyp_pred_pair_freq:
        min_hyp_pred_pair_freq = MIN_PRED_FUNC_FREQ
    print ("hyp_file: {}".format(hyp_file))
    print ("min_hyp_pred_pair_freq: {}".format(min_hyp_pred_pair_freq))

    hyp_wn_path = os.path.join(hyp_dir, hyp_file)
    content_pred2cnt_path = os.path.join(transformed_dir, "info", "content_pred2cnt.txt")
    pred2ix_path = os.path.join(transformed_dir, "info", "pred2ix.txt")
    pred_func2cnt_path = os.path.join(transformed_dir, "info", "pred_func2cnt.txt")
    pred_func2ix_path = os.path.join(transformed_dir, "info", "pred_func2ix.txt")

    pred_funcs = []
    with open(pred_func2ix_path) as f:
        line = f.readline()
        while line:
            ix, pred_func = line.strip().split("\t")
            pred_funcs.append(pred_func)
            line = f.readline()

    pred_func2cnt = Counter()
    with open(pred_func2cnt_path) as f:
        line = f.readline()
        while line:
            ix, cnt = line.strip().split("\t")
            if int(cnt) >= MIN_PRED_FUNC_FREQ:
                pred_func2cnt[pred_funcs[int(ix)]] = int(cnt)
            line = f.readline()
    
    # pred_func_preds = set(map(lambda x: x.rsplit("@", 1)[0], pred_funcs))

    # content_preds = []
    # with open(pred2ix_path) as f:
    #     line = f.readline()
    #     while line:
    #         ix, pred = line.strip().split("\t")
    #         content_preds.append(pred)
    #         line = f.readline()

    # content_pred2cnt = Counter()
    # with open(content_pred2cnt_path) as f:
    #     line = f.readline()
    #     while line:
    #         pred_ix, cnt = line.strip().split("\t")
    #         if int(cnt) < MIN_CONENT_PRED_FREQ:
    #             print ("min exceeded:", cnt)
    #             break
    #         content_pred2cnt[content_preds[int(pred_ix)]] = int(cnt)
    #         line = f.readline()

    lemma2pred2cnt = defaultdict(Counter)
    for pred_func, cnt in pred_func2cnt.items():
        pred = pred_func.rsplit("@", 1)[0]
        pred_lemma, pred_pos = util.get_lemma_pos(pred)
        if pred_pos == 'n':
            # noun_lemmas_dmrs2cnt[pred_lemma] = content_pred2cnt[pred]
            lemma2pred2cnt[pred_lemma][pred] += cnt

    noun_lemmas_wn = set()
    hyp_pairs = []
    with open(hyp_wn_path) as f:
        line = f.readline()
        while line:
            hypos, hypers = line.strip().split("\t")
            hypos = hypos.split(", ")
            hypers = hypers.split(", ")
            noun_lemmas_wn.update(set(hypos))
            noun_lemmas_wn.update(set(hypers))
            common_hypos = filter(lambda x: x in lemma2pred2cnt and lemma2pred2cnt[x].most_common()[0][1] >= min_hyp_pred_pair_freq, hypos)
            common_hypers = filter(lambda x: x in lemma2pred2cnt and lemma2pred2cnt[x].most_common()[0][1] >= min_hyp_pred_pair_freq, hypers)
            hyp_pairs.extend(product(common_hypos, common_hypers))
            line = f.readline()

    hyp_pred_pairs = []
    for hyp_pair in hyp_pairs:
        hyp_pred_pairs.append(tuple(map(lambda x: lemma2pred2cnt[x].most_common()[0][0], hyp_pair)))

    # print (len(noun_lemmas_wn))

    common_lemmas = noun_lemmas_wn.intersection(lemma2pred2cnt.keys())
    print ("#common_lemmas:", len(common_lemmas))
    print ("#hyp_paris:", len(hyp_pairs))
    print ()
    # pprint (hyp_pred_pairs)
    # pprint (common_lemmas)
    # print ()
    # for k, v in lemma2pred2cnt.items():
    #     print (k, v)
    
    return hyp_pred_pairs


def main(config, hyp_dir, eval_dir):

    transformed_dir  = config["data_loader"]["args"]["transformed_dir"]
    transformed_dir_name = config["transformed_dir_name"]

    config_eval_dir = os.path.join(eval_dir, transformed_dir_name)
    config_eval_hyp_dir = os.path.join(config_eval_dir, "hyp")
    os.makedirs(config_eval_hyp_dir, exist_ok = True)

    min_freqs = [0, 5000, 50000]
    for trasitive_num in [1,2]:
        hyp_file = "hypernyms_wn_{}.txt".format(trasitive_num)
        for freq in min_freqs:
            hyp_pred_pairs_wn = prepare_hyp(config, hyp_dir, hyp_file, transformed_dir, freq)
            if hyp_pred_pairs_wn:
                hyp_pred_pairs_path = os.path.join(config_eval_hyp_dir, "hyp_pred_pairs_t{}_f{}.json".format(trasitive_num, freq))
                with open(hyp_pred_pairs_path, "w") as f:
                    json.dump(hyp_pred_pairs_wn, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare_train')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    parser.add_argument('-p', '--hyp_dir', default=None, type=str,
                      help='directory to hypernym data (default: None)')
    parser.add_argument('-e', '--eval_dir', default=None, type=str,
                      help='directory to evaluation data (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()
    # custom cli options to modify configuration from default values given in json file.
    options = []
    config = ConfigParser.from_args(parser, options)
    main(config, args.hyp_dir, args.eval_dir)