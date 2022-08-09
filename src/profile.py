import os

import matplotlib.pyplot as plt

from delphin import itsdb
from delphin.codecs import simplemrs, dmrsjson
from delphin.dmrs import from_mrs
from delphin.mrs._exceptions import MRSError, MRSSyntaxError

import argparse
import sys
import re
from pprint import pprint
from collections import defaultdict, Counter
import json
import os

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm
    
import tarfile

import gzip
import shutil

from pprint import pprint
          
GENERICS = ['generic_entity_rel', 'pron_rel']
    
def print_pred(predicate):

    pred_lemma, pred_pos = None, None
    
    if predicate[0] == "_":
   
        if "unknown" in predicate and predicate not in ['_unknown_a_1', '_unknown_n_1']:
            try:
                pred_lemma, pred_unk_pos = predicate.rsplit("/",1)
            except:
                print (predicate)
            pred_unk_pos_split = pred_unk_pos.rsplit("_", 2)
            assert pred_unk_pos_split[1:3] == ['u', 'unknown']
            pred_pos = pred_unk_pos_split[0]
            pred_lemma = pred_lemma.replace('+',' ')[1:]
            # print (predicate)
            # print (pred_lemma, pred_pos)
            # print ()

        elif predicate.count('_') not in [2,3]:
            if predicate in ['_only_child_n_1', '_nowhere_near_x_deg']:
                pred_lemma, pred_pos, *_ = predicate.rsplit("_",2)
                pred_lemma = pred_lemma[1:]
            else:
                print (predicate)
                print ()

        else:
            _, pred_lemma, pred_pos, *_ = predicate.split("_")
            pred_lemma = pred_lemma.replace('+',' ')
            if pred_pos == 'dir':
                pred_pos = 'p'
                print (pred_pos)
                print ()
            if pred_pos == 'state':
                print (pred_pos)
                print ()

        if not "unknown" in predicate and predicate not in ['_'] and pred_pos not in 'a v n q x p c':
            print (predicate)
            print (pred_lemma, pred_pos)
            pred_lemma, pred_pos, *_ = predicate.rsplit("_", 2)
            pred_lemma = pred_lemma.replace('+',' ')[1:]
            print (pred_lemma, pred_pos)
            print ()
    

def profile(dmrs_dir):
    
    
    stats = Counter()
    stats['pred2cnt'] = Counter()
    files = [f for f in os.listdir(dmrs_dir) if os.path.isfile(os.path.join(dmrs_dir, f))]
    # for root, dirs, files in os.walk(dmrs_dir):
    for file in tqdm(files):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(dmrs_dir, file), "r") as f:
            id2instance_str = f.read()
            id2instance = json.loads(id2instance_str)
            stats['dmrs_cnt'] += len(id2instance)
            for idx, instance in id2instance.items():
                dmrs = instance['dmrs']
                has_generics = False
                # count all preds freq
                for node in dmrs['nodes']:
                    stats['pred2cnt'][node['predicate']] += 1
                    print_pred(node['predicate'])
                
                    if node['predicate'] in GENERICS:
                        has_generics = True
                # pron
                if has_generics: stats['#dmrs_with_generics'] += 1
                # assertion
                if len(dmrs['nodes']) == 0:
                    print ('0 nodes found')
                    print (dmrs)
                    input()
            

    return stats
                  


def count_dmrs_wo_rare_pred(dmrs_dir, profile_dir, pred2cnt_filename):
    
    pred2cnt = Counter()
    pred2cnt_path = os.path.join(profile_dir, pred2cnt_filename)
    with open(pred2cnt_path, "r", encoding = 'utf-8') as f:
        line = f.readline()
        while line:
            pred, freq = line.split("\t")
            pred2cnt[pred] = int(freq)
            line = f.readline()
    
    min_freq2num_dmrs = Counter()
    files = [f for f in os.listdir(dmrs_dir) if os.path.isfile(os.path.join(dmrs_dir, f))]
    for file in tqdm(files):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(dmrs_dir, file), "r") as f:
            idx2instance_str = f.read()
            idx2instance = json.loads(idx2instance_str)
            for snt_id, instance in idx2instance.items():
                dmrs = instance['dmrs']
                min_freq2num_dmrs[min([pred2cnt[node['predicate']] for node in dmrs['nodes']])] += 1
                            
    return min_freq2num_dmrs
    
        
def main(dmrs_dir, profile_dir):
    
    # stats = profile(dmrs_dir)
    
    # os.makedirs(profile_dir, exist_ok = True)
    
    # stats_path = os.path.join(profile_dir, "stats.txt")
    # with open(stats_path, "w") as f:
    #     for stat in stats:
    #         if stat != 'pred2cnt':
    #             f.write("{}\t{}\n".format(stat, str(stats[stat])))
                
    pred2cnt_filename = "pred2cnt_27072022.txt"
    # pred2cnt_path = os.path.join(profile_dir, pred2cnt_filename)
    # with open(pred2cnt_path, "w") as f:
    #     for k,v in stats['pred2cnt'].most_common():
    #         f.write("{}\t{}\n".format(k,v))
    
    min_freq2num_dmrs = count_dmrs_wo_rare_pred(dmrs_dir, profile_dir, pred2cnt_filename)
    
    min_freq2num_dmrs_filename = "min_freq2num_dmrs.txt"
    min_freq2num_dmrs_path = os.path.join(profile_dir, min_freq2num_dmrs_filename)
    with open(min_freq2num_dmrs_path, "w") as f:
        for k,v in min_freq2num_dmrs.most_common():
            f.write("{}\t{}\n".format(str(k),str(v)))
    

    # return num_dmrs_wo_rare_pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dmrs_dir', default=None, help='path to target data directory')
    parser.add_argument('profile_dir', default=None, help='path to target data directory')
    args = parser.parse_args()
    main(args.ww1212_dir)
                
    