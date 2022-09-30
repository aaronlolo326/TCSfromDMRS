import nltk
# nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

import networkx as nx

from delphin import itsdb
from delphin.codecs import simplemrs, dmrsjson
from delphin.dmrs import from_mrs
from delphin.mrs._exceptions import MRSError, MRSSyntaxError
from delphin.util import SExpr
from delphin.derivation import from_string, DerivationSyntaxError

import argparse
import sys
import re
from pprint import pprint
from collections import defaultdict, Counter
import json
import os
from copy import deepcopy
from multiprocessing import Pool
from functools import reduce

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm
    
import tarfile
import gzip
import shutil

from pprint import pprint

from src import util, dg_util
from src.dg_util import DerivationBranchingError, DMRSNoLnkError, DisconnectedDMRSError, ExtraPredicateError

import warnings
warnings.filterwarnings("ignore")

VERBOSE = False

def _cpd_remove_name():
    pass

def _cpd_merge():
    pass

def _cpd_remove_num():
    pass

def _cpd_remove_city():
    pass

def _cpd_unchange():
    pass


CMP2OP = {
    'np-hdn_cpd_c': _cpd_remove_name,
    'np-hdn_ttl-cpd_c': _cpd_merge,
    'np-hdn_nme-cpd_c': _cpd_merge,
    'np-hdn_num-cpd_c': _cpd_remove_num,
    'np-hdn_cty-cpd_c': _cpd_remove_city,
    'n-hdn_cpd_c': _cpd_unchange,
    'n-hdn_j-n-cpd_c': _cpd_unchange,
    'n-hdn_ttl-cpd_c': _cpd_unchange,
    'n-nh_vorj-cpd_c': _cpd_unchange,
    'n-nh_j-cpd_c': _cpd_unchange,
}

cmp_args = Counter()
                
def lemmatize_unk(pred_lemma, pred_pos):
    pos_unk2pos_wn = {"j": "a", "v": "v", "n": "n", "r": "r"}
    wordnet_lemmatizer = WordNetLemmatizer()
    if pred_lemma[0] in pos_unk2pos_wn:
        norm_lemma = wordnet_lemmatizer.lemmatize(pred_lemma, pos_unk2pos_wn[pred_lemma[0]])
    else:
        norm_lemma = wordnet_lemmatizer.lemmatize(pred_lemma)
    return norm_lemma
    
def normalize_pred(pred, unk2pos):
    norm_pred = None
    if not "_u_unknown" in pred:
        norm_pred = pred
    else:
        pred_lemma, pred_pos = util.get_lemma_pos(pred)
        norm_prefix = "u"
        norm_lemma = lemmatize_unk(pred_lemma, pred_pos)
        norm_pos = unk2pos[pred_pos]
        norm_pred = "_".join([norm_prefix, norm_lemma, norm_pos])
    if norm_pred.endswith("_rel"):
        norm_pred = norm_pred[:-4]
    return norm_pred


def propagate_anchors(deriv, curr_node):
    noOfChildren = len(deriv.out_edges(nbunch = [curr_node]))
    # Terminal
    if noOfChildren == 0:
        # normalize form
        # deriv.nodes[curr_node]['form'] = normalize_sentence(deriv.nodes[curr_node]['form'])
        try:
            return (deriv.nodes[curr_node]['anchor_from'], deriv.nodes[curr_node]['anchor_to'])
        except:
            dg_util.draw_deriv(deriv)
            input()
    # Non-terminal
    else:
        children_anchors = [propagate_anchors(deriv, list(deriv.out_edges(nbunch = [curr_node]))[i][1])
                 for i in range(noOfChildren)]
        deriv.nodes[curr_node]['anchor_from'] = str(min([int(ancfrom) for ancfrom, ancto in children_anchors]))
        deriv.nodes[curr_node]['anchor_to'] = str(max([int(ancto) for ancfrom, ancto in children_anchors]))
        # print (deriv.nodes[curr_node]['entity'], children_anchors,
        #         deriv.nodes[curr_node]['anchor_from'], deriv.nodes[curr_node]['anchor_to'])
        return (deriv.nodes[curr_node]['anchor_from'], deriv.nodes[curr_node]['anchor_to'])
    
def syn_tree_anchors_from_deriv(syn_tree, curr_syn_tree_node, deriv, curr_deriv_node):
    noOfChildren = len(syn_tree.out_edges(nbunch = [curr_syn_tree_node]))
    syn_tree.nodes[curr_syn_tree_node]['anchor_from'] = deriv.nodes[curr_deriv_node]['anchor_from']
    syn_tree.nodes[curr_syn_tree_node]['anchor_to'] = deriv.nodes[curr_deriv_node]['anchor_to']
    deriv.nodes[curr_deriv_node]['cat'] = syn_tree.nodes[curr_syn_tree_node]['label']
    if noOfChildren > 0:
        for idx, (src,targ) in enumerate(list(syn_tree.out_edges(nbunch = [curr_syn_tree_node]))):
            try:
                next_deriv_node = list(deriv.out_edges(nbunch = [curr_deriv_node]))[idx][1]
                syn_tree_anchors_from_deriv(syn_tree, targ, deriv, next_deriv_node)
            except:
                # print (idx, deriv.out_edges(nbunch = [curr_deriv_node]))
                return False
    return True

def dmrs_rewrite(snt_id, snt, erg_digraphs, rewrite_type):
    
    def _pred_match(pred, rewrite_type):
        if rewrite_type in ['nominalization', 'eventuality']:
            return pred[:-4] == rewrite_type
        elif rewrite_type == 'modal':
            return pred.endswith('modal')
        elif rewrite_type == 'compound':
            return pred in ['compound_name', 'compound']
            
    # deriv_dg = erg_digraphs.deriv_dg
    dmrs_dg = erg_digraphs.dmrs_dg
    
    erg_digraphs_re = deepcopy(erg_digraphs)

    node2new_node = defaultdict()

    # [((merging_nodes..),(retain_node))..]
    merge_list = []

    for node, node_prop in dmrs_dg.nodes(data = True):

        node2new_node[node] = node

        merging_nodes, retain_node = None, None
        
        if _pred_match(node_prop['predicate'], rewrite_type):
            
            if rewrite_type in ['nominalization', 'eventuality', 'modal']:
                arg1_node = None
                out_edges = list(dmrs_dg.out_edges(node, data='label'))
                # try:
                #     assert len(out_edges) == 1 # False when mod/eq
                # except:
                #     pprint (node_prop)
                #     print (out_edges)
                for src, targ, lbl in dmrs_dg.out_edges(node, data='label'):
                    if lbl.startswith('ARG1'):
                        arg1_node = targ
                        break
                if arg1_node:
                    merging_nodes = (node, arg1_node)
                    retain_node = arg1_node
                # draw before
                # if True and rewrite_type == 'modal':
                #     erg_digraphs_re.draw_dmrs(name = '{}1'.format(rewrite_type))

            elif rewrite_type == 'compound':
                # compound name connects arg1 named
                # ignore the connected predicates during training
                arg1_node, arg2_node = None, None
                for src, targ, lbl in dmrs_dg.out_edges(node, data='label'):
                    if lbl.startswith('ARG1'): arg1_node = targ
                    elif lbl.startswith('ARG2'): arg2_node = targ
                if arg1_node and arg2_node:
                    arg1_node_prop = dmrs_dg.nodes[arg1_node]
                    arg2_node_prop = dmrs_dg.nodes[arg2_node]

                    if arg1_node_prop['predicate'] == 'named' or arg2_node_prop['predicate'] == 'named':
                        merging_nodes = (node, arg1_node, arg2_node)

                        if arg2_node_prop['predicate'] == 'named':
                            retain_node = arg1_node
                        else:
                            retain_node = arg2_node
                        
            
            if merging_nodes and retain_node:
                merge_list.append((merging_nodes, retain_node))

    for merging_nodes, retain_node in merge_list:

        merging_nodes_new = set(map(lambda x: node2new_node[x], merging_nodes))
        retain_node_new = node2new_node[retain_node]
        # if snt_id == "1012500800170":
        #     print (merging_nodes, retain_node)
        #     print (merging_nodes_new)
        #     print ()
        try: 
            erg_digraphs_re.dmrs_dg.merge_nodes(merging_nodes_new, retain_node_new)
        except:
            print (snt_id)
            # print (set(map(lambda x: node2new_node[x], merging_nodes)))
            erg_digraphs_re.draw_dmrs(name = '{}'.format(rewrite_type))
            erg_digraphs.draw_dmrs(name = '{}_b4'.format(rewrite_type))
            input()
            
        for node in node2new_node:
            if node2new_node[node] in merging_nodes_new:
                node2new_node[node] = retain_node_new
        for node in merging_nodes_new:
            node2new_node[node] = retain_node_new
    
        # if snt_id == "1012500800170":
        #     for node in node2new_node:
        #         if node2new_node[node] != node:
        #             print (node, node2new_node[node])
        #     print ()

    # draw after
    # if rewrite_type == 'modal' and merge_list and True:
    #     erg_digraphs_re.draw_dmrs(name = '{}2'.format(rewrite_type))
    #     input()
    return erg_digraphs_re


def to_json(targ_export_dir, targ_data_dir, unk2pos, save_deriv = "no", sample_only = "no"):
    
    os.makedirs(targ_data_dir, exist_ok = True)
    os.makedirs(targ_data_dir, exist_ok = True)
    sys.stderr.write("Writing DMRS as json\nfrom: {}/\ninto: {}/\n".format(targ_export_dir, targ_data_dir))

    workers_args = [(targ_export_dir, targ_data_dir, unk2pos, save_deriv, sample_only, worker_id) for worker_id in range(10)]
    with Pool(10) as p:
        data_info_workers = list(p.imap(to_json_worker, workers_args))
    
    id2file_path = defaultdict()
    err2cnt = Counter()
    pred2cnt = Counter()
    for data_info_worker in data_info_workers:
        id2file_path = id2file_path | data_info_worker["id2file_path"]
        err2cnt = err2cnt + data_info_worker["err2cnt"]
        pred2cnt = pred2cnt + data_info_worker["pred2cnt"]


    targ_data_info_dir = os.path.join(targ_data_dir, "info")
    os.makedirs(targ_data_info_dir, exist_ok = True)

    with open(os.path.join(targ_data_info_dir, "err2cnt.txt"), "w") as f:
        f.write(str(err2cnt))
    with open(os.path.join(targ_data_info_dir, "pred2cnt.txt"), "w") as f:
        for pred, cnt in pred2cnt.most_common():
            f.write("{}\t{}\n".format(pred, str(cnt)))
    with open(os.path.join(targ_data_info_dir, "id2file_path.json"), "w") as f:
        json.dump(id2file_path, f)

def to_json_worker(args):
    '''
    1. Normalize lnk of each DMRS graph
        - 'predicate': 'pron<0:2>' => 'predicate': 'pron'; 'lnk': {'from': 0, 'to': 2}
    2. Save the DMRS graph and HPSG derivation of every sentence as json
        - exports/export#/uio/wikiwoods/1212/export/20910 => data/preprocessed/#_20910.json
        - create the directory from your code, using os.makedirs(<path>, exist_ok = True)
    '''
    targ_export_dir, targ_data_dir, unk2pos, save_deriv, sample_only, worker_id = args


    lnk_regex = re.compile(r'(.+)<(\d+):(\d+)>')# patterm checking
    
    start_process = False
    pred2cnt = Counter()
    err2cnt = Counter()
    start = False
    
    no_instance = 0
    id2file_path = defaultdict()
    
    for root, dirs, files in os.walk(targ_export_dir):
        path = root.split(os.sep)
        if files:
            no_files = len(files)
            for file_idx, file in enumerate(files):
                id2instance = defaultdict(defaultdict)
                export_no = root.split("/")[1][-1]
                if export_no != str(worker_id):
                    continue
                # if file == '09490.gz':
                #     start = True
                # if not start:
                #     continue
                if VERBOSE:
                    sys.stderr.write("processing {}".format(os.path.join(root, file)))

                if file_idx%(no_files/20) == 0:
                    print ("worker {}: {}% done".format(worker_id, file_idx/no_files * 100))
                
                targ_filename = "{}_{}.json".format(export_no, file[:-3])


                with gzip.open(os.path.join(root, file), "rb") as f:
                    text = f.read().decode('utf-8')
                    structs = text.split("\4")
                    
                    # for each sentence, we turn mrs to dmrs, and align the derivation and syntax tree to the dmrs
                    for idx, struct in enumerate(structs):
                        try:
                            id_snt, anc, yyinput, drv, cat, mrs, eds, dmrs_noScope, _ = struct.split("\n\n")
                        except:
                            if VERBOSE:
                                sys.stderr.write("wrongly formatted instance:")
                                if VERBOSE:
                                    print (struct)
                            continue
                        id_snt = id_snt.strip()
                        id_snt = id_snt.split("\n")[0]
                        snt_id = id_snt.split("]", 1)[0][1:]
                        snt = id_snt.split("`")[1][:-1]
                        # print (snt_id, snt)
                        
                        if (snt_id != '1000103100080'):
                            pass
                            
                        try:
                            mrs = mrs.replace('CARG: *TOP*', 'CARG: "*TOP*"')
                            dmrs = from_mrs(simplemrs.decode(mrs))
                            dmrs_json = dmrsjson.to_dict(dmrs)
                        except MRSSyntaxError as e:
                            err2cnt['mrs-X->dmrs_json'] += 1
                            if VERBOSE:
                                print (snt)
                                print (mrs)
                            # print (simplemrs.decode(mrs))
                            
                        if save_deriv:
                            try:
                                syn_tree = SExpr.parse(cat).data
                            except Exception as e:
                                err2cnt['-X->syn_tree'] += 1

                            try:
                                deriv = from_string(drv)
                            except DerivationSyntaxError as e:
                                err2cnt['-X->deriv'] += 1
                            
                        if not dmrs_json or (save_deriv and not (syn_tree and deriv)):
                            continue
                            
                        # remove dmrs with only one node or no edges
                        if not dmrs_json['links'] or len(dmrs_json['nodes']) == 1:
                            err2cnt['1node/0edge dmrs'] += 1
                            continue
                        if not dmrs.top:
                            err2cnt['dmrs no top'] += 1
                            continue
                            
                        # normalize lnk info
                        for node in dmrs_json['nodes']:
                            re_match = lnk_regex.match(node['predicate'])
                            
                            if re_match:
                                pred_lnk_split = node['predicate'].split("<")
                                node['predicate'] = pred_lnk_split[0]
                                lnk_split = pred_lnk_split[1].split(":")
                                anc_from = int(lnk_split[0])
                                lnk_split = lnk_split[1].split(">")
                                anc_to = int(lnk_split[0])
                                node['lnk'] = {'from':anc_from,'to':anc_to}
                                
                                
                            norm_pred = normalize_pred(node['predicate'], unk2pos)
                            # node['predicate'] = pred
                            node['predicate'] = norm_pred
                            pred2cnt[node['predicate']] += 1
                            
                        # create erg_digraph object
                        erg_digraphs = dg_util.Erg_DiGraphs()
                        draw = False
                        # uncomment this line for debugging
                        # draw = True
                        
                        try:
                            erg_digraphs.init_dmrsjson(dmrs_json, minimal_prop = True)
                        except ExtraPredicateError as e:
                            # sys.stderr.write("A DMRS node predicate is '_'; discarding the instance\n")
                            err2cnt['dmrs pred:"_"'] += 1
                            continue
                        except DMRSNoLnkError as e:
                            # sys.stderr.write("A DMRS node has no lnk; discarding the instance\n")
                            err2cnt['dmrs node no lnk'] += 1
                            continue
                        except DisconnectedDMRSError as e:
                            # sys.stderr.write("A DMRS is weakly disconnected\n")
                            err2cnt['dmrs disconnected'] += 1
                            continue
                            
                        if save_deriv:
                            try:
                                erg_digraphs.init_erg_deriv(deriv, draw = draw)
                            except DerivationBranchingError as e:
                                if VERBOSE:
                                    sys.stderr.write("More than two daughters for a node in the derivaiton; discarding the instance\n")
                                err2cnt['deriv_>2dgtrs'] += 1
                                continue
                        
                        erg_digraphs.init_snt(snt)
                        
                        if save_deriv:
                            erg_digraphs.init_syn_tree(syn_tree, draw = draw)
                            # propagate anchors from terminal to nonterminals in derivation tree
                            deriv_root_node = -1
                            propagate_anchors(erg_digraphs.deriv_dg, deriv_root_node)
                            # align derivation and syntax tree and dmrs
                            syn_tree_root_node = 0
                            deriv_root_dgtr = list(erg_digraphs.deriv_dg.out_edges(-1))[0][1]
                            syn_tree_anchors_from_deriv(erg_digraphs.syn_tree_dg, syn_tree_root_node, erg_digraphs.deriv_dg, deriv_root_dgtr)
                        
                        # cleanse dmrs
                        erg_digraphs_re = erg_digraphs
                        # erg_digraphs_re = dmrs_rewrite(snt_id, snt, erg_digraphs_re, rewrite_type = 'modal')
                        # erg_digraphs_re = dmrs_rewrite(snt_id, snt, erg_digraphs_re, rewrite_type = 'nominalization')
                        # erg_digraphs_re = dmrs_rewrite(snt_id, snt, erg_digraphs_re, rewrite_type = 'eventuality')
                        # erg_digraphs_re = dmrs_rewrite(snt_id, snt, erg_digraphs_re, rewrite_type = 'compound')
                        
                        id2instance[snt_id]['snt'] = erg_digraphs_re.snt
                        id2instance[snt_id]['id'] = snt_id
                        id2instance[snt_id]['dmrs'] = nx.readwrite.json_graph.node_link_data(erg_digraphs_re.dmrs_dg)
                        if save_deriv:
                            id2instance[snt_id]['deriv'] = nx.readwrite.json_graph.node_link_data(erg_digraphs_re.deriv_dg)
                            
                        id2file_path[snt_id] = targ_filename
                        no_instance += 1
                
                with open(os.path.join(targ_data_dir, targ_filename), "w") as f:
                    json.dump(id2instance, f)
                if VERBOSE:
                    sys.stderr.write(str(err2cnt)+"\n")
                # input()
                if sample_only and id2instance: break
            if sample_only and id2instance: break

    return {"err2cnt": err2cnt, "id2file_path": id2file_path, "pred2cnt": pred2cnt}
                        
        
def main(targ_export_dir, targ_data_dir, erg_dir, save_deriv, sample_only, verbose):
    
    save_deriv, sample_only, verbose = [False if arg == 'no' else True if arg == 'yes' else 'err' for arg in [save_deriv, sample_only, verbose]]

    if 'err' in [save_deriv, sample_only, verbose]:
        sys.stderr.write("invalid argument\n")

    else:
        VERBOSE = verbose
        unk2pos_path = os.path.join(erg_dir, "unk2pos.json")
        with open(unk2pos_path) as f:
            unk2pos = json.load(f)
        # extract(ww1212_dir, targ_export_dir)
        to_json(targ_export_dir, targ_data_dir, unk2pos, save_deriv, sample_only)
        # to_json(targ_export_dir, targ_data_dir)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--targ_export_dir', default=None, help='path to target export directory')
    parser.add_argument('--targ_data_dir', default=None, help='path to target data directory')
    parser.add_argument('--erg_dir', default='erg', help='path to directory of English Resource Grammar knowledge')
    # parser.add_argument('log_dir', default='log', help='path to directory of English Resource Grammar knowledge')
    parser.add_argument('--save_deriv', default="no", help='save also derivation trees')
    parser.add_argument('--sample_only', default="no", help='preprocess on a small subset of data first?')
    parser.add_argument('--verbose', default="no", help='enable verbose mode')
    args = parser.parse_args()
    main(
        args.targ_export_dir,
        args.targ_data_dir,
        args.erg_dir,
        args.save_deriv,
        args.sample_only,
        args.verbose
    )
    
'''
module_name, package_name, ClassName, method_name, ExceptionName, function_name, GLOBAL_CONSTANT_NAME, global_var_name, instance_var_name, function_parameter_name, local_var_name.
'''
                
    