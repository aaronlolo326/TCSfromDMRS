import argparse
from collections import namedtuple, Counter, defaultdict
from parse_config import ConfigParser

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

from pprint import pprint
import json
import os
import re

from src import util
from utils import get_transformed_info
from transform.tcs_transform import TruthConditions

from scipy.stats import spearmanr

oovs = set()
vs = set()

def _predarg2ix(pred, arg, predarg2ix):
    if pred + "@" + arg not in predarg2ix:
        # print (pred + "@" + arg)
        if pred + "@" + "NonARG" in predarg2ix:
            oovs.add(pred + "@" + arg)
            return predarg2ix[pred + "@" + "NonARG"]
        else:
            pass
            # print (pred, arg)
    else:
        vs.add(pred + "@" + arg)
        return predarg2ix[pred + "@" + arg]

def get_pos2lemma2pred2cnt(pred_func2cnt, pred_func2ix):
    pred_funcs = list(pred_func2ix.keys())
    pos2lemma2pred2cnt = defaultdict(lambda: defaultdict(Counter))
    for pred_func in pred_funcs:
        pred = pred_func.rsplit("@", 1)[0]
        pred_lemma, pred_pos = util.get_lemma_pos(pred)
        # if pred_pos == 'n':
            # noun_lemmas_dmrs2cnt[pred_lemma] = content_pred2cnt[pred]
        pos2lemma2pred2cnt[pred_pos][pred_lemma][pred] += pred_func2cnt[pred_func2ix[pred_func]]
    return pos2lemma2pred2cnt

# def prepare_hyp(config, hyp_dir, hyp_file, pos2lemma2pred2cnt, min_hyp_pred_pair_freq = None):
    
#     MIN_CONTENT_PRED_FREQ = config["data_loader"]["args"]["min_content_pred_freq"]
#     MIN_PRED_FUNC_FREQ = config["data_loader"]["args"]["min_pred_func_freq"]
#     # MIN_HYP_PAIR_FREQ = 10000
#     if not min_hyp_pred_pair_freq:
#         min_hyp_pred_pair_freq = MIN_PRED_FUNC_FREQ
#     print ("hyp_file: {}".format(hyp_file))
#     print ("min_hyp_pred_pair_freq: {}".format(min_hyp_pred_pair_freq))

#     hyp_wn_path = os.path.join(hyp_dir, hyp_file)

#     noun_lemmas_wn = set()
#     hyp_pairs = []
#     lemma2pred2cnt = pos2lemma2pred2cnt['n']
#     with open(hyp_wn_path) as f:
#         line = f.readline()
#         while line:
#             hypos, hypers = line.strip().split("\t")
#             hypos = hypos.split(", ")
#             hypers = hypers.split(", ")
#             noun_lemmas_wn.update(set(hypos))
#             noun_lemmas_wn.update(set(hypers))
#             common_hypos = filter(lambda x: x in lemma2pred2cnt and lemma2pred2cnt[x].most_common()[0][1] >= min_hyp_pred_pair_freq, hypos)
#             common_hypers = filter(lambda x: x in lemma2pred2cnt and lemma2pred2cnt[x].most_common()[0][1] >= min_hyp_pred_pair_freq, hypers)
#             hyp_pairs.extend(product(common_hypos, common_hypers))
#             line = f.readline()

#     hyp_pred_pairs = []
#     for hyp_pair in hyp_pairs:
#         hyp_pred_pairs.append(tuple(map(lambda x: lemma2pred2cnt[x].most_common()[0][0], hyp_pair)))

#     # print (len(noun_lemmas_wn))

#     common_lemmas = noun_lemmas_wn.intersection(lemma2pred2cnt.keys())
#     print ("#common_lemmas:", len(common_lemmas))
#     print ("#hyp_paris:", len(hyp_pairs))
#     print ()
#     # pprint (hyp_pred_pairs)
#     # pprint (common_lemmas)
#     # print ()
#     # for k, v in lemma2pred2cnt.items():
#     #     print (k, v)
    
#     return hyp_pred_pairs

def prepare_relpron(relpron_dir, encoder_arch_type, svo, pos2lemma2pred2cnt, pred_func2cnt, pred2ix, predarg2ix, pred_func2ix, min_pred_func_freq, sample_only):

    relpron_path = {
        "dev": os.path.join(relpron_dir, "relpron.dev"),
        "test": os.path.join(relpron_dir, "relpron.test")
    }
    word2pred = defaultdict()
    if not sample_only:
        with open(os.path.join(relpron_dir, "word2pred_premap.json")) as f:
            word2pred = json.load(f)

    relpron_data = defaultdict(list)
    relpron_terms = defaultdict(list)
    relpron_props = defaultdict(list)
    relpron_splits = defaultdict()

    term2props_idx = defaultdict(lambda: defaultdict(list))
    for split, path in relpron_path.items():
        # term2idx = defaultdict()
        prop2idx = defaultdict()
        with open(path) as f:
            line = f.readline().strip()
            while line:
                m = re.match('^([OS]BJ) (\S+)_N: (\S+)_N that (\S+)_[VN] (\S+)_[VN]\s*$', line)
                instance = tuple(m.group(i) for i in range(1,6))
                sbj_obj, targ = instance[0], instance[1]
                if sbj_obj == 'SBJ':
                    sbj, verb, obj = instance[2:5]
                elif sbj_obj == 'OBJ':
                    obj, sbj, verb = instance[2:5]
                if targ not in relpron_terms[split]:
                    relpron_terms[split].append(targ)
                    # term2idx[targ] = len(relpron_terms[split])
                if (sbj_obj, sbj, verb, obj) not in relpron_props[split]:
                    relpron_props[split].append((sbj_obj, sbj, verb, obj))
                    prop2idx[(sbj_obj, sbj, verb, obj)] = len(relpron_props[split]) - 1
                term2props_idx[split][targ].append(prop2idx[(sbj_obj, sbj, verb, obj)])
                line = f.readline().strip()
        print ("RELPRON-{} has {}, {} terms, props".format(
            split, len(relpron_terms[split]), len(relpron_props[split])
            )
        )
        
    # filter
    mapped = defaultdict()
    term2term_pred = defaultdict(defaultdict)
    relpron_props_pred = defaultdict(list)
    # prop_idx2filter_idx = defaultdict(defaultdict)
    for split, props in relpron_props.items():
        for prop_idx, prop in enumerate(props):
            sbj_obj, sbj, verb, obj = prop
            sbj_pred = word2pred.get(sbj)
            if not sbj_pred:
                if sbj in pos2lemma2pred2cnt['n']:
                    sbj_pred = pos2lemma2pred2cnt['n'][sbj].most_common()[0][0]
            verb_pred = word2pred.get(verb)
            if not verb_pred:
                if verb in pos2lemma2pred2cnt['v']:
                    verb_pred = pos2lemma2pred2cnt['v'][verb].most_common()[0][0]
            obj_pred = word2pred.get(obj)
            if not obj_pred:
                if obj in pos2lemma2pred2cnt['n']:
                    obj_pred = pos2lemma2pred2cnt['n'][obj].most_common()[0][0]
            if sbj_pred and verb_pred and obj_pred:
                arg0s_exist = all([word + "@ARG0" in pred_func2cnt and pred_func2cnt[word + "@ARG0"] >= min_pred_func_freq for word in [sbj_pred, verb_pred, obj_pred]])
                arg12_exist = all([
                    verb_pred + "@ARG1" in pred_func2cnt and pred_func2cnt[verb_pred + "@ARG1"] >= min_pred_func_freq,
                    verb_pred + "@ARG2" in pred_func2cnt and pred_func2cnt[verb_pred + "@ARG2"] >= min_pred_func_freq
                ])
                if True or arg0s_exist and arg12_exist:
                    mapped[sbj] = sbj_pred
                    mapped[verb] = verb_pred
                    mapped[obj] = obj_pred
                    relpron_props_pred[split].append((sbj_obj, sbj_pred, verb_pred, obj_pred))
                else:
                    print (sbj_pred, verb_pred, obj_pred, arg0s_exist, arg12_exist)

            else:
                pass

    for split, terms in relpron_terms.items():
        for term_idx, term in enumerate(terms):
            term_pred = word2pred.get(term)
            if not term_pred:
                if term in pos2lemma2pred2cnt['n']:
                    term_pred = pos2lemma2pred2cnt['n'][term].most_common()[0][0]
            if term_pred:
                mapped[term] = term_pred
                arg0_exist = term_pred + "@ARG0" in pred_func2cnt and pred_func2cnt[term_pred + "@ARG0"] >= min_pred_func_freq
                if arg0_exist:
                    term2term_pred[split][term] = term_pred
                else:
                    pass
                    # print (term, term_pred)
            else:
                pass
                # print (term)

    term_pred2props_idx = defaultdict(lambda: defaultdict(list))

    trsfm = TruthConditions(*[None] * 11)
    for split in term2term_pred:
        oov = set()
        
        pred_func_nodes_ctxt_predargs_list = []
        pred_func_nodes_ctxt_args_list = []
        logic_expr_list = []
        vars_unzipped_list = []
        targ_terms = []
        for term, term_pred in term2term_pred[split].items():
            for prop_idx, props_pred in enumerate(relpron_props_pred[split]):
                sbj_obj, sbj_pred, verb_pred, obj_pred = props_pred
                targ_terms.append(term)
                # just <target>
                try:
                    pred_funcs = [
                        pred_func2ix[term_pred + "@ARG0"]
                    ]
                    vars_unzipped = [
                        [1 if sbj_obj == 'SBJ' else 3],
                        [0]
                    ]
                    ## all semantic functions
                    # pred_funcs = [
                    #     pred_func2ix[sbj_pred + "@ARG0"],
                    #     pred_func2ix[verb_pred + "@ARG0"],
                    #     pred_func2ix[obj_pred + "@ARG0"],
                    #     pred_func2ix[verb_pred + "@ARG1"],
                    #     pred_func2ix[verb_pred + "@ARG2"],
                    #     pred_func2ix[targ + "@ARG0"]
                    # ]
                    # vars_unzipped = [
                    #     [1, 2, 3, 2, 2, 1 if sbj_obj == 'SBJ' else 3],
                    #     [0, 0, 0, 1, 3, 0]
                    # ]
                    arg2ix = trsfm.arg2ix
                    if not svo:
                        if verb_pred == '_be_v_id':
                            vars_unzipped = [
                                [1],
                                [0]
                            ]
                            pred_func_nodes_ctxt_preds = [
                                [pred2ix[sbj_pred], pred2ix[obj_pred], 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0]
                            ]
                            pred_func_nodes_ctxt_args = [
                                [arg2ix["ARG0"], arg2ix["ARG0"], 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0]
                            ]
                        elif sbj_obj == 'SBJ':
                            pred_func_nodes_ctxt_preds = [
                                [pred2ix[sbj_pred], pred2ix[term_pred], pred2ix[verb_pred], pred2ix[obj_pred]],
                                [pred2ix[verb_pred], pred2ix[sbj_pred], pred2ix[term_pred], pred2ix[obj_pred]],
                                [pred2ix[obj_pred], pred2ix[sbj_pred], pred2ix[term_pred], pred2ix[verb_pred]]
                            ]
                            pred_func_nodes_ctxt_args = [
                                [arg2ix["ARG0"], arg2ix["ARG0"], arg2ix["ARG1"], arg2ix["NonARG"]],
                                [arg2ix["ARG0"], arg2ix["ARG1-rvrs"], arg2ix["ARG1-rvrs"], arg2ix["ARG2-rvrs"]],
                                [arg2ix["ARG0"], arg2ix["NonARG"], arg2ix["NonARG"], arg2ix["ARG2"]],
                            ]
                        elif sbj_obj == 'OBJ':
                            pred_func_nodes_ctxt_preds = [
                                [pred2ix[sbj_pred], pred2ix[verb_pred], pred2ix[obj_pred], pred2ix[term_pred]],
                                [pred2ix[verb_pred], pred2ix[sbj_pred], pred2ix[obj_pred], pred2ix[term_pred]],
                                [pred2ix[obj_pred], pred2ix[term_pred], pred2ix[sbj_pred], pred2ix[verb_pred]]
                            ]
                            pred_func_nodes_ctxt_args = [
                                [arg2ix["ARG0"], arg2ix["ARG1"], arg2ix["NonARG"], arg2ix["NonARG"]],
                                [arg2ix["ARG0"], arg2ix["ARG1-rvrs"], arg2ix["ARG2-rvrs"], arg2ix["ARG2-rvrs"]],
                                [arg2ix["ARG0"], arg2ix["ARG0"], arg2ix["NonARG"], arg2ix["ARG2"]],
                            ]
                    else:
                        if verb_pred == '_be_v_id':
                            vars_unzipped = [
                                [1],
                                [0]
                            ]
                            pred_func_nodes_ctxt_predargs = [
                                [_predarg2ix(sbj_pred, "ARG0", predarg2ix), _predarg2ix(obj_pred, "ARG0", predarg2ix), 0],
                                [0, 0, 0],
                                [0, 0, 0]
                            ]
                            pred_func_nodes_ctxt_args = [
                                [arg2ix["ARG0"], arg2ix["ARG0"], arg2ix["NonARG"]],
                                [0, 0, 0],
                                [0, 0, 0]
                            ]
                        elif True: #sbj_obj == 'SBJ':
                            pred_func_nodes_ctxt_predargs = [
                                [_predarg2ix(sbj_pred, "ARG0", predarg2ix), _predarg2ix(verb_pred, "ARG1", predarg2ix), _predarg2ix(obj_pred, "NonARG", predarg2ix)],
                                [_predarg2ix(verb_pred, "ARG0", predarg2ix), _predarg2ix(sbj_pred, "ARG1-rvrs", predarg2ix), _predarg2ix(obj_pred, "ARG2-rvrs", predarg2ix)],
                                [_predarg2ix(obj_pred, "ARG0", predarg2ix), _predarg2ix(sbj_pred, "NonARG", predarg2ix), _predarg2ix(verb_pred, "ARG2", predarg2ix)]
                            ]
                            pred_func_nodes_ctxt_args = [
                                [arg2ix["ARG0"], arg2ix["ARG1"], arg2ix["NonARG"]],
                                [arg2ix["ARG0"], arg2ix["ARG1-rvrs"], arg2ix["ARG2-rvrs"]],
                                [arg2ix["ARG0"], arg2ix["NonARG"], arg2ix["ARG2"]]
                            ]
                        # elif sbj_obj == 'OBJ':
                        #     if encoder_arch_type == 'MyEncoder':
                        #         pred_func_nodes_ctxt_preds = [
                        #             [pred2ix[sbj_pred], pred2ix[verb_pred], pred2ix[obj_pred]],
                        #             [pred2ix[verb_pred], pred2ix[sbj_pred], pred2ix[obj_pred]],
                        #             [pred2ix[obj_pred], pred2ix[sbj_pred], pred2ix[verb_pred]]
                        #         ]
                        #     elif encoder_arch_type == 'PASEncoder':
                        #         pred_func_nodes_ctxt_predargs = [
                        #             [_predarg2ix(sbj_pred, "ARG0"], _predarg2ix(verb_pred, "ARG1"], _predarg2ix(obj_pred, "NonARG"]],
                        #             [_predarg2ix(verb_pred, "ARG0"], _predarg2ix(sbj_pred, "ARG1-rvrs"], _predarg2ix(obj_pred, "ARG2-rvrs"]],
                        #             [_predarg2ix(obj_pred, "ARG0"], _predarg2ix(sbj_pred, "NonARG"], _predarg2ix(verb_pred, "ARG2"]]
                        #         ]
                        #     pred_func_nodes_ctxt_args = [
                        #         [arg2ix["ARG0"], arg2ix["ARG1"], 0],
                        #         [arg2ix["ARG0"], arg2ix["ARG1-rvrs"], arg2ix["ARG2-rvrs"]],
                        #         [arg2ix["ARG0"], 0, arg2ix["ARG2"]]
                        #     ]
                except KeyError as e:
                    if sample_only:
                        # print (e)
                        continue
                    else:
                        # print (e)
                        oov.add(str(e))
                        p = str(e)[1:-1].split("@")[0]
                        a, b = None, None
                        if p + "@ARG0" in predarg2ix:
                            a = _predarg2ix(p, "ARG0", predarg2ix)
                        if p + "@NonARG" in predarg2ix:
                            b = _predarg2ix(p, "NonARG", predarg2ix)
                        if b == None:
                            pass
                            # print (p, a, b)
                        # input ()
                        continue
                pred_func_nodes_ctxt_predargs_list.append(pred_func_nodes_ctxt_predargs)
                pred_func_nodes_ctxt_args_list.append(pred_func_nodes_ctxt_args)
                logic_expr_list.append(pred_funcs)
                vars_unzipped_list.append(vars_unzipped)
    
        relpron_splits[split] = {
            "pred_func_nodes_ctxt_predargs_list": pred_func_nodes_ctxt_predargs_list,
            "pred_func_nodes_ctxt_args_list": pred_func_nodes_ctxt_args_list,
            "logic_expr_list": logic_expr_list,
            "vars_unzipped_list": vars_unzipped_list,
            "labels": term2props_idx[split], #term2filter_props_idx[split],
            "full_props": relpron_props_pred[split],#filtered_relpron_props[split],
            "terms": targ_terms
        }
        print ("Finally, RELPRON-{} has {}, {}, {} terms, props, instances".format(
            split, len(relpron_terms[split]), len(relpron_props_pred[split]), len(pred_func_nodes_ctxt_predargs_list)
            )
        )
        

    return relpron_splits, mapped

def prepare_gs(gs_dir, encoder_arch_type, pos2lemma2pred2cnt, pred_func2cnt, pred2ix, predarg2ix, pred_func2ix, min_pred_func_freq, sample_only, gs_year = "2011"):

    gs_path = os.path.join(gs_dir, "GS{}data.txt".format(gs_year))
    word2pred = None
    with open(os.path.join(gs_dir, "word2pred_premap.json")) as f:
        word2pred = json.load(f)

    gs_data = list()
    gs_unfilter_inst = set()
    parti2svol2score = defaultdict(defaultdict)
    with open(gs_path) as f:
        line = f.readline().strip()
        line = f.readline().strip()
        while line:
            # participant verb subject object landmark input hilo # 2011, 2013
            # participant20 provide family home supply 4 HIGH
            parti, verb, sbj, obj, landmark, score, hilo = line.split(" ")
            score = int(score)
            gs_data.append((verb, sbj, obj, landmark, score, hilo))
            gs_unfilter_inst.add((verb, sbj, obj, landmark))
            parti2svol2score[parti][(verb, sbj, obj, landmark)] = score
            line = f.readline().strip()

    # compute inter-annotator agreement
    iaa_sep = 0
    iaa_avg = 0
    for parti, svol2score in parti2svol2score.items():
        landmark_truths_sep = []
        landmark_score_sep = []
        landmark_truths_avg = []
        landmark_score_avg = []
        for svol, score in svol2score.items():
            scores_other = [parti2svol2score[parti_other][svol] for parti_other in parti2svol2score if svol in parti2svol2score[parti_other] and parti_other != parti] # and parti_other != parti
            avg_scores_other = sum(scores_other) / len(scores_other)
            landmark_score_sep.extend(scores_other)
            landmark_truths_sep.extend([score] * len(scores_other))
            landmark_score_avg.append(avg_scores_other)
            landmark_truths_avg.append(score)
        rho_sep = spearmanr(landmark_score_sep, landmark_truths_sep)[0]
        rho_avg = spearmanr(landmark_score_avg, landmark_truths_avg)[0]
        iaa_sep += rho_sep
        iaa_avg += rho_avg
    iaa_sep /= len(parti2svol2score)
    iaa_avg /= len(parti2svol2score)
    print ("GS{} iaa_sep: {}; iaa_avg: {}".format(gs_year, iaa_sep, iaa_avg))

    # pprint (gs_unfilter_inst)
    gs_svo_ix2landmark2scores = defaultdict(lambda: defaultdict(list))
    gs_svo_ix2landmark2logic_expr = defaultdict(lambda: defaultdict(list))
    gs_svo2ix = defaultdict()
    gs_ix2svo = defaultdict()

    mapped = defaultdict()
    preds2svol = defaultdict()

    pred_func_nodes_ctxt_predargs_list = []
    pred_func_nodes_ctxt_args_list = []
    vars_unzipped_list = []

    trsfm = TruthConditions(*[None] * 11)

    print ("GS{} has {} instances".format(gs_year, len(gs_unfilter_inst)))
    for verb, sbj, obj, landmark, score, hilo in gs_data:
        # verb_pred, sbj_pred, obj_pred, landmark_pred = None, None, None, None
        sbj_pred = word2pred.get(sbj)
        if not sbj_pred:
            if sbj in pos2lemma2pred2cnt['n']:
                sbj_pred = pos2lemma2pred2cnt['n'][sbj].most_common()[0][0]
        verb_pred = word2pred.get(verb)
        if not verb_pred:
            if verb in pos2lemma2pred2cnt['v']:
                verb_pred = pos2lemma2pred2cnt['v'][verb].most_common()[0][0]
        obj_pred = word2pred.get(obj)
        if not obj_pred:
            if obj in pos2lemma2pred2cnt['n']:
                obj_pred = pos2lemma2pred2cnt['n'][obj].most_common()[0][0]
        landmark_pred = word2pred.get(landmark)
        if not landmark_pred:
            if landmark in pos2lemma2pred2cnt['v']:
                landmark_pred = pos2lemma2pred2cnt['v'][landmark].most_common()[0][0]
        if not all([verb_pred, sbj_pred, obj_pred, landmark_pred]):
            # print (verb, sbj, obj, landmark)
            continue
        else:
            # ensure arg0 of nouns and arg0,1,2 or verbs are not OOV
            arg0s_exist = all([word + "@ARG0" in pred_func2cnt and pred_func2cnt[word + "@ARG0"] >= min_pred_func_freq for word in [verb_pred, sbj_pred, obj_pred, landmark_pred]])
            arg12_exist = all([
                verb_pred + "@ARG1" in pred_func2cnt and pred_func2cnt[verb_pred + "@ARG1"] >= min_pred_func_freq,
                verb_pred + "@ARG2" in pred_func2cnt and pred_func2cnt[verb_pred + "@ARG2"] >= min_pred_func_freq,
                landmark_pred + "@ARG1" in pred_func2cnt and pred_func2cnt[landmark_pred + "@ARG1"] >= min_pred_func_freq,
                landmark_pred + "@ARG2" in pred_func2cnt and pred_func2cnt[landmark_pred + "@ARG2"] >= min_pred_func_freq
            ])
            if arg0s_exist and arg12_exist:
                mapped[verb] = verb_pred
                mapped[sbj] = sbj_pred
                mapped[obj] = obj_pred
                mapped[landmark] = landmark_pred
                if not (verb_pred, sbj_pred, obj_pred) in gs_svo2ix:
                    gs_svo2ix[(verb_pred, sbj_pred, obj_pred)] = len(gs_svo2ix)
                    # just <target>
                    vars_unzipped = [
                        [1, 1, 1],
                        [0, 2, 3]
                    ]
                    ## all semantic functions
                    # pred_funcs = [
                    #     pred_func2ix[sbj + "@ARG0"],
                    #     pred_func2ix[verb + "@ARG0"],
                    #     pred_func2ix[obj + "@ARG0"],
                    #     pred_func2ix[verb + "@ARG1"],
                    #     pred_func2ix[verb + "@ARG2"],
                    #     pred_func2ix[targ + "@ARG0"]
                    # ]
                    # vars_unzipped = [
                    #     [1, 2, 3, 2, 2, 1 if sbj_obj == 'SBJ' else 3],
                    #     [0, 0, 0, 1, 3, 0]
                    # ]
                    arg2ix = trsfm.arg2ix
                    pred_func_nodes_ctxt_preds, pred_func_nodes_ctxt_predargs, pred_func_nodes_ctxt_args = None, None, None
                    try:
                        pred_func_nodes_ctxt_predargs = [
                            [_predarg2ix(verb_pred, "ARG0", predarg2ix), _predarg2ix(sbj_pred, "ARG1-rvrs", predarg2ix), _predarg2ix(obj_pred, "ARG2-rvrs", predarg2ix)],
                            [_predarg2ix(sbj_pred, "ARG0", predarg2ix), _predarg2ix(verb_pred, "ARG1", predarg2ix), _predarg2ix(obj_pred, "NonARG", predarg2ix)],
                            [_predarg2ix(obj_pred, "ARG0", predarg2ix), _predarg2ix(sbj_pred, "NonARG", predarg2ix), _predarg2ix(verb_pred, "ARG2", predarg2ix)]
                        ]
                        pred_func_nodes_ctxt_predargs_list.append(pred_func_nodes_ctxt_predargs)
                        pred_func_nodes_ctxt_args = [
                            [arg2ix["ARG0"], arg2ix["ARG1-rvrs"], arg2ix["ARG2-rvrs"]],
                            [arg2ix["ARG0"], arg2ix["ARG1"], arg2ix["NonARG"]],
                            [arg2ix["ARG0"], arg2ix["NonARG"], arg2ix["ARG2"]]
                        ]
                    except KeyError as e:
                        if sample_only:
                            continue
                        else:
                            print (e)
                            input ()
                    pred_func_nodes_ctxt_args_list.append(pred_func_nodes_ctxt_args)
                    vars_unzipped_list.append(vars_unzipped)
                ix = gs_svo2ix[(verb_pred, sbj_pred, obj_pred)]
                gs_ix2svo[ix] = (verb_pred, sbj_pred, obj_pred)
                
                gs_svo_ix2landmark2scores[ix][landmark_pred].append(score)
                pred_funcs = (
                    pred_func2ix[landmark_pred + "@ARG0"],
                    pred_func2ix[landmark_pred + "@ARG1"],
                    pred_func2ix[landmark_pred + "@ARG2"]
                )
                gs_svo_ix2landmark2logic_expr[ix][landmark_pred] = pred_funcs
                preds2svol["-".join([verb_pred, sbj_pred, obj_pred, landmark_pred])] = (verb, sbj, obj, landmark)

            else:
                pass
                # print (verb_pred, sbj_pred, obj_pred, landmark_pred, arg0s_exist, arg12_exist)
            
    print ("after filtering OOV: GS{} has {} SVO instances".format(gs_year, len(gs_svo_ix2landmark2scores)))
    gs_eval = {
        "svo_ix2landmark2scores": gs_svo_ix2landmark2scores,
        "ix2svo": gs_ix2svo,
        "pred_func_nodes_ctxt_predargs_list": pred_func_nodes_ctxt_predargs_list,
        "pred_func_nodes_ctxt_args_list": pred_func_nodes_ctxt_args_list,
        "svo_ix2landmark2logic_expr": gs_svo_ix2landmark2logic_expr,
        "vars_unzipped_list": vars_unzipped_list,
    }

    return gs_eval, mapped, preds2svol


def prepare_gs2012(gs_dir, encoder_arch_type, pos2lemma2pred2cnt, pred_func2cnt, pred2ix, predarg2ix, pred_func2ix, min_pred_func_freq, sample_only, name = "GS2012"):

    gs_path = os.path.join(gs_dir, "{}data.txt".format(name))
    word2pred = None
    with open(os.path.join(gs_dir, "word2pred_premap.json")) as f:
        word2pred = json.load(f)

    gs_data = list()
    gs_unfilter_inst = set()
    parti2svol2score = defaultdict(defaultdict)
    with open(gs_path) as f:
        line = f.readline().strip()
        line = f.readline().strip()
        while line:
            # sentence_id annotator_id adj_subj subj landmark verb adj_obj obj annotator_score # 2012
            ## wrong order of verb and landmark?
            # participant20 provide family home supply 4 HIGH
            _, parti, adj_sbj, sbj, verb, landmark, adj_obj, obj, score = line.split(" ")
            score = int(score)
            gs_data.append((verb, adj_sbj, sbj, adj_obj, obj, landmark, score))
            gs_unfilter_inst.add((verb, adj_sbj, sbj, adj_obj, obj, landmark))
            parti2svol2score[parti][(verb, adj_sbj, sbj, adj_obj, obj, landmark)] = score
            line = f.readline().strip()


    # compute inter-annotator agreement
    iaa_sep = 0
    iaa_avg = 0
    for parti, svol2score in parti2svol2score.items():
        landmark_truths_sep = []
        landmark_score_sep = []
        landmark_truths_avg = []
        landmark_score_avg = []
        for svol, score in svol2score.items():
            scores_other = [parti2svol2score[parti_other][svol] for parti_other in parti2svol2score if svol in parti2svol2score[parti_other] and parti_other != parti] # and parti_other != parti
            avg_scores_other = sum(scores_other) / len(scores_other)
            landmark_score_sep.extend(scores_other)
            landmark_truths_sep.extend([score] * len(scores_other))
            landmark_score_avg.append(avg_scores_other)
            landmark_truths_avg.append(score)
        rho_sep = spearmanr(landmark_score_sep, landmark_truths_sep)[0]
        rho_avg = spearmanr(landmark_score_avg, landmark_truths_avg)[0]
        iaa_sep += rho_sep
        iaa_avg += rho_avg
    iaa_sep /= len(parti2svol2score)
    iaa_avg /= len(parti2svol2score)
    print ("{} iaa_sep: {}; iaa_avg: {}".format(name, iaa_sep, iaa_avg))
    # pprint (gs_unfilter_inst)
    gs_asvao_ix2landmark2scores = defaultdict(lambda: defaultdict(list))
    gs_asvao_ix2landmark2logic_expr = defaultdict(lambda: defaultdict(list))
    gs_asvao2ix = defaultdict()
    gs_ix2asvao = defaultdict()

    mapped = defaultdict()
    preds2asvaol = defaultdict()

    pred_func_nodes_ctxt_predargs_list = []
    pred_func_nodes_ctxt_args_list = []
    vars_unzipped_list = []

    trsfm = TruthConditions(*[None] * 11)

    print ("{} has {} instances".format(name, len(gs_unfilter_inst)))
    for verb, adj_sbj, sbj, adj_obj, obj, landmark, score in gs_data:
        # verb_pred, sbj_pred, obj_pred, landmark_pred = None, None, None, None
        sbj_pred = word2pred.get(sbj)
        if not sbj_pred:
            if sbj in pos2lemma2pred2cnt['n']:
                sbj_pred = pos2lemma2pred2cnt['n'][sbj].most_common()[0][0]
        verb_pred = word2pred.get(verb)
        if not verb_pred:
            if verb in pos2lemma2pred2cnt['v']:
                verb_pred = pos2lemma2pred2cnt['v'][verb].most_common()[0][0]
        obj_pred = word2pred.get(obj)
        if not obj_pred:
            if obj in pos2lemma2pred2cnt['n']:
                obj_pred = pos2lemma2pred2cnt['n'][obj].most_common()[0][0]
        landmark_pred = word2pred.get(landmark)
        if not landmark_pred:
            if landmark in pos2lemma2pred2cnt['v']:
                landmark_pred = pos2lemma2pred2cnt['v'][landmark].most_common()[0][0]
        adj_sbj_pred = word2pred.get(adj_sbj)
        # if adj_sbj == "second":
        #     print (adj_sbj_pred)
        if not adj_sbj_pred:
            if adj_sbj in pos2lemma2pred2cnt['a'] and adj_sbj != "second":
                adj_sbj_pred = pos2lemma2pred2cnt['a'][adj_sbj].most_common()[0][0]
        adj_obj_pred = word2pred.get(adj_obj)
        # if adj_obj == "second":
        #     print (adj_obj_pred)
        if not adj_obj_pred:
            if adj_obj in pos2lemma2pred2cnt['a'] and adj_obj != "second":
                adj_obj_pred = pos2lemma2pred2cnt['a'][adj_obj].most_common()[0][0]
        if not all([verb_pred, sbj_pred, obj_pred, landmark_pred, adj_sbj_pred, adj_obj_pred]) and adj_obj != "second":
            pass
            # print (verb, verb_pred, sbj, sbj_pred, obj, obj_pred, landmark, landmark_pred, adj_sbj, adj_sbj_pred, adj_obj, adj_obj_pred)
            continue
        else:
            # ensure arg0 of nouns and arg0,1,2 or verbs are not OOV
            verb_adj_preds = {
                "_elect_v_1", "_limit_v_1", "_register_v_1", #GS2012
                "_qualify_v_for", "_disable_v_1", "_skim_v_1", "_color_v_1", "_grit_v_1" #KS2013
            }
            noun_adj_preds = {
                "_parish_n_1", "u_raffle_n", #GS2012
                "_front_n_1", "_rugby_n_1", "_oven_n_1", "_fish_v_1", "_weave_v_1" #KS2013 u_alkaloid_a, 
            }
            arg12_missed = {"_drip_v_cause", "u_demilitarized_v", "u_entangled_v"}
            landmark_pred2sub = {"_drip_v_cause": "_drop_v_cause", "u_demilitarized_v": "_disband_v_cause", "u_entangled_v": "_tangle_v_1"}
            adj_sbj_arg = "ARG1"
            adj_obj_arg = "ARG1"
            if adj_sbj_pred in verb_adj_preds:
                adj_sbj_arg = "ARG2"
            if adj_obj_pred in verb_adj_preds:
                adj_obj_arg = "ARG2"
            if adj_sbj_pred in noun_adj_preds:
                adj_sbj_arg = "NonARG"
            if adj_obj_pred in noun_adj_preds:
                adj_obj_arg = "NonARG"
            arg0s_exist = all([word + "@ARG0" in pred_func2cnt and pred_func2cnt[word + "@ARG0"] >= min_pred_func_freq for word in [landmark_pred]]) # verb_pred, sbj_pred, obj_pred,  adj_sbj_pred, adj_obj_pred
            arg12_exist = all([
                # verb_pred + "@ARG1" in pred_func2cnt and pred_func2cnt[verb_pred + "@ARG1"] >= min_pred_func_freq,
                # verb_pred + "@ARG2" in pred_func2cnt and pred_func2cnt[verb_pred + "@ARG2"] >= min_pred_func_freq,
                landmark_pred + "@ARG1" in pred_func2cnt and pred_func2cnt[landmark_pred + "@ARG1"] >= min_pred_func_freq,
                landmark_pred + "@ARG2" in pred_func2cnt and pred_func2cnt[landmark_pred + "@ARG2"] >= min_pred_func_freq,
            ])
            # adj1_exist = all([
            #     adj_sbj_pred + "@{}".format(adj_sbj_arg) in pred_func2cnt and pred_func2cnt[adj_sbj_pred + "@{}".format(adj_sbj_arg)] >= min_pred_func_freq,
            #     adj_obj_pred + "@{}".format(adj_obj_arg) in pred_func2cnt and pred_func2cnt[adj_obj_pred + "@{}".format(adj_obj_arg)] >= min_pred_func_freq
                 
            # ])

            if arg0s_exist and (arg12_exist or landmark_pred in arg12_missed):# and adj1_exist:
                mapped[verb] = verb_pred
                mapped[sbj] = sbj_pred
                mapped[obj] = obj_pred
                mapped[landmark] = landmark_pred
                mapped[adj_sbj] = adj_sbj_pred
                mapped[adj_obj] = adj_obj_pred
                if not (verb_pred, adj_sbj_pred, sbj_pred, adj_obj_pred, obj_pred) in gs_asvao2ix:
                    gs_asvao2ix[(verb_pred, adj_sbj_pred, sbj_pred, adj_obj_pred, obj_pred)] = len(gs_asvao2ix)
                    # just <target>
                    vars_unzipped = [
                        [1, 1, 1],
                        [0, 2, 3]
                    ]
                    ## all semantic functions
                    # pred_funcs = [
                    #     pred_func2ix[sbj + "@ARG0"],
                    #     pred_func2ix[verb + "@ARG0"],
                    #     pred_func2ix[obj + "@ARG0"],
                    #     pred_func2ix[verb + "@ARG1"],
                    #     pred_func2ix[verb + "@ARG2"],
                    #     pred_func2ix[targ + "@ARG0"]
                    # ]
                    # vars_unzipped = [
                    #     [1, 2, 3, 2, 2, 1 if sbj_obj == 'SBJ' else 3],
                    #     [0, 0, 0, 1, 3, 0]
                    # ]
                    arg2ix = trsfm.arg2ix
                    pred_func_nodes_ctxt_preds, pred_func_nodes_ctxt_predargs, pred_func_nodes_ctxt_args = None, None, None
                    try:
                        if adj_obj == "second" and adj_obj_pred == None:
                                pred_func_nodes_ctxt_predargs = [
                                [_predarg2ix(verb_pred, "ARG0", predarg2ix), _predarg2ix(sbj_pred, "ARG1-rvrs", predarg2ix), _predarg2ix(obj_pred, "ARG2-rvrs", predarg2ix), _predarg2ix(adj_sbj_pred, "NonARG", predarg2ix)],
                                [_predarg2ix(sbj_pred, "ARG0", predarg2ix), _predarg2ix(verb_pred, "ARG1", predarg2ix), _predarg2ix(obj_pred, "NonARG", predarg2ix), _predarg2ix(adj_sbj_pred, adj_sbj_arg, predarg2ix)],
                                [_predarg2ix(obj_pred, "ARG0", predarg2ix), _predarg2ix(verb_pred, "ARG2", predarg2ix), _predarg2ix(sbj_pred, "NonARG", predarg2ix),  _predarg2ix(adj_sbj_pred, "NonARG", predarg2ix)],
                                [_predarg2ix(adj_sbj_pred, "ARG0", predarg2ix), _predarg2ix(verb_pred, "NonARG", predarg2ix), _predarg2ix(sbj_pred, "{}-rvrs".format(adj_sbj_arg), predarg2ix), _predarg2ix(obj_pred, "NonARG", predarg2ix)], # _predarg2ix(sbj_pred, "{}-rvrs".format(adj_sbj_arg), predarg2ix)
                            ]
                        else:
                            pred_func_nodes_ctxt_predargs = [
                                [_predarg2ix(verb_pred, "ARG0", predarg2ix), _predarg2ix(sbj_pred, "ARG1-rvrs", predarg2ix), _predarg2ix(obj_pred, "ARG2-rvrs", predarg2ix), _predarg2ix(adj_sbj_pred, "NonARG", predarg2ix),  _predarg2ix(adj_obj_pred, "NonARG", predarg2ix)],
                                [_predarg2ix(sbj_pred, "ARG0", predarg2ix), _predarg2ix(verb_pred, "ARG1", predarg2ix), _predarg2ix(obj_pred, "NonARG", predarg2ix), _predarg2ix(adj_sbj_pred, adj_sbj_arg, predarg2ix),  _predarg2ix(adj_obj_pred, "NonARG", predarg2ix)],
                                [_predarg2ix(obj_pred, "ARG0", predarg2ix), _predarg2ix(verb_pred, "ARG2", predarg2ix), _predarg2ix(sbj_pred, "NonARG", predarg2ix),  _predarg2ix(adj_sbj_pred, "NonARG", predarg2ix),  _predarg2ix(adj_obj_pred, adj_obj_arg, predarg2ix)],
                                [_predarg2ix(adj_sbj_pred, "ARG0", predarg2ix), _predarg2ix(verb_pred, "NonARG", predarg2ix), _predarg2ix(sbj_pred, "{}-rvrs".format(adj_sbj_arg), predarg2ix), _predarg2ix(obj_pred, "NonARG", predarg2ix),  _predarg2ix(adj_obj_pred, "NonARG", predarg2ix)], # _predarg2ix(sbj_pred, "{}-rvrs".format(adj_sbj_arg), predarg2ix)
                                [_predarg2ix(adj_obj_pred, "ARG0", predarg2ix), _predarg2ix(verb_pred, "NonARG", predarg2ix), _predarg2ix(sbj_pred, "NonARG", predarg2ix),  _predarg2ix(obj_pred, "{}-rvrs".format(adj_obj_arg), predarg2ix),  _predarg2ix(adj_sbj_pred, "NonARG", predarg2ix)] # _predarg2ix(obj_pred, "{}-rvrs".format(adj_obj_arg), predarg2ix)
                            ]
                        pred_func_nodes_ctxt_predargs_list.append(pred_func_nodes_ctxt_predargs)
                        
                        if adj_obj == "second" and adj_obj_pred == None:
                            pred_func_nodes_ctxt_args = [
                                [arg2ix["ARG0"], arg2ix["ARG1-rvrs"], arg2ix["ARG2-rvrs"], arg2ix["NonARG"]],
                                [arg2ix["ARG0"], arg2ix["ARG1"], arg2ix["NonARG"], arg2ix[adj_sbj_arg]],
                                [arg2ix["ARG0"], arg2ix["ARG2"], arg2ix["NonARG"], arg2ix["NonARG"]],
                                [arg2ix["ARG0"], arg2ix["NonARG"], arg2ix["{}-rvrs".format(adj_sbj_arg) if adj_sbj_arg != "NonARG" else "NonARG"], arg2ix["NonARG"]], # 3rd arg2ix["{}-rvrs".format(adj_sbj_arg)]
                            ]
                        else:
                            pred_func_nodes_ctxt_args = [
                                [arg2ix["ARG0"], arg2ix["ARG1-rvrs"], arg2ix["ARG2-rvrs"], arg2ix["NonARG"], arg2ix["NonARG"]],
                                [arg2ix["ARG0"], arg2ix["ARG1"], arg2ix["NonARG"], arg2ix[adj_sbj_arg], arg2ix["NonARG"]],
                                [arg2ix["ARG0"], arg2ix["ARG2"], arg2ix["NonARG"], arg2ix["NonARG"], arg2ix[adj_obj_arg]],
                                [arg2ix["ARG0"], arg2ix["NonARG"], arg2ix["{}-rvrs".format(adj_sbj_arg) if adj_sbj_arg != "NonARG" else "NonARG"], arg2ix["NonARG"], arg2ix["NonARG"]], # 3rd arg2ix["{}-rvrs".format(adj_sbj_arg)]
                                [arg2ix["ARG0"], arg2ix["NonARG"], arg2ix["NonARG"], arg2ix["{}-rvrs".format(adj_obj_arg) if adj_obj_arg != "NonARG" else "NonARG"], arg2ix["NonARG"]] # 4th arg2ix["{}-rvrs".format(adj_obj_arg)]
                            ]
                    except KeyError as e:
                        if sample_only:
                            continue
                        else:
                            print (e)
                            input ()
                    pred_func_nodes_ctxt_args_list.append(pred_func_nodes_ctxt_args)
                    vars_unzipped_list.append(vars_unzipped)
                ix = gs_asvao2ix[(verb_pred, adj_sbj_pred, sbj_pred, adj_obj_pred, obj_pred)]
                verb_pred, adj_sbj_pred, sbj_pred, adj_obj_pred, obj_pred = [str(s) for s in [verb_pred, adj_sbj_pred, sbj_pred, adj_obj_pred, obj_pred]]
                gs_ix2asvao[ix] = (verb_pred, adj_sbj_pred, sbj_pred, adj_obj_pred, obj_pred)
                
                gs_asvao_ix2landmark2scores[ix][landmark_pred].append(score)
                # if pred_func2ix.get(landmark_pred + "@ARG1", -1) == -1:
                #     print (landmark_pred)
                pred_funcs = (
                    pred_func2ix.get(landmark_pred + "@ARG0", -1),
                    pred_func2ix.get(landmark_pred + "@ARG1", pred_func2ix.get(landmark_pred2sub.get(landmark_pred, "None") + "@ARG1", -1)), # -1 for KS2013: u_demilitarized_v and u_entangled_v
                    pred_func2ix.get(landmark_pred + "@ARG2", -1)
                )
                gs_asvao_ix2landmark2logic_expr[ix][landmark_pred] = pred_funcs
                preds2asvaol["-".join([verb_pred, adj_sbj_pred, sbj_pred, adj_obj_pred, obj_pred, landmark_pred])] = (verb, adj_sbj, sbj, adj_obj, obj, landmark)

            else:
                pass
                # print (verb_pred, sbj_pred, obj_pred, landmark_pred, adj_sbj_pred, adj_obj_pred, arg0s_exist, arg12_exist)
            
    print ("after filtering OOV: {} has {} ASVAO instances".format(name, len(gs_asvao_ix2landmark2scores)))

    gs_eval = {
        "svo_ix2landmark2scores": gs_asvao_ix2landmark2scores,
        "ix2svo": gs_ix2asvao,
        "pred_func_nodes_ctxt_predargs_list": pred_func_nodes_ctxt_predargs_list,
        "pred_func_nodes_ctxt_args_list": pred_func_nodes_ctxt_args_list,
        "svo_ix2landmark2logic_expr": gs_asvao_ix2landmark2logic_expr,
        "vars_unzipped_list": vars_unzipped_list,
    }

    return gs_eval, mapped, preds2asvaol
        
def prepare_ks2013(*args, **kwargs):
    return prepare_gs2012(*args, **kwargs)

def prepare_bless(bless_dir, encoder_arch_type, pos2lemma2pred2cnt, pred_func2cnt, pred2ix, predarg2ix, pred_func2ix, min_pred_func_freq, sample_only):
    bless_path = os.path.join(bless_dir, "BLESS.txt")
    word2pred = None
    with open(os.path.join(bless_dir, "word2pred_premap.json")) as f:
        word2pred = json.load(f)
    bless_data = []
    with open(bless_path) as f:
        line = f.readline().strip()
        while line:
            # concept class relation relatum 
            concept, concept_class, relation, relatum = line.split("\t")
            bless_data.append((concept, concept_class, relation, relatum))
            line = f.readline().strip()
    for (concept, concept_class, relation, relatum) in bless_data:
        concept = concept.rsplit("-", 1)[0]
        concept_pred = word2pred.get(concept)
        if not concept_pred:
            if concept in pos2lemma2pred2cnt['n']:
                concept_pred = pos2lemma2pred2cnt['n'][concept].most_common()[0][0]
        concept_class_pred = word2pred.get(concept_class)
        if not "_" in concept_class:
            if not concept_class_pred:
                if concept_class in pos2lemma2pred2cnt['n']:
                    concept_class_pred = pos2lemma2pred2cnt['n'][concept_class].most_common()[0][0]
        else:
            # amphibian_reptile, ground_mammal, musical_instrument, water_animal
            pass
        relatum, relatum_pos = relatum.rsplit("-", 1)
        relatum_pred = word2pred.get(relatum)
        if not relatum_pred:
            if relatum_pos == 'n' and relatum in pos2lemma2pred2cnt['n']:
                relatum_pred = pos2lemma2pred2cnt['n'][relatum].most_common()[0][0]
            elif relatum_pos == 'j' and relatum in pos2lemma2pred2cnt['a']:
                relatum_pred = pos2lemma2pred2cnt['a'][relatum].most_common()[0][0]
            elif relatum_pos == 'v' and relatum in pos2lemma2pred2cnt['v']:
                relatum_pred = pos2lemma2pred2cnt['v'][relatum].most_common()[0][0]
        if not all([concept_pred, relatum_pred]):
            print (concept, concept_class, relatum, concept_pred, concept_class_pred, relatum_pred)
            continue

def prepare_weeds2014(weeds2014_dir, bless_dir, encoder_arch_type, pos2lemma2pred2cnt, pred_func2cnt, pred2ix, predarg2ix, pred_func2ix, min_pred_func_freq, sample_only):
    weeds2014_path = os.path.join(weeds2014_dir, "BLESS_ent-pairs.json")
    word2pred = None
    with open(os.path.join(weeds2014_dir, "word2pred_premap.json")) as f:
        word2pred = json.load(f)
    weeds2014_data = []
    with open(weeds2014_path) as f:
        weeds2014_data = json.load(f)
        num_pos = sum([1 for word1, word2, is_hyp in weeds2014_data if is_hyp == 1])
    print ("Weeds2014 has {} instances, {} of which is positive".format(len(weeds2014_data), num_pos))

    # include concept_class if bless data is provided
    bless_path = os.path.join(bless_dir, "BLESS.txt") if bless_dir else None
    concept2concept_class_pred = defaultdict()
    concept2concept_class = defaultdict()
    with open(bless_path) as f:
        line = f.readline().strip()
        while line:
            # concept class relation relatum 
            concept, concept_class, relation, relatum = line.split("\t")
            concept = concept.rsplit("-", 1)[0]
            # if "_" in concept_class:
            #     print (concept_class)
            concept_class_pred = word2pred.get(concept_class)
            if not "_" in concept_class:
                if not concept_class_pred:
                    if concept_class in pos2lemma2pred2cnt['n']:
                        concept_class_pred = pos2lemma2pred2cnt['n'][concept_class].most_common()[0][0]
            concept2concept_class[concept] = concept_class
            concept2concept_class_pred[concept] = concept_class_pred
            line = f.readline().strip()


    weeds2014_oov = Counter()
    mapped = defaultdict()
    pred_func_nodes_ctxt_predargs_list = []
    pred_func_nodes_ctxt_args_list = []
    pred_func_nodes_ctxt_predargs_cls_list = []
    pred_func_nodes_ctxt_args_cls_list = []
    vars_unzipped_list = []
    trsfm = TruthConditions(*[None] * 11)
    ix2pair, ix2lbl, pred_funcs_list = [], [], []

    for word1, word2, is_hyp in weeds2014_data:
        # verb_pred, sbj_pred, obj_pred, landmark_pred = None, None, None, None
        word1_pred = word2pred.get(word1)
        if not word1_pred:
            if word1 in pos2lemma2pred2cnt['n']:
                word1_pred = pos2lemma2pred2cnt['n'][word1].most_common()[0][0]
        word2_pred = word2pred.get(word2)
        if not word2_pred:
            if word2 in pos2lemma2pred2cnt['n']:
                word2_pred = pos2lemma2pred2cnt['n'][word2].most_common()[0][0]
        mapped[word1] = word1_pred
        mapped[word2] = word2_pred
        if not word1_pred:
            weeds2014_oov[word1] += 1
        if not word2_pred:
            weeds2014_oov[word2] += 1
        if not word1_pred or not word2_pred:
            continue
        else:
            ix2pair.append((word1_pred, concept2concept_class.get(word1), word2_pred))
            ix2lbl.append(is_hyp)
            arg2ix = trsfm.arg2ix

            # encoder
            ## without concept_class
            vars_unzipped = [
                [1],
                [0]
            ]
            pred_func_nodes_ctxt_predargs, pred_func_nodes_ctxt_args = None, None
            try:
                pred_func_nodes_ctxt_predargs = [
                    [_predarg2ix(word1_pred, "ARG0", predarg2ix)]
                ]
                pred_func_nodes_ctxt_predargs_list.append(pred_func_nodes_ctxt_predargs)
                
                pred_func_nodes_ctxt_args = [
                    [arg2ix["ARG0"]]
                ]
            except KeyError as e:
                if sample_only:
                    continue
                else:
                    print (e)
                    input ()
            pred_func_nodes_ctxt_args_list.append(pred_func_nodes_ctxt_args)
            vars_unzipped_list.append(vars_unzipped)

            ## with concept_class
            if bless_path:
                pred_func_nodes_ctxt_predargs_cls, pred_func_nodes_ctxt_args_cls = None, None
                word1_cls = concept2concept_class.get(word1)
                if word1_cls == 'amphibian_reptile':
                    pred_func_nodes_ctxt_predargs_cls = [
                        [_predarg2ix(word1_pred, "ARG0", predarg2ix), _predarg2ix("_amphibian_n_1", "ARG0", predarg2ix), _predarg2ix("_reptile_n_1", "ARG0", predarg2ix)]
                    ]
                    pred_func_nodes_ctxt_args_cls = [
                        [arg2ix["ARG0"], arg2ix["ARG0"], arg2ix["ARG0"]]
                    ]
                elif word1_cls == 'musical_instrument':
                    pred_func_nodes_ctxt_predargs_cls = [
                        [_predarg2ix(word1_pred, "ARG0", predarg2ix), _predarg2ix("_musical_a_1", "ARG1", predarg2ix), _predarg2ix("_instrument_n_of", "ARG0", predarg2ix)]
                    ]
                    pred_func_nodes_ctxt_args_cls = [
                        [arg2ix["ARG0"], arg2ix["ARG1"], arg2ix["ARG0"]]
                    ]
                elif word1_cls == 'ground_mammal':
                    pred_func_nodes_ctxt_predargs_cls = [
                        [_predarg2ix(word1_pred, "ARG0", predarg2ix), _predarg2ix("_ground_n_1", "NonARG", predarg2ix), _predarg2ix("_mammal_n_1", "ARG0", predarg2ix)]
                    ]
                    pred_func_nodes_ctxt_args_cls = [
                        [arg2ix["ARG0"], arg2ix["NonARG"], arg2ix["ARG0"]]
                    ]
                elif word1_cls == 'water_animal':
                    pred_func_nodes_ctxt_predargs_cls = [
                        [_predarg2ix(word1_pred, "ARG0", predarg2ix), _predarg2ix("_water_n_1", "NonARG", predarg2ix), _predarg2ix("_mammal_n_1", "ARG0", predarg2ix)]
                    ]
                    pred_func_nodes_ctxt_args_cls = [
                        [arg2ix["ARG0"], arg2ix["NonARG"], arg2ix["ARG0"]]
                    ]
                elif word1_cls != None:
                    pred_func_nodes_ctxt_predargs_cls = [
                        [_predarg2ix(word1_pred, "ARG0", predarg2ix), _predarg2ix(concept2concept_class_pred.get(word1), "ARG0", predarg2ix)]
                    ]
                    pred_func_nodes_ctxt_args_cls = [
                        [arg2ix["ARG0"], arg2ix["ARG0"]]
                    ]
                else:
                    print (word1, word2, is_hyp)
                    pred_func_nodes_ctxt_predargs_cls = [
                        [_predarg2ix(word1_pred, "ARG0", predarg2ix)]
                    ]
                    pred_func_nodes_ctxt_args_cls = [
                        [arg2ix["ARG0"]]
                    ]
                pred_func_nodes_ctxt_predargs_cls_list.append(pred_func_nodes_ctxt_predargs_cls)
                pred_func_nodes_ctxt_args_cls_list.append(pred_func_nodes_ctxt_args_cls)
                vars_unzipped_list.append(vars_unzipped)
            
            # decoder
            pred_funcs = (
                pred_func2ix.get(word2_pred + "@ARG0", -1),
            )
            pred_funcs_list.append(pred_funcs)

    weeds2014_eval = {
        "ix2lbl": ix2lbl,
        "ix2pair": ix2pair,
        "pred_func_nodes_ctxt_predargs_list": pred_func_nodes_ctxt_predargs_list,
        "pred_func_nodes_ctxt_args_list": pred_func_nodes_ctxt_args_list,
        "pred_func_nodes_ctxt_predargs_cls_list": pred_func_nodes_ctxt_predargs_cls,
        "pred_func_nodes_ctxt_args_cls_list": pred_func_nodes_ctxt_args_cls,
        "pred_funcs_list": pred_funcs_list,
        "vars_unzipped_list": vars_unzipped_list,
    }

        # if not all([verb_pred, sbj_pred, obj_pred, landmark_pred, adj_sbj_pred, adj_obj_pred]) and adj_obj != "second":
        #     print (verb, verb_pred, sbj, sbj_pred, obj, obj_pred, landmark, landmark_pred, adj_sbj, adj_sbj_pred, adj_obj, adj_obj_pred)
        #     continue
        # else:
            # ensure arg0 of nouns and arg0,1,2 or verbs are not OOV

    return weeds2014_eval, mapped


def main(config,
        #  hyp_dir, relpron_dir, gs2011_dir, gs2012_dir, gs2013_dir, ks2013_dir, bless_dir,
        eval_data_sets_dir, eval_data_dir):

    transformed_dir  = config["data_loader"]["args"]["transformed_dir"]
    transformed_dir_name = config["transformed_dir_name"]

    # train_dataloader = config['data_loader']['type']
    encoder_arch_type = config['encoder_arch']['type']

    config_eval_dir = os.path.join(eval_data_dir, transformed_dir_name)

    transformed_info_dir = os.path.join(transformed_dir, "info")
    transformed_info = get_transformed_info(transformed_info_dir)

    pred_func_ix2cnt, content_pred2cnt, pred2ix, content_predarg2ix, pred_func2ix = transformed_info
    pos2lemma2pred2cnt = get_pos2lemma2pred2cnt(pred_func_ix2cnt, pred_func2ix)

    pred_func2cnt = Counter()
    for pred_func, ix in pred_func2ix.items():
        pred_func2cnt[pred_func] = pred_func_ix2cnt[ix]

    sample_only = config["sample_only"]
    eval_data_sets_dirs = ["RELPRON", "GS2011", "GS2013", "GS2012", "bless-gems", "Weeds2014"]
    relpron_dir, gs2011_dir, gs2013_dir, gs2012_dir, bless_dir, weeds2014_dir = [
        os.path.join(eval_data_sets_dir, dir) if dir != None else None for dir in eval_data_sets_dirs 
    ]
    # hyp
    # if hyp_dir:
    #     config_eval_hyp_dir = os.path.join(config_eval_dir, "hyp")
    #     os.makedirs(config_eval_hyp_dir, exist_ok = True)
    #     min_freqs = [0, 5000, 50000]
    #     for trasitive_num in [1,2]:
    #         hyp_file = "hypernyms_wn_{}.txt".format(trasitive_num)
    #         for freq in min_freqs:
    #             hyp_pred_pairs_wn = prepare_hyp(config, hyp_dir, hyp_file, pos2lemma2pred2cnt, freq)
    #             if hyp_pred_pairs_wn:
    #                 hyp_pred_pairs_path = os.path.join(config_eval_hyp_dir, "hyp_pred_pairs_t{}_f{}.json".format(trasitive_num, freq))
    #                 with open(hyp_pred_pairs_path, "w") as f:
    #                     json.dump(hyp_pred_pairs_wn, f, indent = 4)
    # relpron
    if relpron_dir:
        config_eval_relpron_dir = os.path.join(config_eval_dir, "relpron")
        config_eval_relpron_data_dir = os.path.join(config_eval_relpron_dir, "data")
        config_eval_relpron_info_dir = os.path.join(config_eval_relpron_dir, "info")
        svo = config["eval_relpron_dataloader"]["args"]["svo"]
        os.makedirs(config_eval_relpron_data_dir, exist_ok = True)
        os.makedirs(config_eval_relpron_info_dir, exist_ok = True)
        min_pred_func_freqs = [0]
        for freq in min_pred_func_freqs:
            relpron_splits, mapped = prepare_relpron(relpron_dir, encoder_arch_type, svo, pos2lemma2pred2cnt, pred_func2cnt, pred2ix, content_predarg2ix, pred_func2ix, freq, sample_only)
            if relpron_splits:
                for split, data in relpron_splits.items():
                    relpron_split_path = os.path.join(config_eval_relpron_data_dir, "relpron_{}_f{}.json".format(split, freq))
                    with open(relpron_split_path, "w") as f:
                        json.dump(data, f, indent = 4)
                relpron_mapped_path = os.path.join(config_eval_relpron_info_dir, "relpron_f{}_mapped.json".format(freq))
                with open(relpron_mapped_path, "w") as f:
                    json.dump(mapped, f, indent = 4)

    # GS2011
    if gs2011_dir:
        config_eval_gs2011_dir = os.path.join(config_eval_dir, "gs2011")
        config_eval_gs2011_data_dir = os.path.join(config_eval_gs2011_dir, "data")
        config_eval_gs2011_info_dir = os.path.join(config_eval_gs2011_dir, "info")
        os.makedirs(config_eval_gs2011_data_dir, exist_ok = True)
        os.makedirs(config_eval_gs2011_info_dir, exist_ok = True)
        min_pred_func_freqs = [0]
        for freq in min_pred_func_freqs:
            gs2011_eval, mapped, preds2svol = prepare_gs(gs2011_dir, encoder_arch_type, pos2lemma2pred2cnt, pred_func2cnt, pred2ix, content_predarg2ix, pred_func2ix, freq, sample_only, "2011")
            if gs2011_eval:
                gs2011_eval_path = os.path.join(config_eval_gs2011_data_dir, "gs2011_f{}.json".format(freq))
                with open(gs2011_eval_path, "w") as f:
                    json.dump(gs2011_eval, f, indent = 4)
                gs2011_mapped_path = os.path.join(config_eval_gs2011_info_dir, "gs2011_f{}_mapped.json".format(freq))
                with open(gs2011_mapped_path, "w") as f:
                    json.dump(mapped, f, indent = 4)
                gs2011_preds2svol_path = os.path.join(config_eval_gs2011_info_dir, "gs2011_f{}_preds2svol.json".format(freq))
                with open(gs2011_preds2svol_path, "w") as f:
                    json.dump(preds2svol, f, indent = 4)

    # GS2013
    if gs2013_dir:
        config_eval_gs2013_dir = os.path.join(config_eval_dir, "gs2013")
        config_eval_gs2013_data_dir = os.path.join(config_eval_gs2013_dir, "data")
        config_eval_gs2013_info_dir = os.path.join(config_eval_gs2013_dir, "info")
        os.makedirs(config_eval_gs2013_data_dir, exist_ok = True)
        os.makedirs(config_eval_gs2013_info_dir, exist_ok = True)
        min_pred_func_freqs = [0]
        for freq in min_pred_func_freqs:
            gs2013_eval, mapped, preds2svol = prepare_gs(gs2013_dir, encoder_arch_type, pos2lemma2pred2cnt, pred_func2cnt, pred2ix, content_predarg2ix, pred_func2ix, freq, sample_only, "2013")
            if gs2013_eval:
                gs2013_eval_path = os.path.join(config_eval_gs2013_data_dir, "gs2013_f{}.json".format(freq))
                with open(gs2013_eval_path, "w") as f:
                    json.dump(gs2013_eval, f, indent = 4)
                gs2013_mapped_path = os.path.join(config_eval_gs2013_info_dir, "gs2013_f{}_mapped.json".format(freq))
                with open(gs2013_mapped_path, "w") as f:
                    json.dump(mapped, f, indent = 4)
                gs2013_preds2svol_path = os.path.join(config_eval_gs2013_info_dir, "gs2013_f{}_preds2svol.json".format(freq))
                with open(gs2013_preds2svol_path, "w") as f:
                    json.dump(preds2svol, f, indent = 4)

    # GS2012
    if gs2012_dir:
        config_eval_gs2012_dir = os.path.join(config_eval_dir, "gs2012")
        config_eval_gs2012_data_dir = os.path.join(config_eval_gs2012_dir, "data")
        config_eval_gs2012_info_dir = os.path.join(config_eval_gs2012_dir, "info")
        os.makedirs(config_eval_gs2012_data_dir, exist_ok = True)
        os.makedirs(config_eval_gs2012_info_dir, exist_ok = True)
        min_pred_func_freqs = [0]
        for freq in min_pred_func_freqs:
            gs2012_eval, mapped, preds2asvaol = prepare_gs2012(gs2012_dir, encoder_arch_type, pos2lemma2pred2cnt, pred_func2cnt, pred2ix, content_predarg2ix, pred_func2ix, freq, sample_only, name = "GS2012")
            if gs2012_eval:
                gs2012_eval_path = os.path.join(config_eval_gs2012_data_dir, "gs2012_f{}.json".format(freq))
                with open(gs2012_eval_path, "w") as f:
                    json.dump(gs2012_eval, f, indent = 4)
                gs2012_mapped_path = os.path.join(config_eval_gs2012_info_dir, "gs2012_f{}_mapped.json".format(freq))
                with open(gs2012_mapped_path, "w") as f:
                    json.dump(mapped, f, indent = 4)
                gs2012_preds2asvaol_path = os.path.join(config_eval_gs2012_info_dir, "gs2012_f{}_preds2asvaol.json".format(freq))
                with open(gs2012_preds2asvaol_path, "w") as f:
                    json.dump(preds2asvaol, f, indent = 4)

    # if ks2013_dir:
    #     config_eval_ks2013_dir = os.path.join(config_eval_dir, "ks2013")
    #     config_eval_ks2013_data_dir = os.path.join(config_eval_ks2013_dir, "data")
    #     config_eval_ks2013_info_dir = os.path.join(config_eval_ks2013_dir, "info")
    #     os.makedirs(config_eval_ks2013_data_dir, exist_ok = True)
    #     os.makedirs(config_eval_ks2013_info_dir, exist_ok = True)
    #     min_pred_func_freqs = [0]
    #     for freq in min_pred_func_freqs:
    #         ks2013_eval, mapped = prepare_ks2013(ks2013_dir, encoder_arch_type, pos2lemma2pred2cnt, pred_func2cnt, pred2ix, content_predarg2ix, pred_func2ix, freq, sample_only, name = "KS2013")
    #         if ks2013_eval:
    #             ks2013_eval_path = os.path.join(config_eval_ks2013_data_dir, "ks2013_f{}.json".format(freq))
    #             with open(ks2013_eval_path, "w") as f:
    #                 json.dump(ks2013_eval, f, indent = 4)
    #             ks2013_mapped_path = os.path.join(config_eval_ks2013_info_dir, "ks2013_f{}_mapped.json".format(freq))
    #             with open(ks2013_mapped_path, "w") as f:
    #                 json.dump(mapped, f, indent = 4)

    # if bless_dir:
    #     config_eval_bless_dir = os.path.join(config_eval_dir, "bless")
    #     config_eval_bless_data_dir = os.path.join(config_eval_bless_dir, "data")
    #     config_eval_bless_info_dir = os.path.join(config_eval_bless_dir, "info")
    #     os.makedirs(config_eval_bless_data_dir, exist_ok = True)
    #     os.makedirs(config_eval_bless_info_dir, exist_ok = True)
    #     min_pred_func_freqs = [0]
    #     for freq in min_pred_func_freqs:
    #         bless_eval, mapped = prepare_bless(bless_dir, encoder_arch_type, pos2lemma2pred2cnt, pred_func2cnt, pred2ix, content_predarg2ix, pred_func2ix, freq, sample_only)
    #         if bless_eval:
    #             bless_eval_path = os.path.join(config_eval_bless_data_dir, "bless_f{}.json".format(freq))
    #             with open(bless_eval_path, "w") as f:
    #                 json.dump(bless_eval, f, indent = 4)
    #             bless_mapped_path = os.path.join(config_eval_bless_info_dir, "bless_f{}_mapped.json".format(freq))
    #             with open(bless_mapped_path, "w") as f:
    #                 json.dump(mapped, f, indent = 4)

    # if weeds2014_dir:
    #     config_eval_weeds2014_dir = os.path.join(config_eval_dir, "weeds2014")
    #     config_eval_weeds2014_data_dir = os.path.join(config_eval_weeds2014_dir, "data")
    #     config_eval_weeds2014_info_dir = os.path.join(config_eval_weeds2014_dir, "info")
    #     os.makedirs(config_eval_weeds2014_data_dir, exist_ok = True)
    #     os.makedirs(config_eval_weeds2014_info_dir, exist_ok = True)
    #     min_pred_func_freqs = [0]
    #     for freq in min_pred_func_freqs:
    #         weeds2014_eval, mapped = prepare_weeds2014(weeds2014_dir, bless_dir, encoder_arch_type, pos2lemma2pred2cnt, pred_func2cnt, pred2ix, content_predarg2ix, pred_func2ix, freq, sample_only)
    #         if weeds2014_eval:
    #             weeds2014_eval_path = os.path.join(config_eval_weeds2014_data_dir, "weeds2014_f{}.json".format(freq))
    #             with open(weeds2014_eval_path, "w") as f:
    #                 json.dump(weeds2014_eval, f, indent = 4)
    #             weeds2014_mapped_path = os.path.join(config_eval_weeds2014_info_dir, "weeds2014_f{}_mapped.json".format(freq))
    #             with open(weeds2014_mapped_path, "w") as f:
    #                 json.dump(mapped, f, indent = 4)

    with open("vs.txt", "w") as f:
        json.dump(list(vs), f, indent = 4)
    with open("oovs.txt", "w") as f:
        json.dump(list(oovs), f, indent = 4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare_eval')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    # parser.add_argument('-p', '--hyp_dir', default=None, type=str,
    #                   help='directory to hypernym data (default: None)')
    # parser.add_argument('-l', '--relpron_dir', default=None, type=str,
    #                   help='directory to relpron data (default: None)')
    # parser.add_argument('-g', '--gs2011_dir', default=None, type=str,
    #                   help='directory to GS2011 data (default: None)')
    # parser.add_argument('-a', '--gs2012_dir', default=None, type=str,
    #                   help='directory to GS2011 data (default: None)')
    # parser.add_argument('-s', '--gs2013_dir', default=None, type=str,
    #                   help='directory to GS2011 data (default: None)')
    # parser.add_argument('-k', '--ks2013_dir', default=None, type=str,
    #                   help='directory to KS2013 data (default: None)')
    # parser.add_argument('-b', '--bless_dir', default=None, type=str,
    #                   help='directory to BLESS data (default: None)')
    # parser.add_argument('-w', '--weeds2014_dir', default=None, type=str,
    #                   help='directory to BLESS data (default: None)')
    parser.add_argument('-a', '--eval_data_sets_dir', default=None, type=str,
                      help='directory to raw evaluation data sets (default: None)')
    parser.add_argument('-e', '--eval_data_dir', default=None, type=str,
                      help='directory to preprocessed evaluation data (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()
    # custom cli options to modify configuration from default values given in json file.
    options = []
    config = ConfigParser.from_args(parser, options)
    main(
        config,
        # args.hyp_dir,
        # args.relpron_dir,
        # args.gs2011_dir,
        # args.gs2012_dir,
        # args.gs2013_dir,
        # args.ks2013_dir,
        # args.bless_dir,
        # args.weeds2014_dir,
        args.eval_data_sets_dir,
        args.eval_data_dir
    )