import argparse
import collections
import json
from json import decoder

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

from pprint import pprint
import os
from collections import Counter, defaultdict
from itertools import chain
import copy
import re

import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, BertForMaskedLM
from transformers import pipeline, logging
from pattern.en import conjugate, lemma, lexeme, PRESENT, SG, PL
from pattern.en import quantify

from scipy.stats import spearmanr

logging.set_verbosity_error()

def gs_lemmas_to_snt(pos2lemma):
    snt = ""
    if not pos2lemma["adj_sbj"] or not pos2lemma["adj_obj"]:
        sbj_atc = quantify([pos2lemma["sbj"]]).split(" ")[0]
        obj_atc = quantify([pos2lemma["obj"]]).split(" ")[0]
        snt = " ".join([
            sbj_atc,
            pos2lemma["sbj"],
            conjugate(verb=pos2lemma["verb"],tense=PRESENT,number=SG),
            obj_atc,
            pos2lemma["obj"]
        ])
    else:
        sbj_atc = quantify([pos2lemma["adj_sbj"]]).split(" ")[0]
        obj_atc = quantify([pos2lemma["adj_obj"]]).split(" ")[0]
        snt = " ".join([
            sbj_atc,
            pos2lemma["adj_sbj"],
            pos2lemma["sbj"],
            conjugate(verb=pos2lemma["verb"],tense=PRESENT,number=SG),
            obj_atc,
            pos2lemma["adj_obj"],
            pos2lemma["obj"]
        ])
    return snt + "."

def relpron_lemmas_to_snt(pos2lemma, sbj_obj):
    snt = ""
    sbj_atc = quantify([pos2lemma["sbj"]]).split(" ")[0]
    obj_atc = quantify([pos2lemma["obj"]]).split(" ")[0]
    if sbj_obj == 'SBJ':
        snt = " ".join([
            sbj_atc,
            pos2lemma["sbj"],
            "that",
            conjugate(verb=pos2lemma["verb"],tense=PRESENT,number=SG),
            obj_atc,
            pos2lemma["obj"],
            "is",
            "a",
            "[MASK]"
        ])
    elif sbj_obj == 'OBJ':
        atc = quantify([pos2lemma["obj"]]).split(" ")[0]
        snt = " ".join([
            obj_atc,
            pos2lemma["obj"],
            "that",
            sbj_atc,
            pos2lemma["sbj"],
            conjugate(verb=pos2lemma["verb"],tense=PRESENT,number=SG),
            "is",
            "a",
            "[MASK]"
        ])

    return snt + "."
    # return "[CLS] " + snt + "[SEP]"

def eval_relpron_map(term2prop_idx2prob, term2props_idx, idx2prop):

    # also in evaluator.py
    term2props = defaultdict(list)
    confounders = []
    mean_ap = 0
    confounder_ranks = []
    # compute MAP
    ap = []
    # sort the truths to get the ranked prop_idx
    for term_idx, (term, truth_of_props) in enumerate(term2prop_idx2prob.items()):
        ap.append(0.0)
        ranked = sorted(range(len(truth_of_props)), key = lambda i: truth_of_props[i], reverse = True)
        # compute AP
        correct_at_k = 0
        for prop_rank, prop_idx in enumerate(ranked):
            # print (relpron_labels[relpron_file_name])
            # print (prop_idx, term, term2props_idx[term])
            if prop_idx in term2props_idx[term]:
                # print (len(relpron_props))
                term2props[term].append((idx2prop[prop_idx], prop_rank + 1, True))
                correct_at_k += 1
                prec = correct_at_k / (prop_rank + 1)
                ap[term_idx] += prec
                if any([term == prop_word for prop_word in idx2prop[prop_idx]]):
                    print ("true confounder:", term, idx2prop[prop_idx])
            else:
                if prop_rank < 10:
                    term2props[term].append((idx2prop[prop_idx], prop_rank + 1, False))
                # confounder
                if any([term == prop_word for prop_word in idx2prop[prop_idx]]):
                    confounders.append((term, idx2prop[prop_idx], prop_rank + 1, False))
                    confounder_ranks.append(prop_rank + 1)
        ap[term_idx] = ap[term_idx] / correct_at_k
    mean_ap = sum(ap) / len(ap)
    
    return mean_ap, term2props, confounders, confounder_ranks

def bert_on_relpron(relpron_dir, model_name, data_set_name):

    def v4(snt, all_terms, fill_masker):
        filled_list = fill_masker(snt, top_k = 100, targets = all_terms)
        # score_sum = sum([d['score'] for d in filled_list])
        # for i in range(len(filled_list)):
        #     filled_list[i]['score'] /= score_sum
        for filled in filled_list:
            token_str = filled['token_str'] 
            if filled['token_str'] in model_oov:
                token_str_full = model_oov[token_str]
            else:
                token_str_full = token_str
            split2term2prop_idx2prob[split][token_str_full].append(filled['score'])

    def v2(snt, all_terms, tokenizer, model):
        model.eval()
        tokenized_text = tokenizer.tokenize(snt)
        # print (tokenized_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        # print (indexed_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        # print (tokens_tensor)
        mask_index = tokenized_text.index('[MASK]')
        # print (mask_index)
        with torch.no_grad():
            outputs = model(tokens_tensor)
            # print (outputs[0].shape)
            predictions_for_mask = outputs[0][0, mask_index]
            # print (predictions_for_mask.shape)
            for term in all_terms:
                term_id = tokenizer.convert_tokens_to_ids([term])[0]
                score = predictions_for_mask[term_id]
                split2term2prop_idx2prob[split][term].append(score)

    relpron_path = {
        "dev": os.path.join(relpron_dir, "relpron.dev"),
        "test": os.path.join(relpron_dir, "relpron.test")
    }

    relpron_data = defaultdict(list)
    split2term2props_idx = defaultdict(lambda: defaultdict(set))
    split2props = defaultdict(list)
    split2prop2snt = defaultdict(defaultdict)
    split2idx2prop = defaultdict(defaultdict)
    split2prop2idx = defaultdict(defaultdict)
    for split, path in relpron_path.items():
        # term2idx = defaultdict()
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
                if (sbj_obj, sbj, verb, obj) not in split2props[split]:
                    split2props[split].append((sbj_obj, sbj, verb, obj))
                    split2prop2idx[split][(sbj_obj, sbj, verb, obj)] = len(split2props[split]) - 1
                    split2idx2prop[split][len(split2props[split]) - 1] = (sbj_obj, sbj, verb, obj)
                    pos2lemma = {
                        "sbj": sbj, "verb": verb, "obj":  obj
                    }
                    snt = relpron_lemmas_to_snt(pos2lemma, sbj_obj)
                    # print (snt)
                    split2prop2snt[split][(sbj_obj, sbj, verb, obj)] = snt

                split2term2props_idx[split][targ].add(split2prop2idx[split][(sbj_obj, sbj, verb, obj)])
                line = f.readline().strip()

    split2term2prop_idx2prob = defaultdict(lambda: defaultdict(list))
    split2term2props = defaultdict(lambda: defaultdict(list))
    results_metrics = defaultdict(defaultdict)
    split2confounders = defaultdict()
    model_oov = {'learn': 'learner', 'rode': 'rodent'}
    
    fill_masker = pipeline(model = model_name)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    for split, props in split2props.items():
        all_terms = list(split2term2props_idx[split].keys())
        for prop in props:
            snt = split2prop2snt[split][prop]
            v4(snt, all_terms, fill_masker)
            # v2(snt, all_terms, tokenizer, model)
            


        mean_ap, term2props, confounders, confounder_ranks = eval_relpron_map(
            split2term2prop_idx2prob[split], split2term2props_idx[split], split2idx2prop[split]
        )
        mean_confounder_rank = sum(confounder_ranks)/len(confounder_ranks)
        split2term2props[split] = term2props
        split2confounders[split] = confounders
        # metrics
        results_metrics[split] = {
            "map": mean_ap,
            "mean_confounder_rank": mean_confounder_rank
            # "percent_term_max_truth": sum(check_max_truth[relpron_file_name]) / len(check_max_truth[relpron_file_name])
        }
    return {
        "results_metrics": results_metrics,
        "outputs": {
            "split2term2props": split2term2props,
            "split2confounders": split2confounders
        }
    }


def bert_on_gs(gs_dir, model_name, data_set_name):
    
    gs_path = os.path.join(gs_dir, "{}data.txt".format(data_set_name))

    gs_data = list()
    svo2snt = defaultdict()

    snts = []
    snts_verbs_set = set()
    snts_landmarks_set = set()
    # parti2svol2score = defaultdict(defaultdict)
    svol2scores = defaultdict(list)
    svol2pairs = defaultdict()

    with open(gs_path) as f:
        line = f.readline().strip()
        line = f.readline().strip()
        while line:
            # sentence_id annotator_id adj_subj subj landmark verb adj_obj obj annotator_score # 2012
            ## wrong order of verb and landmark?
            # participant20 provide family home supply 4 HIGH
            adj_sbj, adj_obj = None, None
            if data_set_name in ['GS2011', "GS2013"]:
                parti, verb, sbj, obj, landmark, score, hilo = line.split(" ")
                gs_data.append((verb, sbj, obj, landmark, score))
                # gs_data.append((verb, sbj, obj, landmark, score))
            elif data_set_name in ['GS2012']:
                _, parti, adj_sbj, sbj, verb, landmark, adj_obj, obj, score = line.split(" ")
                gs_data.append((verb, adj_sbj, sbj, adj_obj, obj, landmark, score))
            
            score = int(score)
            pos2lemma = {"adj_sbj": adj_sbj, "sbj": sbj, "verb": verb, "adj_obj": adj_obj, "obj":  obj}
            snt_verb = gs_lemmas_to_snt(pos2lemma)
            pos2lemma = {"adj_sbj": adj_sbj, "sbj": sbj, "verb": landmark, "adj_obj": adj_obj, "obj":  obj}
            snt_landmark = gs_lemmas_to_snt(pos2lemma)
            # snts.append(snt)
            snts_verbs_set.add((snt_verb, conjugate(verb=verb,tense=PRESENT,number=SG)))
            snts_landmarks_set.add((snt_landmark, conjugate(verb=landmark,tense=PRESENT,number=SG)))
            # parti2svol2score[parti][(snt_verb, snt_landmark)] = score
            if data_set_name in ['GS2011', "GS2013"]:
                svol_key = (verb, sbj, obj, landmark)
                svo_key = (verb, sbj, obj)
            elif data_set_name in ['GS2012']:
                svol_key = (verb, adj_sbj, sbj, adj_obj, obj, landmark)
                svo_key = (verb, adj_sbj, sbj, adj_obj, obj)
            svol2scores[svol_key].append(score)
            svo2snt[svo_key] = snt_verb
            svo2snt[svo_key] = snt_landmark
            svol2pairs[svol_key] = (snt_verb, snt_landmark)
                # verb, adj_sbj, sbj, adj_obj, obj, landmark)] = score
            line = f.readline().strip()

    # apply BERT for verb's and landmark's (avg-subwords) embeddings

    tokenizer = BertTokenizer.from_pretrained(model_name)
    auto_tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name)

    snt2emb_verb = defaultdict()
    verb_3sg2subwords = {'alleges': ['all', '##ege', '##s'], 'satisfies': ['sat', '##is', '##fies']}
    # verb_3sg2subwords = {'allege': ['all', '##ege']} #v2
    for snts_set in [snts_verbs_set, snts_landmarks_set]:
        for snt, verb_3sg in snts_set:
            tokenized = auto_tokenizer.tokenize(snt)
            if not verb_3sg in tokenized:
                for i in range(0, len(tokenized) - len(verb_3sg2subwords[verb_3sg]) + 1):
                    if tokenized[i: i + len(verb_3sg2subwords[verb_3sg])] == verb_3sg2subwords[verb_3sg]:
                        verb_idx = [i, i + len(verb_3sg2subwords[verb_3sg])]
            else:
                verb_idx = [tokenized.index(verb_3sg), tokenized.index(verb_3sg) + 1]
            encoded_input = tokenizer(snt, return_tensors='pt')
            output = bert_model(**encoded_input)
            # + 1 for 101 [cls]
            emb_verb = torch.mean(output['last_hidden_state'].squeeze()[verb_idx[0] + 1: verb_idx[1] + 1], dim = 0)
            snt2emb_verb[snt] = emb_verb

    # find cos_sim
    svol2cos_sim = defaultdict()
    for svol, (snt_verb, snt_landmark) in svol2pairs.items():
        svol2cos_sim[svol] = torch.nn.functional.cosine_similarity(snt2emb_verb[snt_verb], snt2emb_verb[snt_landmark], dim = 0).item()
    # print (svol2cos_sim)
    # compute rhos
    landmark_score_sep = []
    landmark_score_avg = []
    landmark_cos_sims_sep = []
    landmark_cos_sims_avg = []
    processed_svol = set()
    for *svol, score in gs_data:
        svol = tuple(svol)
        cos_sim = svol2cos_sim[svol]
        landmark_score_sep.append(score)
        landmark_cos_sims_sep.append(cos_sim)
        if svol not in processed_svol:
            processed_svol.add(svol)
            avg_score = sum(svol2scores[svol])/len(svol2scores[svol])
            landmark_score_avg.append(avg_score)
            landmark_cos_sims_avg.append(cos_sim)
    rho_avg = spearmanr(landmark_score_avg, landmark_cos_sims_avg)[0]
    rho_sep = spearmanr(landmark_score_sep, landmark_cos_sims_sep)[0]
    results_metrics = {
        "rho_sep": rho_sep,
        "rho_avg": rho_avg
    }

    return {
        "results_metrics": results_metrics,
        "outputs": {
            "svol2cos_sim": svol2cos_sim,
            "svol2scores": svol2scores
        }
    }

def main(eval_data_sets_dir):
    
    model_name2ds_dir2model = {
        "bert-base-uncased": {
            "RELPRON": bert_on_relpron,
            "GS2011": bert_on_gs,
            "GS2013": bert_on_gs,
            "GS2012": bert_on_gs
        }
    }

    model2ds2metrics = defaultdict(defaultdict)
    model2ds2outputs = defaultdict(lambda: defaultdict(defaultdict))

    for model_name, ds_dir2model in model_name2ds_dir2model.items():
        results_baseline_dir = os.path.join(
            'results_baselines',
            model_name
        )
        os.makedirs(results_baseline_dir, exist_ok = True)
        for data_set_dir, model in ds_dir2model.items():
            data_set_dir_path = os.path.join(eval_data_sets_dir, data_set_dir) 
            data_set_name = data_set_dir.lower()
            results = model(data_set_dir_path, model_name, data_set_name = data_set_dir)
            model2ds2metrics[model_name][data_set_name] = results['results_metrics']
            model2ds2outputs[model_name][data_set_name] = results['outputs']     
            
        with open(os.path.join(results_baseline_dir, "metrics"), "w") as f:
            json.dump(model2ds2metrics[model_name], f, indent = 4)
        with open(os.path.join(results_baseline_dir, "outputs.json"), "w") as f:
            json.dump(model2ds2outputs[model_name]['relpron'], f, indent = 4)

    return {
        "model2ds2metrics": model2ds2metrics,
        "model2ds2outputs": model2ds2outputs
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='baselines')
    parser.add_argument('-a', '--eval_data_sets_dir', default=None, type=str,
                      help='directory to raw evaluation data sets (default: None)')
    args = parser.parse_args()
    # custom cli options to modify configuration from default values given in json file.
    results = main(
        args.eval_data_sets_dir,
    )
    pprint (results['model2ds2metrics'])


