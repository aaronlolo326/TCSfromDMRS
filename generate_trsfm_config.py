import os
import argparse
from collections import defaultdict
import json


ignore_scopal_adv = True
ignore_modal_verb = True

lexical_pos = ['n', 'a', 'v']
abs_pred_func = {
    'sem': ['poss', 'loc_nonsp', 'of_p', 'with_p'],
    'cpd': ['compound', 'compound_name'],
    # 'and': ['implicit_conj_rel', 'subord_rel'], 
    'neg': ['neg', 'not_x_deg'], #_hardly_x_deg
    'gen': ['pron', 'generic_entity', 'generic_entity','pron','thing','person','place_n','time_n','reason'],
    'carg': ['season', 'holiday']
}
neg_quantifier = ['_no_q', '_neither_q', '_no+more_q']
transparent_preds = ['nominalization', 'eventuality']
ignore = ['_threedot-pnct_a_1']

transform_config = defaultdict()
type2pred2args_op = defaultdict(defaultdict)

def get_type2pred2args_op(logic_pred_anno_path):

    with open(logic_pred_anno_path) as f:
        line = f.readline().strip()
        while line:
            line = line.strip()
            if line:
                logic_op, pred_args = line.split("\t")
                if logic_op.startswith("*"):
                    pass
                    # logical_pred2args2op[pred_args] = {"args": [], "op": logic_op}
                else:
                    op_type, op = logic_op.split("-")
                    pred, args = pred_args.split(";")
                    args = tuple(args.split("|"))
                    type2pred2args_op[op_type][pred] = (args, logic_op)
            line = f.readline()

    return type2pred2args_op

def main(logic_pred_anno_path, trsfm_config_path):
    type2pred2args_op = get_type2pred2args_op(logic_pred_anno_path)
    transform_config = {
        "lexical_pos": lexical_pos,
        "abs_pred_func": abs_pred_func,
        "neg_quantifier": neg_quantifier,
        "transparent_preds": transparent_preds,
        "logical_preds": type2pred2args_op,
        "ignore": ignore,
        "ignore_scopal_adv": ignore_scopal_adv,
        "ignore_modal_verb": ignore_modal_verb
    }
    with open(trsfm_config_path, "w") as f:
        json.dump(transform_config, f, indent = 4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate_config')
    parser.add_argument('-l', '--logic_pred_anno_path', default=None, type=str,
                      help='file path to logical predicate annotation (default: None)')
    parser.add_argument('-c', '--trsfm_config_path', default=None, type=str,
                      help='path to save the trasnform config file (default: None)')
    args = parser.parse_args()
    # custom cli options to modify configuration from default values given in json file.
    main(
        args.logic_pred_anno_path,
        args.trsfm_config_path
    )