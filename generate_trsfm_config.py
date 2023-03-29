import os
import argparse
from collections import defaultdict
import json
from pickle import FALSE


ignore_scopal_adv = True
ignore_modal_verb = True

content_pos = ['n', 'a', 'v']
abs_pred_func = {
    # 'sem': ['poss', 'loc_nonsp', 'of_p', 'with_p'],
    # 'cpd': ['compound', 'compound_name'],
    # 'and': ['implicit_conj_rel', 'subord_rel'], 
    'neg': ['neg', 'not_x_deg'], #_hardly_x_deg
    'gen': ['pron', 'generic_entity', 'thing','person','place_n','time_n','reason'],
    'carg': ['season', 'holiday'],
    'keep': ['generic_entity', 'person']
}
neg_quantifier = ['_no_q', '_neither_q', '_no+more_q']
transparent_preds = ['nominalization', 'eventuality']
ignore = [
    '_threedot-pnct_a_1',
    "_also_a_1",
    "_other_a_1",
    "_re-_a_again",
    "_however_a_1",
    "_several_a_1",
    "_then_a_1",
    "_only_a_1",
    "_there_a_1",
]
modals = [
    "_can_v_modal",
    "_would_v_modal",
    "_may_v_modal",
    "_could_v_modal",
    "_must_v_modal",
    "_should_v_modal",
    "_have_v_qmodal",
    "_might_v_modal",
    "_going+to_v_qmodal",
    "_used+to_v_qmodal",
    "_ought_v_qmodal",
    "_need_v_qmodal",
    "_dare_v_qmodal",
    "_had+better_v_qmodal",
    "_had+best_v_qmodal",
    "_gotta_v_modal"
]

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
        "content_pos": content_pos,
        "abs_pred_func": abs_pred_func,
        "neg_quantifier": neg_quantifier,
        "transparent_preds": transparent_preds,
        "logical_preds": type2pred2args_op,
        "modals": modals,
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