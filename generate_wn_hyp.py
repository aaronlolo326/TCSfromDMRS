from nltk.corpus import wordnet as wn

import os
import argparse
from collections import defaultdict
import json
from pprint import pprint
from itertools import islice

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm


def is_one_word(word):
    if word.isalpha():
        return True
    else:
        return False

def gen_wn_hyp(depth = 1):
    '''
    depth = transitive depth of hypernyms
    '''
    hyp_pairs = []
    # print (len(list(wn.all_synsets('n'))))
    for synset in tqdm(wn.all_synsets('n')):
        curr_lemmas = [str(lemma.name())
            for lemma in synset.lemmas()
            if is_one_word(str(lemma.name()))
        ]
        if not curr_lemmas:
            continue
        get_hyp = lambda x: x.hypernyms()
        hyp_tc = list(synset.closure(get_hyp, depth = depth))
        hyp_tc_lemmas = [str(lemma.name())
            for synset in hyp_tc
                for lemma in synset.lemmas()
            if is_one_word(str(lemma.name()))
        ]
        if not hyp_tc_lemmas:
            continue
        # print (curr_lemmas)
        # pprint (hyp_tc_lemmas)
        hyp_pairs.append((curr_lemmas, hyp_tc_lemmas))

    return hyp_pairs

def main(hyp_dir):

    hyp_pairs_1 = gen_wn_hyp(1)

    os.makedirs(hyp_dir, exist_ok = True)

    hypernyms_wn_1_path = os.path.join(hyp_dir, "hypernyms_wn_1.txt")
    with open(hypernyms_wn_1_path, "w") as f:
        for hypos, hypers in hyp_pairs_1:
            if not hypers:
                continue
            hypos_str = ", ".join(hypos)
            hypers_str = ", ".join(hypers)
            f.write(hypos_str + "\t" + hypers_str)
            f.write("\n")

    hyp_pairs_2 = gen_wn_hyp(2)
    hypernyms_wn_2_path = os.path.join(hyp_dir, "hypernyms_wn_2.txt")
    with open(hypernyms_wn_2_path, "w") as f:
        for hypos, hypers in hyp_pairs_2:
            if not hypers:
                continue
            hypos_str = ", ".join(hypos)
            hypers_str = ", ".join(hypers)
            f.write(hypos_str + "\t" + hypers_str)
            f.write("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate_config')
    parser.add_argument('-d', '--hyp_dir', default=None, type=str,
                      help='dir path to hypernym data (default: None)')
    args = parser.parse_args()
    # custom cli options to modify configuration from default values given in json file.
    main(
        args.hyp_dir,
    )