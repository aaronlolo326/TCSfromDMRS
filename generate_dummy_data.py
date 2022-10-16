from delphin import ace
from delphin import scope
from delphin.dmrs import from_mrs
from delphin.codecs import simplemrs, dmrsjson

from src import dg_util, util

import os
import json
from collections import defaultdict, Counter
from pprint import pprint

import nltk
# nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

from networkx.readwrite.json_graph import node_link_data
from networkx.drawing.nx_agraph import to_agraph

os.environ["LC_ALL"] ="en_US.UTF-8"
os.environ["LC_CTYPE"] ="en_US.UTF-8"

erg_dir = "./erg"
erg_path = "./erg/erg-1214-x86-64-0.9.34.dat"
dummy_data_dir = "./data/dummy_data/"
dummy_data_fig_dir = os.path.join(dummy_data_dir, "figures")
data_json_filename = "0_00000.json"

unk2pos = None
unk2pos_path = os.path.join(erg_dir, "unk2pos.json")
with open(unk2pos_path) as f:
    unk2pos = json.load(f)

idx2instance = defaultdict(defaultdict)
idx2file_path = defaultdict()
err2cnt = Counter()
pred2cnt = Counter()

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
    return norm_pred


text_no_parse = [
    # ['Dogs love cats.', 0],
    # ['Dogs hope to want to eat fish', 0],
    # ['Happy or sad dogs run or swim.', 0],
    # ['Dogs, cats, lions or ducks swim.', 0],
    # ['Happy dogs, cats and lions run.', 2],
    # ['Dogs run, swim or bark very happily or sadly.', 9],
    # ['Dogs listen, swim, eat or bark very happily or sadly.', 2],
    # ['Big and small dogs run or swim very happily or sadly.', 3],
    # ['Dogs or lions and bears or cats learn, lose and eat.', 0],
    # ['Dogs can probably not be running very happily.', 0], # neg
    # ['No humans could survive with no water or oxygen.', 0],
    # ['happy dogs and sad cats that run eat no apples and oranges', 0], # prb
    # ['large pet dogs with paws which no normal people are afraid of are gentle if dogs are raised properly or not trained.', 0],
    # ['dogs or cats that run or swim eat food', 0],
    # ['If bears and sad cats eat quickly or slowly, birds run.', 0],
    # ['Kim knows that Sandy did not leave.', 0],
    # ['Dogs want to run, play or swim happily, immediately or sadly.', 0],
    # ['Dogs like running or swimming happily', 16],
    # ['Lions know no information about no dogs and cats', 0],
    # ['no dogs and cats eat no apples and oranges', 0],
    # ['no dogs and cats want to eat no apples and oranges', 0],
    # ['It is raining.', 0],
    # ['Every dog that does not bark does not love cats that do not run.', 0],
    # ['Not every dog runs.', 0],
    # ['Cats, dogs and lions that do not run do not walk', 1],
    # ['Software engineers love programming.', 2],
    # ["Dogs bark or don't bark happily", 0],
    # ['I am the claim that dogs do not run and swim', 0],
    # ['Every dog did not leave.', 0],
    # ['I want to eat all apples or oranges like all dogs or cats', 0],
    # ['Not working out or not eating enough proteins are not all the reasons for not gaining muscles or not graduating', 0],
    # ['Do not let nobody not eat no food', 0],
    # ['My mum or I made a sandwich or chicken for breakfast and lunch', 0],
    # ['A cat can either be a house cat, a farm cat, or a feral cat.', 0],
    # ['Lions can be cats, animals and information about books', 0]
    ['He is cats.', 0],
    ['Dogs are cats.', 0]
]

os.makedirs(dummy_data_dir, exist_ok = True)
os.makedirs(dummy_data_fig_dir, exist_ok = True)

figs_path = []

for no_instance, text_no_parse in enumerate(text_no_parse):

    snt, no_parse = text_no_parse 

    ace_results = ace.parse(erg_path, snt)

    mrs = ace_results['results'][no_parse]['mrs']

    dmrs = from_mrs(simplemrs.decode(mrs))
    dmrs_json = dmrsjson.to_dict(dmrs)

    for node in dmrs_json['nodes']:      
        norm_pred = normalize_pred(node['predicate'], unk2pos)
        # node['predicate'] = pred
        node['predicate'] = norm_pred
        pred2cnt[norm_pred] += 1

    erg_digraphs = dg_util.Erg_DiGraphs()
    draw = False
    # uncomment this line for debugging
    # draw = True
    
    erg_digraphs.init_dmrsjson(dmrs_json, minimal_prop = True)
    erg_digraphs.init_snt(snt)
    
    # cleanse dmrs
    erg_digraphs_re = erg_digraphs
    # erg_digraphs_re = dmrs_rewrite(snt_id, snt, erg_digraphs_re, rewrite_type = 'modal')
    # erg_digraphs_re = dmrs_rewrite(snt_id, snt, erg_digraphs_re, rewrite_type = 'nominalization')
    # erg_digraphs_re = dmrs_rewrite(snt_id, snt, erg_digraphs_re, rewrite_type = 'eventuality')
    # erg_digraphs_re = dmrs_rewrite(snt_id, snt, erg_digraphs_re, rewrite_type = 'compound')
    
    idx2instance[no_instance]['snt'] = erg_digraphs_re.snt
    idx2instance[no_instance]['id'] = no_instance
    idx2instance[no_instance]['dmrs'] = node_link_data(erg_digraphs_re.dmrs_dg)

    idx2file_path[no_instance] = data_json_filename

    save_path = os.path.join(dummy_data_fig_dir, "dmrs_{}.png".format(no_instance))#+ time.asctime( time.localtime(time.time()) ).replace(" ", "-") +".png"
    erg_digraphs_re.draw_dmrs(save_path = save_path)
    figs_path.append(save_path)


targ_data_info_dir = os.path.join(dummy_data_dir, "info")
os.makedirs(targ_data_info_dir, exist_ok = True)

with open(os.path.join(targ_data_info_dir, "err2cnt.txt"), "w") as f:
    f.write(str(err2cnt))
with open(os.path.join(targ_data_info_dir, "pred2cnt.txt"), "w") as f:
    for pred, cnt in pred2cnt.most_common():
        f.write("{}\t{}\n".format(pred, str(cnt)))
with open(os.path.join(targ_data_info_dir, "idx2file_path.json"), "w") as f:
    json.dump(idx2file_path, f)

pprint (figs_path)
                
with open(os.path.join(dummy_data_dir, data_json_filename), "w") as f:
    json.dump(idx2instance, f, indent = 2)