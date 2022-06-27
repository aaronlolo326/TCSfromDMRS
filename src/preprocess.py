import os

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
    
def extract(ww1212_dir, targ_export_dir):
    sys.stderr.write("Extracting export files from {}/ to {}/\n".format(ww1212_dir, targ_export_dir))
    for root, dirs, files in os.walk(ww1212_dir, topdown=False):
        for name in files:
            tar_path = os.path.join(root, name)
            sys.stderr.write("Extracting {}\n".format(tar_path))
            if name.startswith("export") and name.endswith(".tar"):
                export_name = name.split(".")[0]
                tar = tarfile.open(tar_path)
                tar.extractall(os.path.join(targ_export_dir, export_name))

def print_pred(predicate):

    if predicate[0] != "_":
        pass

    elif "unknown" in predicate:
        pred_lemma, pred_pos, *_ = predicate.rsplit("_", 2)
        pred_lemma = pred_lemma.replace('+',' ')[1:]
        print (predicate)
        print (pred_lemma, pred_pos)
        print ()

    elif predicate.count('_') > 3:
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

    if pred_pos not in 'a v n q x p c':
        print (predicate)
        print (pred_lemma, pred_pos)
        pred_lemma, pred_pos, *_ = predicate.rsplit("_", 2)
        pred_lemma = pred_lemma.replace('+',' ')[1:]
        print (pred_lemma, pred_pos)
        print ()



def to_json(targ_export_dir, targ_data_dir):
    '''
    1. Normalize lnk of each DMRS graph
        - 'predicate': 'pron_rel<0:2>' => 'predicate': 'pron_rel'; 'lnk': {'from': 0, 'to': 2}
    2. Save the DMRS graph of each sentence to json
        - exports/export#/uio/wikiwoods/1212/export/20910 => data/dmrs/#_20910.json
        - create the directory from your code, using os.makedirs(<path>, exist_ok = True)
    Note: Refer to Luyi.parse
    '''
    sys.stderr.write("Writing DMRS as json\nfrom: {}\nto: {}\n".format(targ_export_dir, targ_data_dir))
    # code here
    os.makedirs(targ_data_dir, exist_ok = True)
    targ_dir = os.path.join(targ_data_dir, "dmrs")
    os.makedirs(targ_dir, exist_ok = True)
    regexp_1 = re.compile(r'(.+)<(\d+):(\d+)>')# patterm checking
    
    start_process = False
    for root, dirs, files in tqdm(os.walk(targ_export_dir)):
        path = root.split(os.sep)
        if files:
            for file in tqdm(files):
                
                if not start_process:
                    start_process = (file == "09130.gz")
                if not start_process:
                    continue
                  
                dmrs_list = []
                err_cnt = 0
                export_no = root.split("/")[1][-1]
                sys.stderr.write("processing {}".format(os.path.join(root, file)))
                with gzip.open(os.path.join(root, file), "rb") as f:
                    text = f.read().decode('utf-8')
                    structs = text.split("\4")
                    
                    # for each sentence, we process the mrs
                    for idx, struct in enumerate(tqdm(structs)):
                        try:
                            snt, anc, yyinput, drv, cat, mrs, eds, dmrs_noScope, _ = struct.split("\n\n")
                        except:
                            sys.stderr.write("wrongly formatted instance:")
                            print (struct)
                        try:
                            dmrs = from_mrs(simplemrs.decode(mrs))
                            dmrs_json = dmrsjson.to_dict(dmrs)
                        except MRSSyntaxError as e:
                            err_cnt += 1
                        # change the format
                        for node in dmrs_json['nodes']:
                            
                            re_match = regexp_1.match(node['predicate'])
                            my_list = node['predicate'].split("<")
                            if re_match:
                                node['predicate'] = my_list[0]
                                my_list = my_list[1].split(":")
                                item_1 = int(my_list[0])
                                my_list = my_list[1].split(">")
                                item_2 = int(my_list[0])
                                node['lnk'] = {'from':item_1,'to':item_2}
 
                        dmrs_list.append(dmrs_json)
                targ_filename = "{}_{}.json".format(export_no, file[:-3])
                with open(os.path.join(targ_dir, targ_filename), "w") as f:
                    json.dump(dmrs_list, f)
                sys.stderr.write("No. of err-ed DMRS: {}\n".format(err_cnt))
                # input()
            # break
                
                    
#     with gzip.open('file.txt.gz', 'rb') as f:
#         file_content = f.read()
    
#     with open("exports/export0/uio/wikiwoods/1212/export/20910", encoding = 'utf-8') as f:
#         text = f.read()
#         structs = text.split("\4")
#         for idx, struct in enumerate(tqdm(structs)):
#             snt, anc, yyinput, drv, cat, mrs, eds, dmrs_noScope, _ = struct.split("\n\n")
#             try:
#                 dmrs = from_mrs(simplemrs.decode(mrs))
#                 dmrs_json = dmrsjson.to_dict(dmrs)
#             except MRSSyntaxError as e:
#                 err_cnt += 1
#                 pass
#     with open():
#         pass

    
    # with open(targ_data_dir
        
def main(ww1212_dir, targ_export_dir, targ_data_dir):
    # extract(ww1212_dir, targ_export_dir)
    to_json(targ_export_dir, targ_data_dir)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ww1212_dir', default=None, help='path to wikiwoods 1212 directory')
    parser.add_argument('targ_export_dir', default=None, help='path to target export directory')
    parser.add_argument('targ_data_dir', default=None, help='path to target data directory')
    args = parser.parse_args()
    main(args.ww1212_dir)
    
'''
module_name, package_name, ClassName, method_name, ExceptionName, function_name, GLOBAL_CONSTANT_NAME, global_var_name, instance_var_name, function_parameter_name, local_var_name.
'''

#json_dir = "data"
def get_predicate_from_json(json_dir):
    print (123145)
        
    
    
    pred_list = []
    for root, dirs, files in tqdm(os.walk(json_dir)):
        if files:
            for file in files:
                if file == '0_00120.json':
                    with open(os.path.join(root, file), "r") as f:
                        exported_dictionary = json.loads(f.read())
                        pprint (exported_dictionary[:10])
                        # for root in exported_dictionary['root']:
                        #     for node in root['nodes']:
                        #         pred_list.append(node['predicate'])

    # return pred_list  
                   
                
    