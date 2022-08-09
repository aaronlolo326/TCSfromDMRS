import os

from delphin import itsdb
from delphin.codecs import simplemrs

import argparse
import sys
import re
from pprint import pprint
from collections import defaultdict, Counter
import json

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm
    
    
def preprocess_itsdb(wwItsdb_dir):
    sys.stderr.write("Preprocessing itsdbs\n")
    sntid2item = defaultdict(dict)
    filt_sntid2item = defaultdict(dict)
    sntid2cnt = Counter()
    amb_sntid = set()
    bad_mrs_sntid = set()
    bad_syn_tree_sntid = set()
    bad_deriv_sntid = set()
    no_parse_sntid = set()
    unusable_sntid = set()
    usable_sntid = set()
    # ww/itsdb2/uio/wikiwoods/1212/tsdb/00102
    for root, dirs, files in os.walk(wwItsdb_dir):
        if 'itsdb0' in root or 'itsdb1' in root or 'itsdb2' in root:
            continue
        if 'item.gz' in files:
            profile = root.split("/")[-1]
            ts = itsdb.TestSuite(root)
            for key in ts:
                for idx, row in enumerate(ts[key]):
                    for row_key in row.keys():
                        # print (key, row_key, row[row_key])
                        if row_key == 'mrs':
                            print (row[row_key])
                    break
            
            # input()
#             # add sentence
#             print (len(ts['item']))
#             for idx, item in enumerate(ts['item']):
#                 for key in item.keys():
#                     print (key, item[key])
#                 input()
#                 # print (item)
#                 key = str(item['i-id'])
#                 sntid2cnt[key] += 1
#                 sntid2item[key] = {"profile": profile, "sentence": item['i-input']}
#             print ()

#             # add syntactic tree, derivation and dmrs for each sentence
#             for idx, result in enumerate(ts['result']):
#                 for key in result.keys():
#                     print (key, result[key])
#                 if any([k in sntid2item[key] for k in ['mrs_str', 'syn_tree', 'derivation', 'dmrs_json']]):
#                     amb_sntid.add(key)
#                     pass 
#                 else:
#                     try:
#                         mrs_str = result['mrs']
#                         print (mrs_str)
#                         mrs = simplemrs.loads(mrs_str)
#                         dmrs_str = delphin.dmrs.from_mrs(mrs[0])
#                         print (dmrs_str)
#                         dmrs_json = json.loads(dmrsjson.encode(dmrs_str, indent=True))
#                     except:
#                         bad_mrs_sntid.add(key)
#                     try:
#                         sntid2item[key]['syn_tree'] = delphin.util.SExpr.parse(result['tree']).data
#                         print (sntid2item[key]['syn_tree'])
#                     except Exception as e:
#                         bad_syn_tree_sntid.add(key)
#                     try:
#                         sntid2item[key]['derivation'] = delphin.derivation.from_string(result['derivation'])
#                         print (ntid2item[key]['derivation'])
#                     except Exception as e:
#                         bad_deriv_sntid.add(key)
#             break

def preprocess_export(wwItsdb_dir):
    

def inspect():
    orig_data_path = os.path.join("data", "data.txt")
    with open(orig_data_path) as f:
        lines = f.readlines()
        print (lines)
        input()
        
def main(wwItsdb_dir):
    preprocess(wwItsdb_dir)
    inspect()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wwItsdb_dir', default=None, help='path to wikiwoods 1212 directory')
    args = parser.parse_args()
    main(args.wwItsdb_dir)
    
'''
module_name, package_name, ClassName, method_name, ExceptionName, function_name, GLOBAL_CONSTANT_NAME, global_var_name, instance_var_name, function_parameter_name, local_var_name.
'''
 