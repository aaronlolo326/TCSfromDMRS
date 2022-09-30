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
                        
        
def main(ww1212_dir, targ_export_dir):
    extract(ww1212_dir, targ_export_dir)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ww1212_dir', default=None, help='path to wikiwoods 1212 directory')
    parser.add_argument('--targ_export_dir', default=None, help='path to target export directory')
    args = parser.parse_args()
    main(
        args.ww1212_dir,
        args.targ_export_dir
    )
    
'''
module_name, package_name, ClassName, method_name, ExceptionName, function_name, GLOBAL_CONSTANT_NAME, global_var_name, instance_var_name, function_parameter_name, local_var_name.
'''
                
    