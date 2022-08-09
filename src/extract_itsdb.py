import tarfile
import gzip
import os
import shutil

import argparse
import sys

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm
    
def extract(wwTar_dir):
    sys.stderr.write("Extracting itsdbs\n")
    for i in range(0,10):
        itsdb_name = 'itsdb' + str(i)
        itsdb_tar_name = itsdb_name + ".tar"
        itsdb_tar_path = os.path.join(wwTar_dir, itsdb_tar_name)
        file_obj = tarfile.open(itsdb_tar_path)
        file = file_obj.extractall(os.path.join("ww", itsdb_name))
        file_obj.close()
        
def main(wwTar_dir):
    extract(wwTar_dir)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wwTar_dir', default=None, help='path to wikiwoods 1212 tar files directory')
    args = parser.parse_args()
    main(args.wwItsdb_dir)
    
'''
module_name, package_name, ClassName, method_name, ExceptionName, function_name, GLOBAL_CONSTANT_NAME, global_var_name, instance_var_name, function_parameter_name, local_var_name.
'''
 