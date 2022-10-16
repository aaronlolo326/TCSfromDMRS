import os

import json
import torch
from torch.utils.data import Dataset, IterableDataset

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm
    
import networkx as nx

from pprint import pprint
from src import util

import sys
from pympler import asizeof

    
class DmrsDataset(Dataset):
    
    def __init__(self, data_dir, transformed_dir, transform = None, num_replicas = 0, rank = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        ## Lazy loading
        # self.data_dir = data_dir
        # data_info_dir = os.path.join(data_dir, "info")
        # with open(os.path.join(data_info_dir, "idx2file_path.json")) as f:
        #     idx2file_path = json.load(f)
        # self.idx2file_path = idx2file_path
        # self.transform = transform
        
        ## Load all at once

        self.data_dir = data_dir
        self.transformed_dir = transformed_dir
        self.transform = transform
        self.instance_list = []
        self.transformed_list = []
        self.sub_transformed_list = []
        self.num_instance = 0
        self.rank = rank
        self.num_replicas = num_replicas
        self.unloaded_files = []

        # if transform:
        #     for root, dirs, files in tqdm(os.walk(self.data_dir)):
        #         for file in files:
        #             if not util.is_data_json(file):
        #                 continue
        #             with open(os.path.join(self.data_dir, file)) as f:
        #                 idx2instance = json.load(f)
        #                 self.num_instance += len(idx2instance)
        #                 self.instance_list.extend(list(idx2instance.values()))

        ## elif transformed
        if True:
            for file in os.listdir(self.transformed_dir):
                if all([
                    os.path.isfile(os.path.join(self.transformed_dir, file)),
                    file.startswith("transformed_"),
                    util.is_data_json(file)
                ]):
                # for splitting dataset based on rank
                    # print (file)
                    suffix = int(file.rsplit(".", 1)[0].split("_")[1])
                    # print (suffix)
                    # print (num_replicas, rank)
                    if rank == 'cpu' or num_replicas == 0:
                        self.unloaded_files.append(file)
                    elif rank == None or suffix % num_replicas == rank:
                        self.unloaded_files.append(file)
            for file in tqdm(self.unloaded_files):
                with open(os.path.join(self.transformed_dir, file)) as f:
                    print (os.path.join(self.transformed_dir, file))
                    transformed = json.load(f)
                    # print ("transformed: ", asizeof.asizeof(transformed))
                    # print ("len(transformed): ", (len(transformed)))
                    self.num_instance += len(transformed)
                    self.transformed_list.extend(transformed)
                    # print ("transformed_list: ", asizeof.asizeof(self.transformed_list))
                    print ("len(transformed_list): ", len(self.transformed_list))
                    del transformed
                    # if self.sub_transformed_list == []:
                    
        # with open(os.path.join(data_info_dir, "idx2file_path.json")) as f:
        #     idx2file_path = json.load(f)
        # self.idx2file_path = idx2file_path
        # self.transform = transform       
        

    def __len__(self):

        return self.num_instance 
        # if self.transform:
        #     return len(self.instance_list)
        # else:
        #     return len(self.transformed_list)

    def __getitem__(self, idx):
        
        ## Lazy loading
        # if self.sub_transformed_list == []:
        #     if self.unloaded_files == []:
        #         # no more data
        #         pass
        #     else:
        #         file = self.unloaded_files.pop(0)
        #         with open(os.path.join(self.transformed_dir, file)) as f:
        #             transformed = json.load(f)
        #             self.sub_transformed_list = transformed

        # return self.sub_transformed_list[idx]

        ## Load all at once
        if self.transform:
            sample = self.instance_list[idx]
            sample = self.transform(sample)
        else:
            sample = self.transformed_list[idx]
            
        return sample


class DmrsIterDataset(IterableDataset):
    
    def __init__(self, data_dir, transformed_dir, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        ## Lazy loading
        # self.data_dir = data_dir
        # data_info_dir = os.path.join(data_dir, "info")
        # with open(os.path.join(data_info_dir, "idx2file_path.json")) as f:
        #     idx2file_path = json.load(f)
        # self.idx2file_path = idx2file_path
        # self.transform = transform
        
        ## Load all at once

        self.data_dir = data_dir
        self.transformed_dir = transformed_dir
        self.transform = transform
        self.transformed_list = []
        self.sub_transformed_list = []

        # if transform:
        #     for root, dirs, files in tqdm(os.walk(self.data_dir)):
        #         for file in files:
        #             if not util.is_data_json(file):
        #                 continue
        #             with open(os.path.join(self.data_dir, file)) as f:
        #                 idx2instance = json.load(f)
        #                 self.num_instance += len(idx2instance)
        #                 self.instance_list.extend(list(idx2instance.values()))

        ## elif transformed
        if True:
            self.unloaded_files = [file for file in os.listdir(self.transformed_dir) if all([
                os.path.isfile(os.path.join(self.transformed_dir, file)),
                file.startswith("transformed_"),
                util.is_data_json(file)
                ])
            ]
                    
        # with open(os.path.join(data_info_dir, "idx2file_path.json")) as f:
        #     idx2file_path = json.load(f)
        # self.idx2file_path = idx2file_path
        # self.transform = transform       

    def __iter__(self):
        
        # Lazy loading
        if self.sub_transformed_list == []:
            if self.unloaded_files == []:
                # no more data
                pass
            else:
                file = self.unloaded_files.pop(0)
                with open(os.path.join(self.transformed_dir, file)) as f:
                    transformed = json.load(f)
                    self.sub_transformed_list = transformed

        # return self.sub_transformed_list[idx]

        ## Load all at once
        if self.transform:
            sample = self.instance_list[idx]
            sample = self.transform(sample)
        else:
            sample = self.transformed_list[idx]
            
        return sample


class HypEvalDataset(Dataset):
            
    def __init__(self, hyp_data_path = None, do_trasnform = True, pred_func2ix = None, pred2ix = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.hyp_data_path = hyp_data_path
        self.hyp_pred_pairs = []
        self.pred_func2ix = pred_func2ix
        self.pred2ix = pred2ix
        self.do_trasnform = do_trasnform
        if True:
            with open(self.hyp_data_path) as f:
                hyp_pred_pairs = json.load(f)
                self.hyp_pred_pairs = hyp_pred_pairs

    def __len__(self):
        return len(self.hyp_pred_pairs)

    def __getitem__(self, idx):
        if not self.do_trasnform: 
            sample = self.hyp_pred_pairs[idx]
        else:
            # print (idx)
            # print (self.hyp_pred_pairs)
            sample_list = [
                *[self.pred2ix[pred] for pred in self.hyp_pred_pairs[idx]],
                # *[self.pred2ix[pred] for pred in self.hyp_pred_pairs[idx][::-1]],
                *[self.pred_func2ix[pred + "@ARG0"] for pred in self.hyp_pred_pairs[idx]]
            ]
            sample = torch.tensor(sample_list, dtype = torch.int32)
        return sample
