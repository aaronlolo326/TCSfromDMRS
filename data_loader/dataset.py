import os
from collections import Counter

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

class TrainDataset(Dataset):
    def __init__(self, data_dir, transformed_dir, transform = None, num_replicas = 0, rank = None, device = None):
        """
        Args:
            data_dir (string): Path to the csv file with annotations.
            transformed_dir (string): Directory to the transformed data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
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

        if True:
            for file in os.listdir(self.transformed_dir):
                if all([
                    os.path.isfile(os.path.join(self.transformed_dir, file)),
                    file.startswith("transformed_"),
                    util.is_data_json(file)
                ]):
                    suffix = int(file.rsplit(".", 1)[0].split("_")[1])
                    if num_replicas == 0:
                        self.unloaded_files.append(file)
                    elif rank == None or suffix % num_replicas == rank:
                        self.unloaded_files.append(file)
            for file in tqdm(self.unloaded_files):
                with open(os.path.join(self.transformed_dir, file)) as f:
                    print (os.path.join(self.transformed_dir, file))
                    transformed = json.load(f)
                    self.num_instance += len(transformed)
                    self.transformed_list.extend(transformed)
                    del transformed

    def __len__(self):

        return self.num_instance 

    def __getitem__(self, idx):

        ## Load all at once
        if self.transform:
            sample = self.instance_list[idx]
            sample = self.transform(sample)
        else:
            sample = self.transformed_list[idx]
            
        return sample



class EvalHypDataset(Dataset):
            
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

class EvalRelpronDataset(Dataset):
            
    def __init__(self, relpron_data_path = None, svo = True, do_trasnform = True, pred_func2ix = None, pred2ix = None, encoder_arch_type = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.relpron_data_path = relpron_data_path
        self.pred_func2ix = pred_func2ix
        self.pred2ix = pred2ix
        self.do_trasnform = do_trasnform
        if True:
            with open(self.relpron_data_path) as f:
                relpron_split = json.load(f)
                self.relpron_split = relpron_split
                # relpron_splits[split] = {
                #     "pred_func_nodes_ctxt_preds_list": pred_func_nodes_ctxt_preds_list,
                #     "pred_func_nodes_ctxt_args_list": pred_func_nodes_ctxt_args_list,
                #     "logic_expr_list": logic_expr_list,
                #     "vars_unzipped_list": vars_unzipped_list,
                #     "labels": term2filter_props_idx[split],
                #     "full_props": filtered_relpron_props[split],
                #     "terms": terms
                # }
                self.terms = relpron_split['terms']
                self.labels = relpron_split['labels']
                self.full_props = relpron_split['full_props']

                if encoder_arch_type == "MyEncoder":
                    self.pred_func_nodes_ctxt_predORpredargs_list = relpron_split['pred_func_nodes_ctxt_preds_list']
                elif encoder_arch_type == "PASEncoder":
                    self.pred_func_nodes_ctxt_predORpredargs_list = relpron_split['pred_func_nodes_ctxt_predargs_list']
                self.pred_func_nodes_ctxt_args_list = relpron_split['pred_func_nodes_ctxt_args_list']
                self.logic_expr_list = relpron_split['logic_expr_list']
                self.vars_unzipped_list = relpron_split['vars_unzipped_list']
                

    def __len__(self):
        return len(self.pred_func_nodes_ctxt_predORpredargs_list)

    def __getitem__(self, idx):
        if not self.do_trasnform: 
            sample = self.relpron_split
        else:
            sample = [
                self.pred_func_nodes_ctxt_predORpredargs_list[idx], self.pred_func_nodes_ctxt_args_list[idx],
                self.logic_expr_list[idx], self.vars_unzipped_list[idx],
                self.terms[idx],
                self.labels, self.full_props  # for eval and display of reulsts
            ]

        return sample

class EvalGS2011Dataset(Dataset):
            
    def __init__(self, data_path = None, do_trasnform = True, pred_func2ix = None, pred2ix = None, encoder_arch_type = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # gs_eval = {
        #     "svo_ix2landmark2scores": gs2011_svo_ix2landmark2scores,
        #     "ix2svo": gs2011_ix2svo,
        #     "pred_func_nodes_ctxt_preds_list": pred_func_nodes_ctxt_preds_list,
        #     "pred_func_nodes_ctxt_args_list": pred_func_nodes_ctxt_args_list,
        #     "svo_ix2landmark2logic_expr": gs2011_svo_ix2landmark2logic_expr,
        #     "vars_unzipped_list": vars_unzipped_list,
        # }
        self.data_path = data_path
        self.pred_func2ix = pred_func2ix
        self.pred2ix = pred2ix
        self.do_trasnform = do_trasnform
        with open(self.data_path) as f:
            gs_eval = json.load(f)
            self.gs_eval = gs_eval

            if encoder_arch_type == "MyEncoder":
                self.pred_func_nodes_ctxt_predORpredargs_list = gs_eval['pred_func_nodes_ctxt_preds_list']
            elif encoder_arch_type == "PASEncoder":
                self.pred_func_nodes_ctxt_predORpredargs_list = gs_eval['pred_func_nodes_ctxt_predargs_list']
            self.pred_func_nodes_ctxt_args_list = gs_eval['pred_func_nodes_ctxt_args_list']
            self.svo_ix2landmark2logic_expr = gs_eval['svo_ix2landmark2logic_expr']
            self.vars_unzipped_list = gs_eval['vars_unzipped_list']
            self.svo_ix2landmark2scores = gs_eval['svo_ix2landmark2scores']
            self.ix2svo = gs_eval['ix2svo']

            self.svo_ix2landmark2logic_expr = {int(key): value for key, value in self.svo_ix2landmark2logic_expr.items()}
            self.svo_ix2landmark2scores = {int(key): value for key, value in self.svo_ix2landmark2scores.items()}
            self.ix2svo = {int(key): value for key, value in self.ix2svo.items()}
            

    def __len__(self):
        return len(self.pred_func_nodes_ctxt_predORpredargs_list)

    def __getitem__(self, idx):
        if not self.do_trasnform: 
            pass
        else:
            sample = [
                self.pred_func_nodes_ctxt_predORpredargs_list[idx], self.pred_func_nodes_ctxt_args_list[idx],
                self.svo_ix2landmark2logic_expr[idx], self.vars_unzipped_list[idx],
                self.svo_ix2landmark2scores[idx], self.ix2svo
            ]
# [tensor([[5157, 2908,  976],        [2908, 5157,  976],
#         [ 976, 2908, 5157]], dtype=torch.int32),
#  tensor([[1, 6, 7],
#         [1, 2, 0],
#         [1, 0, 3]], dtype=torch.int32),
#  tensor([[ 423, 2572,  956]], dtype=torch.int32),
#  tensor([[1, 1, 1],
#         [0, 2, 3]], dtype=torch.int32),
#  [3, 1, 2, 3, 1, 1, 2, 2, 2, 4, 2, 2, 2]]

        return sample


class EvalWeeds2014Dataset(Dataset):
            
    def __init__(self, data_path = None, do_trasnform = True, pred_func2ix = None, pred2ix = None, encoder_arch_type = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_path = data_path
        self.pred_func2ix = pred_func2ix
        self.pred2ix = pred2ix
        self.do_trasnform = do_trasnform
        with open(self.data_path) as f:
            weeds2014_eval = json.load(f)
            self.weeds2014_eval = weeds2014_eval

            self.pred_func_nodes_ctxt_predargs_list = weeds2014_eval['pred_func_nodes_ctxt_predargs_list']
            self.pred_func_nodes_ctxt_args_list = weeds2014_eval['pred_func_nodes_ctxt_args_list']
            self.pred_func_nodes_ctxt_predargs_cls_list = weeds2014_eval['pred_func_nodes_ctxt_predargs_cls_list']
            self.pred_func_nodes_ctxt_args_cls_list = weeds2014_eval['pred_func_nodes_ctxt_args_cls_list']
            self.pred_funcs_list = weeds2014_eval['pred_funcs_list']
            self.vars_unzipped_list = weeds2014_eval['vars_unzipped_list']
            self.ix2pair = weeds2014_eval['ix2pair']
            self.ix2lbl = weeds2014_eval['ix2lbl']


    def __len__(self):
        return len(self.pred_func_nodes_ctxt_predORpredargs_list)

    def __getitem__(self, idx):
        if not self.do_trasnform: 
            pass
        else:
            sample = [
                self.pred_func_nodes_ctxt_predargs_list[idx], self.pred_func_nodes_ctxt_args_list[idx],
                self.pred_func_nodes_ctxt_predargs_cls_list[idx], self.pred_func_nodes_ctxt_args_cls_list[idx],
                self.pred_funcs_list[idx], self.vars_unzipped_list[idx],
                self.ix2lbl[idx], self.ix2pair[idx]
            ]

        return sample
