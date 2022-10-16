from re import L
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler

from base import BaseDataLoader
from data_loader.dataset import DmrsDataset, HypEvalDataset
from transform.tcs_transform import TruthConditions
from prepare_train import trsfm_key2abbrev

import json
import os
from collections import defaultdict

from pprint import pprint

class TrainCollator(object):

    def __init__(self, pred2ix):
        self.pred2ix = pred2ix
        self.key2ix = {
            "node2pred": 0,
            "pred_func_nodes": 1,
            "content_preds": 2,
            "logic_expr": 3,
            "pred_func_used": 4
        }

    def __call__(self, instance_batch):
        # each instance in the batch looks like:
            # [
            #     dict: node2pred,
            #     list: pred_func_nodes,
            #     list: content_preds,
            #     list of lists ...: logic_expr
            #     list: pred_func_used
            # ]

        batch_size = len(instance_batch)

        ## if use dict in transformed
        # key2batch = defaultdict(list)
        # for instance in instance_batch:
        #     for key, data in instance.items():
                # key2batch[key].append(data)

        # if use list in transformed
        key_ix2batch = [[] for i in range(len(self.key2ix))]
        for instance in instance_batch:
            for key_ix, data in enumerate(instance):
                key_ix2batch[key_ix].append(data)
                # pprint (key_ix)
                # pprint (data)
            # pprint (key_ix2batch)

        # node2pred_batch = [instance['node2pred'] for instance in instance_batch]
        # encoder_data_batch = [instance['encoders'] for instance in instance_batch]
        # decoders_data_batch = [instance['decoders'] for instance in instance_batch]

        # for decoders_data in decoders_data_batch:
        #     pred_func_used_accum.update(decoders_data['pred_func_used'])
        # print (key_ix2batch)

        pred_func_used_accum = set().union(*[pred_func_used for pred_func_used in key_ix2batch[self.key2ix['pred_func_used']]])
        # encoder
        # TODO: what if targ in content?
        num_targ_preds_batch = torch.tensor([len(pred_func_nodes) for pred_func_nodes in key_ix2batch[self.key2ix['pred_func_nodes']]], dtype = torch.int32)
        num_content_preds_batch = torch.tensor([len(content_preds) for content_preds in key_ix2batch[self.key2ix['content_preds']]], dtype = torch.int32)

        max_num_targ_preds = torch.max(num_targ_preds_batch)
        max_num_content_preds = torch.max(num_content_preds_batch)

        max_num_targ_preds_batch = max_num_targ_preds.expand(batch_size)
        max_num_content_preds_batch = max_num_content_preds.expand(batch_size)

        # targ_preds_ix_batch = torch.tensor([
        #     [self.pred2ix[key2batch[trsfm_key2abbrev['node2pred']][inst_idx][str(targ_node)]] for targ_node in pred_func_nodes] + [0] * (max_num_targ_preds - len(pred_func_nodes))
        #     for inst_idx, pred_func_nodes in enumerate(key2batch[trsfm_key2abbrev['pred_func_nodes']])],
        #     dtype = torch.int32
        # ).unsqueeze(dim = 2)
        try:
            targ_preds_ix_batch = torch.tensor([
                [key_ix2batch[self.key2ix['node2pred']][inst_idx][str(targ_node)] for targ_node in pred_func_nodes] + [0] * (max_num_targ_preds - len(pred_func_nodes))
                for inst_idx, pred_func_nodes in enumerate(key_ix2batch[self.key2ix['pred_func_nodes']])],
                dtype = torch.int32
            ).unsqueeze(dim = 2)
        except:
            print (instance_batch)
            input()
        # print 
        # print (targ_preds_ix_batch.shape)

        # content_preds_ix_batch = torch.tensor([
        #     [self.pred2ix[content_pred] for content_pred in data['content_preds']] + [0] * (max_num_content_preds - len(data['content_preds']))
        #     for data in encoder_data_batch],
        #     dtype = torch.int32
        # ).unsqueeze(dim = 1)
        content_preds_ix_batch = torch.tensor([
            [content_pred for content_pred in content_preds] + [0] * (max_num_content_preds - len(content_preds))
            for content_preds in key_ix2batch[self.key2ix['content_preds']]],
            dtype = torch.int32
        ).unsqueeze(dim = 1)
        # print (content_preds_ix_batch.shape)
        
        # return #encoder_data_batch, decoders_data_batch, targ_preds_ix_batch ...
        return {
            "encoder": [targ_preds_ix_batch, content_preds_ix_batch, num_targ_preds_batch, num_content_preds_batch, max_num_targ_preds_batch],
            "decoder": [key_ix2batch[self.key2ix['logic_expr']], key_ix2batch[self.key2ix['pred_func_nodes']]],
            "pred_func_used_accum": pred_func_used_accum
        }

class DmrsDataLoader(BaseDataLoader):
    """
    DMRS data loading using BaseDataLoader
    """
    def __init__(
        self, data_dir, transformed_dir, batch_size, shuffle = False, num_workers = 0, use_custom_collate_fn = True, pin_memory = False, transformed_dir_suffix = None,
        pred2ix = None, num_replicas = 0, rank = None,
        transform_config_file_path = None, content_pred2cnt = None, pred_func2cnt = None, min_pred_func_freq = 100, min_content_pred_freq = 100, filter_min_freq = True, training = True
    ):
        # with open(transform_config_file_path) as f:
        #     transform_config = json.load(f)
        sample_str = "sample"
        trsfm = None
        # trsfm = TruthConditions(transform_config, min_pred_func_freq, min_content_pred_freq, content_pred2cnt, pred_func2cnt, filter_min_freq = True)
        # transformed_dir = os.path.join(data_dir, transformed_dir_suffix)
        self.dataset = DmrsDataset(data_dir, transformed_dir, transform = trsfm, num_replicas = num_replicas, rank = rank)
        self.sampler = None
        # if num_replicas > 0:
        #     self.sampler = DistributedSampler(self.dataset, num_replicas = num_replicas)
        #     shuffle = False
        collate_fn = lambda x: x
        if use_custom_collate_fn:
            collate_fn = TrainCollator(pred2ix = pred2ix)
        super().__init__(self.dataset, batch_size, shuffle, num_workers, collate_fn = collate_fn, pin_memory = pin_memory, sampler = self.sampler)


class EvalHypCollate(object):

    def __init__(self):
        pass

    def __call__(self, instance_batch):

        batch_size = len(instance_batch)
        instance_batch = torch.stack(instance_batch) 

        targ_preds_ix_batch = instance_batch[:,0].unsqueeze(dim = 1).unsqueeze(dim = 1)
        # print (targ_preds_ix_batch.shape)
        content_preds_ix_batch = instance_batch[:,1].unsqueeze(dim = 1).unsqueeze(dim = 2)
        # print (content_preds_ix_batch.shape)

        decoder_pred_func_ix_batch = instance_batch[:,2:4]

        # targ1_preds_ix_batch
        # targ2_preds_ix_batch

        return [targ_preds_ix_batch, content_preds_ix_batch, decoder_pred_func_ix_batch]


        targ_preds_ix_batch = torch.tensor([
            [key2batch[trsfm_key2abbrev['node2pred']][inst_idx][str(targ_node)] for targ_node in pred_func_nodes] + [0] * (max_num_targ_preds - len(pred_func_nodes))
            for inst_idx, pred_func_nodes in enumerate(key2batch[trsfm_key2abbrev['pred_func_nodes']])],
            dtype = torch.int32
        ).unsqueeze(dim = 2)
        # print (targ_preds_ix_batch.shape)

        # content_preds_ix_batch = torch.tensor([
        #     [self.pred2ix[content_pred] for content_pred in data['content_preds']] + [0] * (max_num_content_preds - len(data['content_preds']))
        #     for data in encoder_data_batch],
        #     dtype = torch.int32
        # ).unsqueeze(dim = 1)
        content_preds_ix_batch = torch.tensor([
            [content_pred for content_pred in content_preds] + [0] * (max_num_content_preds - len(content_preds))
            for content_preds in key2batch[trsfm_key2abbrev['content_preds']]],
            dtype = torch.int32
        ).unsqueeze(dim = 1)


class EvalHypDataLoader(BaseDataLoader):
    """
    DMRS data loading using BaseDataLoader
    """
    def __init__(
        self, hyp_data_dir, hyp_data_path, batch_size, shuffle = False, num_workers = 0, pin_memory = False, num_replicas = 0,
        pred_func2ix = None, pred2ix = None
    ):
        # with open(transform_config_file_path) as f:
        #     transform_config = json.load(f)
        sample_str = "sample"
        trsfm = None
        # trsfm = TruthConditions(transform_config, min_pred_func_freq, min_content_pred_freq, content_pred2cnt, pred_func2cnt, filter_min_freq = True)
        # transformed_dir = os.path.join(data_dir, transformed_dir_suffix)
        self.dataset = HypEvalDataset(hyp_data_path = hyp_data_path, do_trasnform = True, pred_func2ix = pred_func2ix, pred2ix = pred2ix)
        self.sampler = None
        # if num_replicas > 0:
        #     self.sampler = DistributedSampler(self.dataset, num_replicas = num_replicas)
        #     shuffle = False
        # collate_fn = lambda x: x
        collate_fn = EvalHypCollate()
        super().__init__(self.dataset, batch_size, shuffle, num_workers, collate_fn = collate_fn, pin_memory = pin_memory, sampler = self.sampler)
