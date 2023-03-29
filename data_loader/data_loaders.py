from re import L
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler

from base import BaseDataLoader
from data_loader.dataset import TrainDataset, EvalHypDataset, EvalRelpronDataset, EvalGS2011Dataset # FLDataset
import data_loader.collators as collators
from transform.tcs_transform import TruthConditions
from prepare_train import trsfm_key2abbrev

import json
import os
from collections import defaultdict

from pprint import pprint

class TrainDataLoader(BaseDataLoader):
    """
    DMRS data loading using BaseDataLoader (MyEncoder)
    """
    def __init__(
        self, data_dir, transformed_dir, batch_size, shuffle = False, num_workers = 0, collate_fn = None, pin_memory = False, transformed_dir_suffix = None,
        pred2ix = None, num_replicas = 0, rank = None, device = None,
        transform_config_file_path = None, content_pred2cnt = None, pred_func2cnt = None, min_pred_func_freq = 100, min_content_pred_freq = 100, filter_min_freq = True, training = True
    ):
        sample_str = "sample"
        trsfm = None
        self.dataset = TrainDataset(data_dir, transformed_dir, transform = trsfm, num_replicas = num_replicas, rank = rank, device = device)
        self.sampler = None
        collate_fn = getattr(collators, collate_fn)(pred2ix = pred2ix)
        super().__init__(self.dataset, batch_size, shuffle, num_workers, collate_fn = collate_fn, pin_memory = pin_memory, sampler = self.sampler)


class EvalRelpronDataLoader(BaseDataLoader):
    """
    DMRS data loading using BaseDataLoader
    """
    def __init__(
        self, relpron_data_dir, relpron_data_path, split, svo, batch_size, shuffle = False, num_workers = 0, pin_memory = False, collate_fn = None, num_replicas = 0,
        pred_func2ix = None, pred2ix = None, encoder_arch_type = None
    ):
        sample_str = "sample"
        trsfm = None
        self.dataset = EvalRelpronDataset(relpron_data_path = relpron_data_path, svo = svo, do_trasnform = True, pred_func2ix = pred_func2ix, pred2ix = pred2ix, encoder_arch_type = encoder_arch_type)
        self.sampler = None
        # if num_replicas > 0:
        #     self.sampler = DistributedSampler(self.dataset, num_replicas = num_replicas)
        #     shuffle = False
        # collate_fn = lambda x: x
        collate_fn = getattr(collators, collate_fn)()
        super().__init__(self.dataset, batch_size, shuffle, num_workers, collate_fn = collate_fn, pin_memory = pin_memory, sampler = self.sampler)


class EvalGS2011DataLoader(BaseDataLoader):
    """
    DMRS data loading using BaseDataLoader
    """
    def __init__(
        self, data_dir, data_path, batch_size, shuffle = False, num_workers = 0, pin_memory = False, collate_fn = None, num_replicas = 0,
        pred_func2ix = None, pred2ix = None, encoder_arch_type = None
    ):
        sample_str = "sample"
        trsfm = None
        self.dataset = EvalGS2011Dataset(data_path = data_path, do_trasnform = True, pred_func2ix = pred_func2ix, pred2ix = pred2ix, encoder_arch_type = encoder_arch_type)
        self.sampler = None
        # if num_replicas > 0:
        #     self.sampler = DistributedSampler(self.dataset, num_replicas = num_replicas)
        #     shuffle = False
        # collate_fn = lambda x: x
        collate_fn = getattr(collators, collate_fn)()
        super().__init__(self.dataset, batch_size, shuffle, num_workers, collate_fn = collate_fn, pin_memory = pin_memory, sampler = self.sampler)

