from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.dataset import DmrsDataset
from transform.tcs_transform import TruthConditions
import json



class DmrsDataLoader(BaseDataLoader):
    """
    DMRS data loading using BaseDataLoader
    """
    def __init__(self, data_dir, transform_config_file_path, batch_size, shuffle = True, validation_split = 0.0, num_workers = 1,
        min_pred_func_freq = 100, min_lex_pred_freq = 100, use_custom_collate_fn = True,
        lex_pred2cnt = None, pred_func2cnt = None, training = True):
        with open(transform_config_file_path) as f:
            transform_config = json.load(f)
        trsfm = TruthConditions(transform_config, min_pred_func_freq, min_lex_pred_freq, lex_pred2cnt, pred_func2cnt, filter_min_freq = True)
        self.dataset = DmrsDataset(data_dir, transform = trsfm, sample_only = True)
        collate_fn = None
        if use_custom_collate_fn:
            collate_fn = lambda x: x
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn = collate_fn)
