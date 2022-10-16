import numpy as np
import torch
from torchvision.utils import make_grid
# from base import BaseTrainer
# from utils import inf_loop, MetricTracker

# import model.model as module_arch

from pprint import pprint
import time
import json
import os

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

import matplotlib.pyplot as plt

class Evaluator():
    """
    Evaluator class
    """
    def __init__(self, dataloaders, autoencoder, config, device):

        cfg_evaluator = config['evaluator']
        self.results_dir = cfg_evaluator['results_dir']
        self.truth_thresold = cfg_evaluator['truth_thresold']
        self.autoencoder = autoencoder

        eval_hyp_dataloaders, = dataloaders
        self.eval_hyp_dataloaders = eval_hyp_dataloaders

        self.device = device

    def eval_hyp(self, batch_hyp_results_dir, epoch, batch_idx):

        # fig_path = os.path.join(
        # product of gaussians
        results = {}
        results_metric = {}
        # print (rank, "evaluating")
        with torch.no_grad():
            for file_idx, (hyp_file_name, dataloader) in tqdm(enumerate(self.eval_hyp_dataloaders.items())):
                results[hyp_file_name] = []
                results_metric[hyp_file_name] = []
                for eval_batch_idx, data in enumerate(dataloader):
                    hypo_preds_ix_batch, hyper_preds_ix_batch, decoder_pred_func_ix_batch = data
                    hypo_preds_ix_batch = hypo_preds_ix_batch.to(self.device)
                    hyper_preds_ix_batch = hyper_preds_ix_batch.to(self.device)
                    batch_size = hypo_preds_ix_batch.size(dim = 0)
                    # print (data)
                    mu_hypo_batch, log_sigma2_hypo_batch = self.autoencoder.encoder(hypo_preds_ix_batch, hyper_preds_ix_batch, 1, 1, 1)
                    sigma2_hypo_batch = torch.exp(log_sigma2_hypo_batch)
                    # mu_hyper_batch, log_sigma2_hyper_batch = self.autoencoder.encoder(hyper_preds_ix_batch, hypo_preds_ix_batch, 1, 1, 1)
                    # print ("mu_hypo_batch:", mu_hypo_batch)
                    # print ("sigma2_hypo_batch:", sigma2_hypo_batch)
                    sample_zs_batch = self.autoencoder.sample_from_gauss(mu_hypo_batch, sigma2_hypo_batch, num_samples = 10000)
                    # shape = (bs, 1, num_samples, input_dim)
                    # print ("sample_zs_batch:", sample_zs_batch)
                    # print ("batch_size:", batch_size)
                    # print ("mu_hypo_batch:", mu_hypo_batch.shape)
                    # print ("log_sigma2_hypo_batch:", log_sigma2_hypo_batch.shape)
                    # print ("sample_zs_batch:", sample_zs_batch.shape)

                    hypo_sem_func_batch = decoder_pred_func_ix_batch[:,0]
                    hyper_sem_func_batch = decoder_pred_func_ix_batch[:,1]
                    
                    # print (batch_size)
                    # print (hypo_sem_func_batch.shape)torch.tensor([1.0]
                    # print (sample_zs_batch.shape)
                    fuzzy_truths_hypo_batch = [self.autoencoder.decoder.sem_funcs[hypo_sem_func_batch[inst_idx]](sample_zs_batch[inst_idx])
                        for inst_idx in range(batch_size)
                    ]
                    fuzzy_truths_hyper_batch = [self.autoencoder.decoder.sem_funcs[hyper_sem_func_batch[inst_idx]](sample_zs_batch[inst_idx])
                        for inst_idx in range(batch_size)
                    ]
                    for inst_idx in range(batch_size):
                        # print ("fuzzy_truths_hypo_batch[inst_idx].shape", fuzzy_truths_hypo_batch[inst_idx].shape)
                        # # only consider cases where truthness >= self.truth_thresold
                        fuzzy_truths_hypo_mask = fuzzy_truths_hypo_batch[inst_idx].ge(0.75) # self.truth_thresold
                        true_hypos = torch.masked_select(fuzzy_truths_hypo_batch[inst_idx], fuzzy_truths_hypo_mask)
                        fuzzy_truths_hypers = torch.masked_select(fuzzy_truths_hyper_batch[inst_idx], fuzzy_truths_hypo_mask)
                        is_hyper = torch.ge(fuzzy_truths_hypers, true_hypos)
                        # print ("true_hypos.shape", true_hypos.shape)
                        # print ("fuzzy_truths_hypers.shape", fuzzy_truths_hypers.shape)
                        hypo_ind_nums = len(true_hypos)
                        is_hyper_nums = torch.count_nonzero(is_hyper)
                        # print ("hypo_ind_nums", hypo_ind_nums)
                        # print ("is_hyper_nums", is_hyper_nums)
                        percent_true = is_hyper_nums/hypo_ind_nums
                        results[hyp_file_name].append(percent_true.item())

                # metrics
                print ("sum:", sum(results[hyp_file_name]))
                print ("results:", results[hyp_file_name])
                print ("len:", len(results[hyp_file_name]))
                results_metric[hyp_file_name].append({
                    "mean": sum(results[hyp_file_name])/len(results[hyp_file_name])
                })

        # figures
        fig, axs = plt.subplots(len(results), 1, figsize=(5, 5 * len(results)), sharex=True, sharey=True, tight_layout=True)
        for file_idx, (hyp_file_name, results) in enumerate(results.items()):
            axs[file_idx].set_title(hyp_file_name)
            axs[file_idx].hist(results, bins = 50)
        fig.savefig(os.path.join(batch_hyp_results_dir, "hyp_histograms.png"))
                
        return results_metric

    def eval_poly(self):
        pass

    def eval_relpron(self):
        pass

    def eval_gs(self):
        pass


    def eval(self, results_dir, epoch, batch_idx, len_epoch):
        
        batch_results_dir = os.path.join(results_dir, "epoch" + str(epoch) + "_" + str(int(batch_idx / len_epoch)))
        os.makedirs(batch_results_dir, exist_ok = True)

        t0 = time.time()

        self.autoencoder.encoder.eval()
        for sem_func_ix in range(len(self.autoencoder.decoder.sem_funcs)):
            self.autoencoder.decoder.sem_funcs[sem_func_ix].eval()

        hyp_results_metrics = self.eval_hyp(batch_results_dir, epoch, batch_idx)

        t1 = time.time()
        print ("Evaluation finishes in {}s".format(t1 - t0))

        results_metrics = {
            "hyp": hyp_results_metrics
        }
        
        results_path = os.path.join(batch_results_dir, "metrics")
        with open(results_path, "w") as f:
            json.dump(results_metrics, f)
        # with open(results_path, "r") as f:
        #     results_metrics_load = json.load(f)

        return results_metrics
    