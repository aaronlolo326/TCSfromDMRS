import numpy as np
import torch
from pprint import pprint
import time
import json
import os
from collections import defaultdict

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

import matplotlib.pyplot as plt

from scipy.stats import spearmanr

class Evaluator():
    """
    Evaluator class
    """
    def __init__(self, results_dir, dataloaders, autoencoder, config, device):

        # cfg_evaluator = config['evaluator']
        self.config = config
        self.results_dir = results_dir
        # self.truth_thresold = cfg_evaluator['truth_thresold']
        self.autoencoder = autoencoder
        self.with_logic = config['with_logic']

        eval_relpron_dataloaders, eval_gs2011_dataloaders, eval_gs2013_dataloaders, eval_gs2012_dataloaders = dataloaders
        # self.eval_hyp_dataloaders = eval_hyp_dataloaders
        self.eval_relpron_dataloaders = eval_relpron_dataloaders
        self.eval_gs2011_dataloaders = eval_gs2011_dataloaders
        self.eval_gs2013_dataloaders = eval_gs2013_dataloaders
        self.eval_gs2012_dataloaders = eval_gs2012_dataloaders

        self.device = device
        self.write_file = True

    def write_file_off():
        self.write_file = False

    def _eval_relpron_map(self, term2truth_of_props, relpron_file_name):
        
        term2props = defaultdict(list)
        confounders = []
        mean_ap = 0
        confounder_ranks = []
        # compute MAP
        ap = []
        # sort the truths to get the ranked prop_idx
        for term_idx, (term, truth_of_props) in enumerate(term2truth_of_props.items()):
            ap.append(0.0)
            ranked = sorted(range(len(truth_of_props)), key = lambda i: truth_of_props[i], reverse = True)
            # compute AP
            correct_at_k = 0
            for prop_rank, prop_idx in enumerate(ranked):
                # print (relpron_labels[relpron_file_name])
                if prop_idx in self.relpron_labels[relpron_file_name][term]:
                    # print (len(relpron_props))
                    term2props[term].append((self.relpron_full_props[relpron_file_name][prop_idx], prop_rank + 1, True))
                    correct_at_k += 1
                    prec = correct_at_k / (prop_rank + 1)
                    ap[term_idx] += prec
                    if any(["_"+term+"_" in prop_pred for prop_pred in self.relpron_full_props[relpron_file_name][prop_idx]]):
                        print ("true confounder:", term, self.relpron_full_props[relpron_file_name][prop_idx])
                else:
                    if prop_rank < 10:
                        term2props[term].append((self.relpron_full_props[relpron_file_name][prop_idx], prop_rank + 1, False))
                    # confounder
                    if any(["_"+term+"_" in prop_pred for prop_pred in self.relpron_full_props[relpron_file_name][prop_idx]]):
                        confounders.append((term, self.relpron_full_props[relpron_file_name][prop_idx], prop_rank + 1, False))
                        confounder_ranks.append(prop_rank + 1)
            ap[term_idx] = ap[term_idx] / correct_at_k
        mean_ap = sum(ap) / len(ap)
        
        return mean_ap, term2props, confounders, confounder_ranks

    def _eval_gs_rho(self, landmark_scores, landmark_truths):
        return spearmanr(landmark_scores, landmark_truths)[0]
    
    def init_relpron(self):
        self.relpron_labels = defaultdict()
        self.relpron_full_props = defaultdict()
        with torch.no_grad():
            for file_idx, (relpron_file_name, dataloader) in tqdm(enumerate(self.eval_relpron_dataloaders.items())):
                # batch_size can only be 1
                for _, data in enumerate(dataloader):
                    # save labels first
                    if relpron_file_name not in self.relpron_labels:
                        self.relpron_labels[relpron_file_name] = data["labels"] # labels
                    if relpron_file_name not in self.relpron_full_props:
                        self.relpron_full_props[relpron_file_name] = data["full_props"] # full_props
    
    def eval_relpron_one_layer(self, batch_relpron_results_dir, epoch, batch_idx):

        # fig_path = os.path.join(
        # product of gaussians
        split2term2props = {}
        term2truth_of_props = {}
        check_max_truth = {}
        results_metric = {}

        relpron_sampled = defaultdict(list)
        relpron_svos = defaultdict()
        relpron_is_sbjs = defaultdict()
        # print (rank, "evaluating")
        with torch.no_grad():
            for file_idx, (relpron_file_name, dataloader) in tqdm(enumerate(self.eval_relpron_dataloaders.items())):
                term2truth_of_props[relpron_file_name] = defaultdict(list)
                
                results_metric[relpron_file_name] = []
                check_max_truth[relpron_file_name] = []
                # batch_size can only be 1
                for _, data in enumerate(dataloader):
                    # save labels first
                    # if relpron_file_name not in self.relpron_labels:
                    #     self.relpron_labels[relpron_file_name] = data["labels"] # labels
                    # if relpron_file_name not in self.relpron_full_props:
                    #     self.relpron_full_props[relpron_file_name] = data["full_props"] # full_props

                    # data is one term-prop pair
                    encoder_data = [tsr.to(self.device) for tsr in data["encoder"]]
                    decoder_data = [tsr.to(self.device) for tsr in data["decoder"]]
                    term = data["term"]
                    
                    mu_batch, log_sigma2_batch = self.autoencoder.encoder(*encoder_data, device = self.device, train_mode = 0)

                    sigma2_batch = torch.exp(log_sigma2_batch)
                    # compute targ(z) with probit approx.
                    targ_log_truth, *_ = self.autoencoder.decoder(
                        mu_batch, sigma2_batch,
                        decoder_data[0].unsqueeze(dim = 0), decoder_data[1].unsqueeze(dim = 0),
                        samp_neg = False, variational = self.autoencoder.variational, device = self.device
                    )

                    term2truth_of_props[relpron_file_name][term].append(targ_log_truth.item())
                    term2truth_of_props[relpron_file_name][term]
                    # check_max_truth[relpron_file_name].append(torch.argmax(all_terms_truth).item() == decoder_data[0].item())
            
                # compute MAP and save results
                mean_ap, term2props, confounders, confounder_ranks = self._eval_relpron_map(term2truth_of_props[relpron_file_name], relpron_file_name)
                mean_confounder_rank = sum(confounder_ranks)/len(confounder_ranks)
                split2term2props[relpron_file_name] = term2props
                # metrics
                results_metric[relpron_file_name].append({
                    "map": mean_ap,
                    "mean_confounder_rank": mean_confounder_rank
                    # "percent_term_max_truth": sum(check_max_truth[relpron_file_name]) / len(check_max_truth[relpron_file_name])
                })

        if self.write_file:
            with open(os.path.join(batch_relpron_results_dir, "split2term2props.json"), "w") as f:
                json.dump(split2term2props, f, indent = 4)
            with open(os.path.join(batch_relpron_results_dir, "term2truth_of_props.json"), "w") as f:
                json.dump(term2truth_of_props, f, indent = 4)
            with open(os.path.join(batch_relpron_results_dir, "confounders.json"), "w") as f:
                json.dump(confounders, f, indent = 4)

        # # figures
        # fig, axs = plt.subplots(len(results), 1, figsize=(5, 5 * len(results)), sharex=True, sharey=True, tight_layout=True)
        # for file_idx, (hyp_file_name, results) in enumerate(results.items()):
        #     axs[file_idx].set_title(hyp_file_name)
        #     axs[file_idx].hist(results, bins = 50)
        # fig.savefig(os.path.join(batch_hyp_results_dir, "hyp_histograms.png"))
                
        return results_metric, term2truth_of_props


    def eval_gs_one_layer(self, batch_gs_results_dir, use_arg1_arg2, epoch, batch_idx, gs_year = "gs2011"):
        # https://osf.io/hby4e/wiki/home/
        # http://compling.eecs.qmul.ac.uk/wp-content/uploads/2015/07/GS2011data.txt
        results_metric = {}
        with torch.no_grad():
            if gs_year == "gs2011":
                eval_gs_dataloaders = self.eval_gs2011_dataloaders
            elif gs_year == "gs2012":
                eval_gs_dataloaders = self.eval_gs2012_dataloaders
            elif gs_year == "gs2013":
                eval_gs_dataloaders = self.eval_gs2013_dataloaders
            elif gs_year == "ks2013":
                eval_gs_dataloaders = self.eval_ks2013_dataloaders

            for file_idx, (gs_file_name, dataloader) in tqdm(enumerate(eval_gs_dataloaders.items())):
                results_metric[gs_file_name] = []
                landmark_truths_sep = []
                landmark_truths_avg = []
                landmark_score_sep = []
                landmark_score_avg = []
                svo_landmark2results = defaultdict(defaultdict)
                for svo_ix, data in enumerate(dataloader):
                    encoder_data = [tsr.to(self.device) for tsr in data["encoder"]]
                    
                    if self.config['encoder_arch']['type'] == "BSGEncoder":
                        mu_batch, log_sigma2_batch = self.autoencoder.encoder(
                            encoder_data[0].unsqueeze(dim = 0),
                            encoder_data[1].unsqueeze(dim = 0),
                            *encoder_data[2:]
                        )
                    elif self.config['encoder_arch']['type'] in ["MyEncoder", "PASEncoder"]:
                        mu_batch, log_sigma2_batch = self.autoencoder.encoder(*encoder_data, device = self.device, train_mode = 0)
                    
                    # data["decoder"]
                    sigma2_batch = torch.exp(log_sigma2_batch)

                    if True: # self.config['eval_gs2011_dataloader']['type'] == "EvalGS2011DataLoader":
                        decoder_data = data["decoder"]
                        landmark2logic_expr, args_vars_batch = decoder_data
                        args_vars_batch = args_vars_batch.to(self.device).unsqueeze(dim = 0)
                        landmark2scores, ix2svo = data['eval']
                        for svo_landmark_idx, (landmark, logic_expr) in enumerate(landmark2logic_expr.items()):
                            logic_expr = torch.tensor(logic_expr, dtype = torch.int32, device = self.device)
                            logic_expr_batch = logic_expr.unsqueeze(dim = 0)
                            
                            targ_log_truth, *_ = self.autoencoder.decoder(mu_batch, sigma2_batch, logic_expr_batch, args_vars_batch, samp_neg = False, variational = self.autoencoder.variational, return_sum = use_arg1_arg2, device = self.device)
                            
                            if isinstance(use_arg1_arg2, float):
                                targ_log_truth_012, *_ = self.autoencoder.decoder(mu_batch, sigma2_batch, logic_expr_batch, args_vars_batch, samp_neg = False, variational = self.autoencoder.variational, return_sum = True, device = self.device)
                                targ_log_truth_0, *_ = self.autoencoder.decoder(mu_batch, sigma2_batch, logic_expr_batch, args_vars_batch, samp_neg = False, variational = self.autoencoder.variational, return_sum = False, device = self.device)
                                targ_log_truth_0 = targ_log_truth_0[0][0]
                                targ_log_truth = targ_log_truth_0 + use_arg1_arg2 * (targ_log_truth_012 - targ_log_truth_0)
                            if use_arg1_arg2 == "only":
                                targ_log_truth_012, *_ = self.autoencoder.decoder(mu_batch, sigma2_batch, logic_expr_batch, args_vars_batch, samp_neg = False, variational = self.autoencoder.variational, return_sum = True, device = self.device)
                                targ_log_truth_0, *_ = self.autoencoder.decoder(mu_batch, sigma2_batch, logic_expr_batch, args_vars_batch, samp_neg = False, variational = self.autoencoder.variational, return_sum = False, device = self.device)
                                targ_log_truth_0 = targ_log_truth_0[0][0]
                                targ_log_truth = targ_log_truth_012 - targ_log_truth_0

                            elif not use_arg1_arg2:
                                targ_log_truth = targ_log_truth[0][0]

                            landmark_truth = targ_log_truth # torch.exp(targ_log_truth)
                            # print (landmark_truth, not use_arg1_arg2, use_arg1_arg2)
                            
                            avg_score = sum(landmark2scores[landmark])/len(landmark2scores[landmark])
                            svo_landmark_key = "-".join([*ix2svo[svo_ix], landmark])
                            svo_landmark2results[svo_landmark_key]['anno_sep'] = landmark2scores[landmark]
                            svo_landmark2results[svo_landmark_key]['anno_avg'] = avg_score
                            svo_landmark2results[svo_landmark_key]['truth'] = landmark_truth.item()

                            landmark_score_sep.extend(landmark2scores[landmark])
                            landmark_truths_sep.extend([landmark_truth.item()] * len(landmark2scores[landmark]))
                            landmark_score_avg.append(avg_score)
                            landmark_truths_avg.append(landmark_truth.item())

                rho_avg = self._eval_gs_rho(landmark_score_avg, landmark_truths_avg)
                rho_sep = self._eval_gs_rho(landmark_score_sep, landmark_truths_sep)
                results_metric[gs_file_name].append({
                    "rho_sep": rho_sep,
                    "rho_avg": rho_avg
                })
                if self.write_file:
                    with open(os.path.join(batch_gs_results_dir, "{}_svo_landmark2results_arg12{}.json".format(gs_year, use_arg1_arg2)), "w") as f:
                        json.dump(svo_landmark2results, f, indent = 4)
                
                # figures
                fig, axs = plt.subplots(2, 1, figsize=(5, 5 * 2), sharex=True, sharey=True, tight_layout=True)
                axs[0].set_title('separate scores')
                axs[0].scatter(landmark_score_sep, landmark_truths_sep, alpha = 0.6)
                axs[1].set_title('average scores')
                axs[1].scatter(landmark_score_avg, landmark_truths_avg, alpha = 0.6)
                fig.savefig(os.path.join(batch_gs_results_dir, "{}_truth_vs_score_arg12{}.png".format(gs_year, use_arg1_arg2)))

        return results_metric, svo_landmark2results



    def eval_homo_poly(self):
        # homonym and polysemy
        # https://aclanthology.org/2021.acl-long.281.pdf
        pass

    def eval_(self):
        pass
        # https://aclanthology.org/D15-1003/


    def eval(self, epoch, batch_idx, len_epoch):
        
        batch_results_dir = os.path.join(self.results_dir , "epoch" + str(epoch) + "_" + str(int(batch_idx * 100 / len_epoch)))
        os.makedirs(batch_results_dir, exist_ok = True)

        results_metrics = {}
        results = {}

        t0 = time.time()
        print ("Evaluation started")

        relpron_results_metrics, gs2011_results_metrics = None, None

        if self.eval_relpron_dataloaders != None:
            self.init_relpron()
            relpron_results_metrics, term2truth_of_props = self.eval_relpron_one_layer(batch_results_dir, epoch, batch_idx)
            print ("RELPRON evaluated")
            results_metrics['relpron'] = relpron_results_metrics
            results['relpron'] = term2truth_of_props
        use_arg1_arg2s = {**{"": True, "_arg0": False, "_arg12": "only"}}#, **{str(i): i for i in [0.05, 0.1, 0.2, 0.3]}}
        if self.eval_gs2011_dataloaders != None:
            for use_arg1_arg2_str, use_arg1_arg2 in use_arg1_arg2s.items():
                gs2011_results_metrics, gs2011_svo_landmark2results = self.eval_gs_one_layer(batch_results_dir, use_arg1_arg2, epoch, batch_idx, "gs2011")
                results_metrics['gs2011{}'.format(use_arg1_arg2_str)] = gs2011_results_metrics
                results['gs2011{}'.format(use_arg1_arg2_str)] = gs2011_svo_landmark2results
            print ("GS2011 evaluated")
        if self.eval_gs2013_dataloaders != None:
            for use_arg1_arg2_str, use_arg1_arg2 in use_arg1_arg2s.items():
                gs2013_results_metrics, gs2013_svo_landmark2results = self.eval_gs_one_layer(batch_results_dir, use_arg1_arg2, epoch, batch_idx, "gs2013")
                results_metrics['gs2013{}'.format(use_arg1_arg2_str)] = gs2013_results_metrics
                results['gs2013{}'.format(use_arg1_arg2_str)] = gs2013_svo_landmark2results
            print ("GS2013 evaluated")
        if self.eval_gs2012_dataloaders != None:
            for use_arg1_arg2_str, use_arg1_arg2 in use_arg1_arg2s.items():
                gs2012_results_metrics, gs2012_svo_landmark2results = self.eval_gs_one_layer(batch_results_dir, use_arg1_arg2, epoch, batch_idx, "gs2012")
                results_metrics['gs2012{}'.format(use_arg1_arg2_str)] = gs2012_results_metrics
                results['gs2012{}'.format(use_arg1_arg2_str)] = gs2012_svo_landmark2results
            print ("GS2012 evaluated")

        t1 = time.time()
        print ("Evaluation finished in {}s".format(t1 - t0))

        results_path = os.path.join(batch_results_dir, "metrics")
        if epoch >= 0:
            if self.write_file:
                with open(results_path, "w") as f:
                    json.dump(results_metrics, f, indent = 4)
        # with open(results_path, "r") as f:
        #     results_metrics_load = json.load(f)

        return results_metrics, results
    