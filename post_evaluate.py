import argparse
import collections
import json
import torch

import numpy as np
from parse_config import ConfigParser
from trainer import Trainer
# from trainer.trainer_thread import Trainer
from utils import prepare_device, get_transformed_info
from train import get_trainer_args
import baselines

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

from pprint import pprint
import os
from collections import defaultdict
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import bootstrap, spearmanr

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_dtype(torch.float64)
np.random.seed(SEED)

# def get_trainer_args(config):

#     # declare decoders
#     MIN_PRED_FUNC_FREQ = config["data_loader"]["args"]["min_pred_func_freq"]
#     MIN_CONTENT_PRED_FREQ = config["data_loader"]["args"]["min_content_pred_freq"]

#     log_dir = os.path.join("saved/log", config["name"])
#     run_dir = os.path.join("saved/run", config["name"])

#     transformed_dir  = config["data_loader"]["args"]["transformed_dir"]
#     transformed_info_dir = os.path.join(transformed_dir, "info")

#     pred_func2cnt, content_pred2cnt, pred2ix, content_predarg2ix, pred_func2ix = get_transformed_info(transformed_info_dir)

#     sorted_pred_func2ix = sorted(pred_func2ix.items(), key = lambda x: x[1])
#     pred_funcs = []
#     for pred_func, ix in sorted_pred_func2ix:
#         pred_funcs.append(pred_func)

#     print ("Initializing encoder ...")
#     # content_preds = set(content_pred2cnt)
#     if config['encoder_arch']['type'] == 'MyEncoder':
#         num_embs = len(pred2ix)
#     elif config['encoder_arch']['type'] == 'PASEncoder':
#         num_embs = len(content_predarg2ix)

#     encoder = config.init_obj('encoder_arch', module_arch, num_embs = num_embs)
#     for p in encoder.parameters():
#         p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
#     # logger.info(decoder)

#     print ("Initializing decoder ...")
#     if not config['decoder_arch']['args']['use_truth']:
#         decoder = config.init_obj('decoder_arch', module_arch, num_sem_funcs = len(pred_funcs), train = False
#             # pred2cnt = pred2cnt, pred_ix2arg_num2pred_func_ix = pred_ix2arg_num2pred_func_ix, arg_num_sum2preds_ix = arg_num_sum2preds_ix, pred_func_ix2arg_num = pred_func_ix2arg_num
#         )
#     else:
#         decoder = config.init_obj('decoder_arch', module_arch, num_sem_funcs = len(pred_funcs), pred_func2cnt = pred_func2cnt, pred_funcs = pred_funcs) 
    

    

#     # get function handles of loss and metrics
#     # criterion = getattr(module_loss, config['loss'])
#     criterion = None
#     metric_ftns = [getattr(module_metric, met) for met in config['metrics']]

#     trainer_args = {
#         "encoder": encoder,
#         "pred2ix": pred2ix,
#         "pred_func2ix": pred_func2ix,
#         "decoder": decoder,
#         "criterion": criterion,
#         "metric_ftns": metric_ftns,
#         "config": config
#     }

#     return trainer_args

def ensemble(checkpoints_results, evaluator):

    ds2split2metrics = defaultdict(lambda: defaultdict(list))
    ds2split2results = defaultdict(defaultdict)
    ds2split2outputs = defaultdict(lambda: defaultdict(defaultdict))
    ds2split2labels = defaultdict(lambda: defaultdict(defaultdict))
    relpron_split2term2props = defaultdict()
    for result in checkpoints_results:
        for dataset, dataset_result in result.items():
            if dataset == 'relpron':
                evaluator.init_relpron()
                split2term2prop_truths_list = dataset_result
                for split_name, term2prop_truths_list in split2term2prop_truths_list.items():
                    if split_name not in ds2split2results['relpron']:
                        ds2split2results['relpron'][split_name] = term2prop_truths_list
                    else:
                        for term, prop_truths_list in term2prop_truths_list.items():
                            ds2split2results['relpron'][split_name][term] = [x + y for x, y in zip(ds2split2results['relpron'][split_name][term], prop_truths_list)]
                
            elif dataset in ['gs2011', 'gs2013', 'gs2012', 'gs2011_arg0', 'gs2013_arg0', 'gs2012_arg0']:
                svo_landmark2results = dataset_result
                split_name = dataset.split("_")[0] + "_f0"
                # ds2split2results[dataset][split_name] = defaultdict(defaultdict)
                if split_name not in ds2split2results[dataset]:
                    ds2split2results[dataset][split_name] = svo_landmark2results
                else:
                    for svo_landmark_key, scores in svo_landmark2results.items():
                        ds2split2results[dataset][split_name][svo_landmark_key]['truth'] += scores['truth']
                
    
    for dataset, split2results in ds2split2results.items():
        for split_name, results in split2results.items():
            if dataset == 'relpron':
                mean_ap, term2props, relpron_confounders, relpron_confounder_ranks  = evaluator._eval_relpron_map(results, split_name)
                mean_relpron_confounder_rank = sum(relpron_confounder_ranks)/len(relpron_confounder_ranks)
                ds2split2metrics[dataset][split_name].append({
                    "map": mean_ap,
                    "mean_confounder_rank": mean_relpron_confounder_rank
                })
                # ds2split2outputs[dataset][split_name] ?
                # ds2split2labels ?
                relpron_split2term2props[split_name] = term2props
            elif dataset in ['gs2011', 'gs2013', 'gs2012', 'gs2011_arg0', 'gs2013_arg0', 'gs2012_arg0']:
                landmark_score_sep = []
                landmark_score_avg = []
                landmark_truths_sep = []
                landmark_truths_avg = []
                sorted_svo_landmark_keys = sorted(results.keys())
                for svo_landmark_key in sorted_svo_landmark_keys:
                    scores = results[svo_landmark_key]
                    landmark_score_sep.extend(scores['anno_sep'])
                    landmark_truths_sep.extend([scores['truth']] * len(scores['anno_sep']))
                    landmark_score_avg.append(scores['anno_avg'])
                    landmark_truths_avg.append(scores['truth'])
                rho_avg = evaluator._eval_gs_rho(landmark_score_avg, landmark_truths_avg)
                rho_sep = evaluator._eval_gs_rho(landmark_score_sep, landmark_truths_sep)
                ds2split2metrics[dataset][split_name].append({
                    "rho_sep": rho_sep,
                    "rho_avg": rho_avg
                })
                ds2split2outputs[dataset][split_name]["landmark_truths_sep"] = landmark_truths_sep
                ds2split2outputs[dataset][split_name]["landmark_truths_avg"] = landmark_truths_avg
                ds2split2labels[dataset][split_name]["landmark_score_sep"] = landmark_score_sep
                ds2split2labels[dataset][split_name]["landmark_score_avg"] = landmark_score_avg

    return {
        "ds2split2metrics": ds2split2metrics,
        "ds2split2outputs": ds2split2outputs,
        "ds2split2labels": ds2split2labels,
        "relpron_split2term2props": relpron_split2term2props,
        "relpron_confounders": relpron_confounders,
    }

def organize_baseline_results(config, results_baseline):
    transformed_dir_name = config._config['transformed_dir_name']
    eval_data_dir = "eval_data"
    fk = "f0" # filter freq < k
    # baseline_model_name = 'bert-base-uncased'

    ds2outputs_baseline = defaultdict()
    for model_name in results_baseline["model2ds2outputs"]:
        for dataset, output in results_baseline["model2ds2outputs"][model_name].items():
            if dataset in ["gs2011", "gs2013", "gs2012"]:
                svol2scores = output["svol2scores"]
                svol2cos_sim = output["svol2cos_sim"]
                if dataset in ["gs2011", "gs2013"]:
                    preds2svol_file = "{}_{}_preds2svol.json".format(dataset, fk)
                elif dataset in ["gs2012"]:
                    preds2svol_file = "{}_{}_preds2asvaol.json".format(dataset, fk)
                with open(os.path.join(eval_data_dir, transformed_dir_name, dataset, "info", preds2svol_file)) as f:
                    preds2svol = json.load(f)

                svol2preds = {tuple(v): k for k, v in preds2svol.items()}
                sorted_svol_preds =  sorted(preds2svol.keys())

                landmark_score_sep = []
                landmark_score_avg = []
                landmark_cos_sims_sep = []
                landmark_cos_sims_avg = []
                for svol_preds in sorted_svol_preds:
                    svol = tuple(preds2svol[svol_preds])
                    avg_score = sum(svol2scores[svol])/len(svol2scores[svol])
                    landmark_score_avg.append(avg_score)
                    landmark_cos_sims_avg.append(svol2cos_sim[svol])
                    for score in svol2scores[svol]:
                        landmark_score_sep.append(score)
                        landmark_cos_sims_sep.append(svol2cos_sim[svol])
                ds2outputs_baseline[dataset] = {
                    "landmark_cos_sims_sep": landmark_cos_sims_sep,
                    "landmark_cos_sims_avg": landmark_cos_sims_avg,
                    "landmark_score_sep": landmark_score_sep,
                    "landmark_score_avg": landmark_score_avg
                }

    return ds2outputs_baseline

def bootstrap_test(ds2split2outputs_sys, ds2outputs_baseline, ds2split2labels):

    def bs_statistic(sampled_outputs1, sampled_outputs2, sampled_labels):
        rho1 = spearmanr(sampled_outputs1, sampled_labels)[0] 
        rho2 = spearmanr(sampled_outputs2, sampled_labels)[0]
        return rho2 - rho1
    
    rng = np.random.default_rng()
    sep_avg_str = ["sep", "avg"]
    for dataset, outputs_baseline in ds2outputs_baseline.items():
        if dataset in ['gs2011', 'gs2013', 'gs2012']:
            for sep_or_avg in sep_avg_str:
                outputs_baseline_sep_or_avg = outputs_baseline['landmark_cos_sims_{}'.format(sep_or_avg)]
                outputs_sys = ds2split2outputs_sys['{}_arg0'.format(dataset)]['{}_f0'.format(dataset)]['landmark_truths_{}'.format(sep_or_avg)]
                labels = ds2split2labels['{}_arg0'.format(dataset)]['{}_f0'.format(dataset)]['landmark_score_{}'.format(sep_or_avg)]
                
                assert labels == outputs_baseline['landmark_score_{}'.format(sep_or_avg)]

                res = bootstrap((outputs_baseline_sep_or_avg, outputs_sys, labels), bs_statistic, n_resamples = 9999, paired = True, confidence_level = 0.95,
                        random_state = rng)

                hyp_list = [sampled_stat <= 0 for sampled_stat in res.bootstrap_distribution]
                print ("{} {}: p-value (two-sides)=", dataset, sep_or_avg, sum(hyp_list)/len(hyp_list) * 2)

                fig, ax = plt.subplots()
                ax.hist(res.bootstrap_distribution, bins=25)
                ax.set_title('Bootstrap Distribution')
                ax.set_xlabel('statistic value')
                ax.set_ylabel('frequency')
                plt.show()

def main(config, batch_results_dirs, checkpoints, comp_baseline):

    device = 'cpu'
    world_size = 1
    ddp = False
    checkpoints_results = []

    if batch_results_dirs or checkpoints:
        # seed = int(checkpoint[:-4].rsplit("/", 1)[0].split("seed")[1])
        config._config['seed'] = 0
        trainer_args = get_trainer_args(config, train = False)
        trainer_args['config']["data_loader"]['type'] = None # disable loading training data
        trainer = Trainer(world_size = world_size, device = device, rank = -1, ddp = ddp, **trainer_args)
        # trainer.encoder_opt = config.init_obj('encoder_optimizer', torch.optim, trainer.encoder.parameters())
        # trainer.decoder_opt = config.init_obj('decoder_optimizer', torch.optim, trainer.decoder.parameters())
        trainer.initialize_trainer()
        trainer.evaluator.write_file = False

        results_ens_dir = os.path.join(
            'results_ens',
            config._config['name']
        )
        os.makedirs(results_ens_dir, exist_ok = True)

        if batch_results_dirs != None:
            datasets = ["relpron", "gs2011", "gs2013", "gs2012"]
            use_arg1_arg2s = {"": True, "_arg0": False}
            checkpoints_results = []
            for dir in batch_results_dirs:
                results = {}
                for dataset in datasets:
                    if dataset == "relpron":
                        with open(os.path.join(dir, "term2truth_of_props.json")) as f:
                            relpron_truths = json.load(f)
                            results[dataset] = relpron_truths
                    else:
                        for use_arg1_arg2_str, use_arg1_arg2 in use_arg1_arg2s.items():
                            with open(os.path.join(dir, "{}_svo_landmark2results_arg12{}.json".format(dataset, str(use_arg1_arg2)))) as f:
                                svo_landmark2results = json.load(f)
                                results["{}{}".format(dataset,use_arg1_arg2_str)] = svo_landmark2results
                checkpoints_results.append(results)

            checkpoint_names = [
                "".join([
                    "seed" + checkpoint.rsplit("seed", 1)[1].split("/", 1)[0],
                    "epoch" + checkpoint.rsplit("epoch", 1)[1].split("_", 1)[0],
                    "batch" + checkpoint.rsplit("_", 1)[1]
                ]) for checkpoint in batch_results_dirs
            ]

        elif checkpoints != None:
 
            for checkpoint in checkpoints:
                trainer._resume_checkpoint(checkpoint)
                results_metrics, results = trainer.evaluator.eval(-1, -1, -1)
                pprint (results_metrics)
                checkpoints_results.append(results)

            checkpoint_names = [
                "".join([
                    "seed" + checkpoint.rsplit("seed", 1)[1].split("/", 1)[0],
                    "epoch" + checkpoint.rsplit("epoch", 1)[1].split("-", 1)[0],
                    "batch" + checkpoint.rsplit("batch", 1)[1].split(".", 1)[0]
                ]) for checkpoint in checkpoints
            ]

        # ensemble
        results_sys = ensemble(checkpoints_results, trainer.evaluator)
        print ("Ensembled:")
        pprint (results_sys["ds2split2metrics"])

        this_results_ens_dir = os.path.join(results_ens_dir, "-".join(checkpoint_names))
        os.makedirs(this_results_ens_dir, exist_ok = True)
        
        with open(os.path.join(this_results_ens_dir, "metrics"), "w") as f:
            json.dump(results_sys['ds2split2metrics'], f, indent = 4)
        with open(os.path.join(this_results_ens_dir, "relpron_split2term2props.json"), "w") as f:
            json.dump(results_sys['relpron_split2term2props'], f, indent = 4)
        with open(os.path.join(this_results_ens_dir, "relpron_confounders.json"), "w") as f:
            json.dump(results_sys['relpron_confounders'], f, indent = 4)



    else:
        config_name = config['name']
        config._config['seed'] = None
        trainer_args = get_trainer_args(config, train = False)
        trainer_args['config']["data_loader"]['type'] = None
        trainer = Trainer(world_size = world_size, device = device, rank = -1, ddp = ddp, **trainer_args)
        trainer.encoder_opt = config.init_obj('encoder_optimizer', torch.optim, trainer.encoder.parameters())
        trainer.decoder_opt = config.init_obj('decoder_optimizer', torch.optim, trainer.decoder.parameters())
        for epoch in range(1,5):
            for batch in range(100):
                _pth = "saved/models/{}/checkpoint-epoch{}-batch{}.pth".format(config_name, epoch, batch)
                if os.path.isfile(_pth):
                    trainer._resume_checkpoint(_pth)
                    trainer.initialize_trainer()
                    results_metrics, results = trainer.evaluator.eval(epoch, batch, 100)
                    pprint (results_metrics)

    # elif len(checkpoints) == 1:
    #     checkpoint = checkpoints[0]
    #     seed = int(checkpoint[:-4].rsplit("/", 1)[0].split("seed")[1])
    #     config._config['seed'] = seed
    #     trainer_args = get_trainer_args(config, train = False)
    #     trainer_args['config']["data_loader"]['type'] = None
    #     trainer = Trainer(world_size = world_size, device = device, rank = -1, ddp = ddp, **trainer_args)
    #     trainer.encoder_opt = config.init_obj('encoder_optimizer', torch.optim, trainer.encoder.parameters())
    #     trainer.decoder_opt = config.init_obj('decoder_optimizer', torch.optim, trainer.decoder.parameters())
    #     trainer._resume_checkpoint(checkpoint)
    #     trainer.initialize_trainer()
    #     results_metrics, results = trainer.evaluator.eval(-1, -1, -1)
    #     pprint (results_metrics)


    if comp_baseline:
        results_baseline = baselines.main(comp_baseline)
        ds2outputs_baseline = organize_baseline_results(config, results_baseline)
        ds2split2outputs_sys = results_sys["ds2split2outputs"]
        ds2split2labels  = results_sys["ds2split2labels"]

        pprint (results_baseline['model2ds2metrics'])
        pprint (results_sys['ds2split2metrics'])
        bootstrap_test(ds2split2outputs_sys, ds2outputs_baseline, ds2split2labels)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)', required=True)
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-o','--batch_results_dirs', action='append', default=None, type=str, help='paths to result dirs for ensembling; using previous outputs (default: None)')
    parser.add_argument('-p','--checkpoints', action='append', default=None, type=str, help='paths to checkpoints for ensembling; run new eval (default: None)')
    parser.add_argument('-b','--comp_baseline', default=None, type=str, help='eval_data_sets path for comparison against baseline (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        # CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        # CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(parser, options)

    main(config, args.batch_results_dirs, args.checkpoints, args.comp_baseline)