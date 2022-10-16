import os
import argparse
from collections import defaultdict
import json


def gen_configs(sample_or_dummy = False):

    suffix2configs = {}
    transform_config_file_path = "transform/configs/config.json"

    min_pred_or_content_func_freqs = [500, 1000] # 0 for sample config
    num_negative_samples = [8, 32]
    input_dims = [64, 100]
    sample_only = False
    sample_or_dummy_str = ""

    if sample_or_dummy in ["sample", False]:
        data_dir = "data/27092022"
        data_info_dir = "data/27092022/info"
    elif sample_or_dummy == "dummy":
        data_dir = "data/dummy_data"
        data_info_dir = "data/dummy_data/info"
    if sample_or_dummy in ["dummy", "sample"]:
        min_pred_or_content_func_freqs = [0] # 0 for sample config
        num_negative_samples = [8]
        input_dims = [64]
        sample_or_dummy_str = "_" + sample_or_dummy
    if sample_or_dummy == "sample":
        sample_only = True

    for freq in min_pred_or_content_func_freqs:
        for num_neg in num_negative_samples:
            for dim in input_dims:
                config_suffix = "_" + "-".join(["f" + str(freq), "n" + str(num_neg), "d" + str(dim)]) + sample_or_dummy_str
                transformed_dir_suffix = "_" + "f" + str(freq) + sample_or_dummy_str
                if sample_or_dummy == "dummy":
                    config_suffix = sample_or_dummy_str
                    transformed_dir_suffix = sample_or_dummy_str
                name = "TCS" + config_suffix
                transformed_dir_name = "TCS" + transformed_dir_suffix
                transformed_dir = os.path.join(data_dir, "transformed", transformed_dir_name)
                hyp_data_dir = os.path.join("eval_data", transformed_dir_name, "hyp")
                config =  {
                    "name": name,
                    "transformed_dir_name": transformed_dir_name,
                    "sample_only": sample_only,
                    "n_gpu": 4,
                    "ddp": True,
                    "encoder_arch": {
                        "type": "BSGEncoder",
                        "args": {
                            "hidden_act_func": "Tanh",
                            "input_dim": 128,
                            "hidden_dim": 64,
                            "mu_dim": dim,
                            "sigma_dim": 1
                        }
                    },
                    "decoder_arch": {
                        "type": "FLDecoder",
                        "args": {
                            "t_norm": "product",
                            "s_norm": "max",
                            "take_log": False,
                            "freq_sampling": True,
                            "num_negative_samples": num_neg,
                            "neg_cost": 1.0
                        }
                    },
                    "autoencoder_arch": {
                        "type": "VarAutoencoder",
                        "args": {
                            "start_beta": 0.001,
                            "end_beta": 1.0,
                        }
                    },
                    "one_place_sem_func": {
                        "type": "OnePlaceSemFunc",
                        "args": {
                            "input_dim": dim,
                            "hidden_dim": int(dim/2),
                            "hidden_act_func": "LeakyReLU",
                            "output_act_func": "Sigmoid"
                        }
                    },
                    "two_place_sem_func": {
                        "type": "TwoPlaceSemFunc",
                        "args": {
                            "input_dim": dim * 2,
                            "hidden_dim": dim,
                            "hidden_act_func": "LeakyReLU",
                            "output_act_func": "Sigmoid"
                        }
                    },
                    "data_loader": {
                        "type": "DmrsDataLoader",
                        "args":{
                            "data_dir": data_dir,
                            "transformed_dir": transformed_dir,
                            "transform_config_file_path": transform_config_file_path,
                            "batch_size": 1,
                            "shuffle": True,
                            "num_workers": 2,
                            "use_custom_collate_fn": True,
                            "pin_memory": True,
                            "min_pred_func_freq": freq,
                            "min_content_pred_freq": freq,
                            "filter_min_freq": True
                        }
                    },
                    "sem_func_optimizer": {
                        "type": "Adam",
                        "args":{
                            "lr": 0.01,
                            "weight_decay": 0,
                            "amsgrad": True
                        }
                    },
                    "encoder_optimizer": {
                        "type": "Adam",
                        "args":{
                            "lr": 0.001,
                            "weight_decay": 0,
                            "amsgrad": True
                        }
                    },
                    "sem_func_DDPoptimizer": {
                        "type": "ZeroRedundancyOptimizer",
                        "args":{
                            "optimizer_class": "Adam",
                            "lr": 0.01,
                            "weight_decay": 0,
                            "amsgrad": True
                        }
                    },
                    "encoder_DDPoptimizer": {
                        "type": "ZeroRedundancyOptimizer",
                        "args":{
                            "optimizer_class": "Adam",
                            "lr": 0.001,
                            "weight_decay": 0,
                            "amsgrad": True
                        }
                    },
                    "loss": "nll_loss",
                    "metrics": [
                        "training_loss",
                        "log_fuzzy_truthness",
                        "kl_div"
                    ],
                    "lr_scheduler": {
                        "type": "StepLR",
                        "args": {
                            "step_size": 1,
                            "gamma": 0.2
                        }
                    },
                    "trainer": {
                        "epochs": 1,
                        "grad_accum_step": 256,

                        "save_dir": "saved/",
                        "save_period": 1,
                        "verbosity": 2,
                        
                        "monitor": "min training_loss",
                        "early_stop": 10,

                        "tensorboard": True
                    },
                    "eval_hyp_dataloader": {
                        "type": "EvalHypDataLoader",
                        "args":{
                            "hyp_data_dir": hyp_data_dir,
                            "batch_size": 100,
                            "shuffle": False,
                            "num_workers": 1,
                            "pin_memory": True,
                        }
                    },
                    "evaluator": {  
                        "results_dir": "results/",
                        "truth_thresold": 0.8
                    },
                }
                suffix2configs[config_suffix] = config
    return suffix2configs

def write_configs_json(configs_dir, suffix2configs):
    for suffix, config in suffix2configs.items():
        config_path = os.path.join(configs_dir, "config" + suffix) + ".json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent = 4)

def main(configs_dir):

    os.makedirs(configs_dir, exist_ok = True)

    print ("Generating sample config ...")
    suffix2configs = gen_configs(sample_or_dummy = "sample")
    write_configs_json(configs_dir, suffix2configs)

    print ("Generating dummy config ...")
    suffix2configs = gen_configs(sample_or_dummy = "dummy")
    write_configs_json(configs_dir, suffix2configs)

    print ("Generating training configs ...")
    suffix2configs = gen_configs(sample_or_dummy = False)
    write_configs_json(configs_dir, suffix2configs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate_config')
    parser.add_argument('-c', '--configs_dir', default=None, type=str,
                      help='dir to save the config files (default: None)')
    args = parser.parse_args()
    # custom cli options to modify configuration from default values given in json file.
    main(
        args.configs_dir
    )