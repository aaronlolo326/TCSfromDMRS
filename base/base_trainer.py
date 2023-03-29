import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter

import data_loader.data_loaders as module_data
import model.model as module_arch

from evaluator.evaluator import Evaluator

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer

import os
import json
from pprint import pprint
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')
SEED = 29

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, world_size, encoder, decoder, pred2ix, pred_func2ix, criterion, metric_ftns,
        # optimizer,
        config, ddp, device, rank):

        self.config = config
        self.seed = self.config['seed']
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.ddp = ddp
        self.world_size = world_size
        self.rank = rank
        self.device = device

        self.encoder = encoder
        self.decoder = decoder
        self.pred2ix = pred2ix
        self.pred_func2ix = pred_func2ix
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        # self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.grad_accum_step = cfg_trainer['grad_accum_step']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch_ddp(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError


    def initialize_trainer(self):

        print ("Initializing training data loader ...")
        if self.ddp:
            # setup data_loader instances
            num_replicas = self.world_size
        else:
            num_replicas = 0

        if self.config["data_loader"]['type'] != None:
            torch.manual_seed(SEED)
            transformed_dir_suffix = os.path.join("transformed", self.config["name"])
            self.data_loader = self.config.init_obj('data_loader', module_data,
                transformed_dir_suffix = transformed_dir_suffix, pred2ix = self.pred2ix, num_replicas = num_replicas, rank = self.rank, device = self.device)# lex_pred2cnt = lex_pred2cnt, pred_func2cnt = pred_func2cnt)
            print ("# of training instances: {}".format(len(self.data_loader.dataset)))
            self.len_epoch = len(self.data_loader)

        self.valid_data_loader = None
        self.do_validation = self.valid_data_loader is not None

        if self.ddp:
            print ("Sending encoder to DDP device ...")
            if self.device == 'cpu':
                self.encoder = DDP(self.encoder, output_device = None)
            else:
                self.encoder = self.encoder.to(self.rank)
                self.encoder = DDP(self.encoder, device_ids = [self.rank]) # find_unused_parameters = True)
            
            
            print ("Sending deocder DDP device ...")
            if self.device == 'cpu':
                self.decoder = DDP(self.decoder, output_device = None)
            else:
                self.decoder = self.decoder.to(self.rank)
                self.decoder.tsr_to_device(self.rank) # if not use_truth
                self.decoder = DDP(self.decoder, device_ids = [self.rank]) # find_unused_parameters = True)
            # print ("Sending semantic functions to DDP device ...")
            # for sem_func_ix in range(len(self.decoder.sem_funcs)):
            #     if self.device == 'cpu':
            #         # self.decoder.sem_funcs[sem_func_ix] = self.decoder.sem_funcs[sem_func_ix].to(self.device)
            #         self.decoder.sem_funcs[sem_func_ix] = DDP(self.decoder.sem_funcs[sem_func_ix], output_device = None)
            #     else:
            #         self.decoder.sem_funcs[sem_func_ix] = self.decoder.sem_funcs[sem_func_ix].to(self.rank)
            #         self.decoder.sem_funcs[sem_func_ix] = DDP(self.decoder.sem_funcs[sem_func_ix], device_ids=[self.rank]) # find_unused_parameters = True)

        else:
            print ("Sending encoder to device ...")
            self.encoder = self.encoder.to(self.device)
            print ("Sending deocder to device ...")
            self.decoder = self.decoder.to(self.device)
            self.decoder.tsr_to_device(self.device) 
            # print ("Sending semantic functions to device ...")
            # for sem_func_ix in range(len(self.decoder.sem_funcs)):
            #     self.decoder.sem_funcs[sem_func_ix] = self.decoder.sem_funcs[sem_func_ix].to(self.device)

        print ("Initializing autoencoder ...")
        with_logic = self.config['with_logic']
        if self.ddp and self.device != 'cpu':
            self.autoencoder = self.config.init_obj('autoencoder_arch', module_arch, encoder = self.encoder, decoder = self.decoder, with_logic = with_logic, device = self.rank, ddp = self.ddp) 
        else:
            self.autoencoder = self.config.init_obj('autoencoder_arch', module_arch, encoder = self.encoder, decoder = self.decoder,  with_logic = with_logic, device = self.device, ddp = self.ddp)
        ## each sem_func has its own optimizer (This should be faster)
        print ("Initializing optimizer for the encoder and semantic functions ...")

        self.encoder_opt = self.config.init_obj('encoder_optimizer', torch.optim, self.encoder.parameters())
        # sem_funcs_params = [{'params': param} for param in list(self.decoder.parameters())[0]]
        self.decoder_opt = self.config.init_obj('decoder_optimizer', torch.optim, self.decoder.parameters())
        # self.sem_funcs_opt = [self.config.init_obj('sem_func_optimizer', torch.optim, sem_func.parameters()) for sem_func in self.decoder.sem_funcs]
        
        print ("Initializing learning rate scheduler for the optimizer(s) ...")
        self.encoder_lr_scheduler = self.config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.encoder_opt)
        self.decoder_lr_scheduler = self.config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.decoder_opt)
        # self.sem_funcs_lr_scheduler = [self.config.init_obj('lr_scheduler', torch.optim.lr_scheduler, opt) for opt in self.sem_funcs_opt]


        if not self.ddp or self.rank == 0:
            print ("Initializing evaluator ...")
            # if self.ddp:
            #     self.evaluator = Evaluator(dataloaders = [eval_hyp_dataloaders], autoencoder = self.autoencoder, config = self.config, device = self.rank)
            # else:
            print ("Initializing evaluation data loaders ...")

            eval_relpron_dataloaders, eval_gs2011_dataloaders, eval_gs2013_dataloaders, eval_gs2012_dataloaders = None, None, None, None

            if self.config["eval_relpron_dataloader"]['type'] != None:
                relpron_data_dir = os.path.join(self.config["eval_relpron_dataloader"]["args"]["relpron_data_dir"], "data")
                relpron_data_path = [
                    os.path.join(relpron_data_dir, file)
                    for file in os.listdir(relpron_data_dir)
                    if all([
                        os.path.isfile(os.path.join(relpron_data_dir, file)),
                        file.split("_")[1] in self.config["eval_relpron_dataloader"]["args"]["split"]
                    ])
                ]
                relpron_file_names = [file.rsplit(".", 1)[0] for file in os.listdir(relpron_data_dir) if os.path.isfile(os.path.join(relpron_data_dir, file))]
                eval_relpron_dataloaders = {relpron_file_names[idx]: self.config.init_obj(
                    'eval_relpron_dataloader',
                    module_data,
                    relpron_data_path = path, num_replicas = num_replicas, pred_func2ix = self.pred_func2ix, pred2ix = self.pred2ix, encoder_arch_type = self.config['encoder_arch']['type']
                    )
                    for idx, path in enumerate(relpron_data_path)
                }


            if self.config["eval_gs2011_dataloader"]['type'] != None:
                gs2011_data_dir = os.path.join(self.config["eval_gs2011_dataloader"]["args"]["data_dir"], "data")
                gs2011_data_path = [os.path.join(gs2011_data_dir, file) for file in os.listdir(gs2011_data_dir) if os.path.isfile(os.path.join(gs2011_data_dir, file))]
                gs2011_file_names = [file.rsplit(".", 1)[0] for file in os.listdir(gs2011_data_dir) if os.path.isfile(os.path.join(gs2011_data_dir, file))]
                eval_gs2011_dataloaders = {gs2011_file_names[idx]: self.config.init_obj(
                    'eval_gs2011_dataloader',
                    module_data,
                    data_path = path, num_replicas = num_replicas, pred_func2ix = self.pred_func2ix, pred2ix = self.pred2ix, encoder_arch_type = self.config['encoder_arch']['type']
                    )
                    for idx, path in enumerate(gs2011_data_path)
                }

            if self.config["eval_gs2013_dataloader"]['type'] != None:
                gs2013_data_dir = os.path.join(self.config["eval_gs2013_dataloader"]["args"]["data_dir"], "data")
                gs2013_data_path = [os.path.join(gs2013_data_dir, file) for file in os.listdir(gs2013_data_dir) if os.path.isfile(os.path.join(gs2013_data_dir, file))]
                gs2013_file_names = [file.rsplit(".", 1)[0] for file in os.listdir(gs2013_data_dir) if os.path.isfile(os.path.join(gs2013_data_dir, file))]
                eval_gs2013_dataloaders = {gs2013_file_names[idx]: self.config.init_obj(
                    'eval_gs2013_dataloader',
                    module_data,
                    data_path = path, num_replicas = num_replicas, pred_func2ix = self.pred_func2ix, pred2ix = self.pred2ix, encoder_arch_type = self.config['encoder_arch']['type']
                    )
                    for idx, path in enumerate(gs2013_data_path)
                }

            if self.config["eval_gs2012_dataloader"]['type'] != None:
                gs2012_data_dir = os.path.join(self.config["eval_gs2012_dataloader"]["args"]["data_dir"], "data")
                gs2012_data_path = [os.path.join(gs2012_data_dir, file) for file in os.listdir(gs2012_data_dir) if os.path.isfile(os.path.join(gs2012_data_dir, file))]
                gs2012_file_names = [file.rsplit(".", 1)[0] for file in os.listdir(gs2012_data_dir) if os.path.isfile(os.path.join(gs2012_data_dir, file))]
                eval_gs2012_dataloaders = {gs2012_file_names[idx]: self.config.init_obj(
                    'eval_gs2012_dataloader',
                    module_data,
                    data_path = path, num_replicas = num_replicas, pred_func2ix = self.pred_func2ix, pred2ix = self.pred2ix, encoder_arch_type = self.config['encoder_arch']['type']
                    )
                    for idx, path in enumerate(gs2012_data_path)
                }

            if self.config["eval_ks2013_dataloader"]['type'] != None:
                ks2013_data_dir = os.path.join(self.config["eval_ks2013_dataloader"]["args"]["data_dir"], "data")
                ks2013_data_path = [os.path.join(ks2013_data_dir, file) for file in os.listdir(ks2013_data_dir) if os.path.isfile(os.path.join(ks2013_data_dir, file))]
                ks2013_file_names = [file.rsplit(".", 1)[0] for file in os.listdir(ks2013_data_dir) if os.path.isfile(os.path.join(ks2013_data_dir, file))]
                eval_ks2013_dataloaders = {ks2013_file_names[idx]: self.config.init_obj(
                    'eval_ks2013_dataloader',
                    module_data,
                    data_path = path, num_replicas = num_replicas, pred_func2ix = self.pred_func2ix, pred2ix = self.pred2ix, encoder_arch_type = self.config['encoder_arch']['type']
                    )
                    for idx, path in enumerate(ks2013_data_path)
                }


            self.results_dir = os.path.join(
                self.config['evaluator']['results_dir'],
                self.config['name'] + "-seed{}".format(self.seed)
            )
            os.makedirs(self.results_dir, exist_ok = True)
            self.evaluator = Evaluator(results_dir = self.results_dir, dataloaders = [
                    eval_relpron_dataloaders,
                    eval_gs2011_dataloaders,
                    eval_gs2013_dataloaders,
                    eval_gs2012_dataloaders,
                    eval_ks2013_dataloaders
            ], autoencoder = self.autoencoder, config = self.config, device = self.device)


    def train(self):
        """
        Full training logic
        """
        # if self.device !torch.cuda.set_device(self.device)

        self.initialize_trainer()
        print ("Done. Start training {}".format(self.rank))

        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch_ddp(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            best = False
            if self.mnt_mode != 'off':
                results_metrics = None

        
        print ("Done training {}".format(self.rank))

    def _save_checkpoint(self, epoch, batch, save_best = False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        encoder_arch = type(self.encoder).__name__
        if self.ddp:
            state = {
                'encoder_arch': encoder_arch,
                'epoch': epoch,
                'batch': batch,
                'encoder_state_dict': self.encoder.module.state_dict(),
                'decoder_state_dict': self.decoder.module.state_dict(),
                # 'sem_funcs_state_dict': [sem_func.state_dict() for sem_func in self.decoder.sem_funcs],
                # 'optimizer': self.optimizer.state_dict(),
                'encoder_opt': self.encoder_opt.state_dict(),
                'decoder_opt': self.decoder_opt.state_dict(),
                # 'sem_funcs_opt': [opt.state_dict() for opt in self.sem_funcs_opt],
                'monitor_best': self.mnt_best,
                'config': self.config
            }
        else:
            state = {
                'encoder_arch': encoder_arch,
                'epoch': epoch,
                'batch': batch,
                'encoder_state_dict': self.encoder.state_dict(),
                'decoder_state_dict': self.decoder.state_dict(),
                # 'sem_funcs_state_dict': [sem_func.state_dict() for sem_func in self.decoder.sem_funcs],
                # 'optimizer': self.optimizer.state_dict(),
                'encoder_opt': self.encoder_opt.state_dict(),
                'decoder_opt': self.decoder_opt.state_dict(),
                # 'sem_funcs_opt': [opt.state_dict() for opt in self.sem_funcs_opt],
                'monitor_best': self.mnt_best,
                'config': self.config
            }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}-batch{}.pth'.format(epoch, batch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        # if save_best:
        #     best_path = str(self.checkpoint_dir / 'model_best.pth')
        #     torch.save(state, best_path)
        #     self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch']
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['encoder_arch'] != self.config['encoder_arch']:
            self.logger.warning("Warning: Encoder architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        if checkpoint['config']['decoder_arch'] != self.config['decoder_arch']:
            self.logger.warning("Warning: Decoder architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        # for sem_func_ix in range(len(self.decoder.sem_funcs)):
        #     self.decoder.sem_func[sem_func_ix].load_state_dict(checkpoint['decoders_state_dict'][sem_func_ix])

        self.initialize_trainer()
        # load optimizer state from checkpoint only when optimizer type is not changed.
        if any([
            checkpoint['config']['encoder_optimizer']['type'] != self.config['encoder_optimizer']['type'],
            checkpoint['config']['decoder_optimizer']['type'] != self.config['decoder_optimizer']['type'],
            # checkpoint['config']['sem_func_optimizer']['type'] != self.config['sem_func_optimizer']['type']
        ]):
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.encoder_opt.load_state_dict(checkpoint['encoder_opt'])
            self.decoder_opt.load_state_dict(checkpoint['decoder_opt'])
            # for sem_funcs_ix in range(len(self.sem_funcs_opt)):
            #     self.sem_funcs_opt[sem_funcs_ix].load_state_dict(checkpoint['sem_funcs_opt'][sem_funcs_ix])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
        