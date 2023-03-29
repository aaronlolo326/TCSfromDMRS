from filecmp import dircmp
from tkinter import Y
from xml.dom.minidom import parseString
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from collections import defaultdict
from pprint import pprint

import numpy as np
import math

from functools import reduce


# x = F.dropout(x, training=self.training)
# return F.log_softmax(x, dim=1)
class VarAutoencoder(BaseModel):

    def __init__(self, encoder, decoder, start_beta, end_beta, end_beta_epoch, with_logic, variational, std_norm, l2_reg_coeff, device, ddp):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.end_beta_epoch = end_beta_epoch
        self.with_logic = with_logic
        self.variational = variational
        self.std_norm = std_norm
        self.l2_reg_coeff = l2_reg_coeff
        self.device = device
        if ddp:
            self.mu_dim = self.encoder.module.mu_dim
        else:
            self.mu_dim = self.encoder.mu_dim
        self.normal_dist = torch.distributions.Normal(torch.zeros(self.mu_dim, device = self.device), torch.tensor(1.0, device = self.device))

    def sample_from_gauss(self, mu_batch, sigma2_batch, num_samples = 1):
        batch_size, max_num_nodes, mu_dim = mu_batch.size()
        num_nodes_batch = max_num_nodes # num_targ_preds_batch
        # currently supprot batch_size = 1 only, i.e. max_num_nodes = num_nodes
        if num_samples == 1:
            sample_eps = self.normal_dist.sample(torch.Size([batch_size, num_nodes_batch]))
        else:
            sample_eps = self.normal_dist.sample(torch.Size([batch_size, num_nodes_batch, num_samples]))
            mu_batch = mu_batch.unsqueeze(dim = 2)
            sigma2_batch = sigma2_batch.unsqueeze(dim = 2)

        sample_zs = mu_batch + torch.sqrt(sigma2_batch) * sample_eps

        return sample_zs


    def run(self, **kwargs):

        encoder_data = kwargs["encoder"]
        decoder_data = kwargs["decoder"]

        mu_batch, log_sigma2_batch = self.encoder(*encoder_data, device = self.device)
        # print ("mu_batch", mu_batch)
        # print ("log_sigma2_batch", log_sigma2_batch)
        if log_sigma2_batch != None:
            log_sigma2_batch = log_sigma2_batch.to(self.device)
            sigma2_batch = torch.exp(log_sigma2_batch)
            if self.with_logic: # FLDecoder
                sample_zs = self.sample_from_gauss(mu_batch, sigma2_batch, num_samples = 1)#.squeeze(dim = 2)
                log_truth_batch = self.decoder.decode_batch(sample_zs, *decoder_data, device = self.device)#.to(self.device)
            else: # OneLayerSemmFuncsDecoder
                log_truth_batch, pos_sum, neg_sum = self.decoder(mu_batch, sigma2_batch, *decoder_data, samp_neg = True, variational = self.variational, device = self.device)#.to(self.device)
        else: # with logic
            sample_zs = mu_batch
            log_truth_batch = self.decoder.decode_batch(sample_zs, *decoder_data, device = self.device)#.to(self.device)
            
        # standard normal prior
        batch_size, max_num_nodes, mu_dim = mu_batch.size()
        num_nodes_batch = max_num_nodes # num_targ_preds_batch
        if self.variational:
            if self.std_norm:
                kl_div = (1/2) * (torch.sum(sigma2_batch, dim = (1,2)) * mu_dim + torch.sum(torch.square(mu_batch), dim = (1,2)) - num_nodes_batch * mu_dim - torch.sum(log_sigma2_batch, dim = (1,2)) * mu_dim)
            else:
                kl_div = (1/2) * (torch.sum(sigma2_batch, dim = (1,2)) * mu_dim - num_nodes_batch * mu_dim - torch.sum(log_sigma2_batch, dim = (1,2)) * mu_dim)
                # kl_div = (1/2) * (- torch.sum(log_sigma2_batch, dim = (1,2)) * mu_dim)
            kl_div = kl_div.squeeze()
        else:
            kl_div = torch.tensor([0.0], device = self.device, requires_grad=True)
        
        l2_norm_reg = 0 # self.l2_reg_coeff * torch.sum(torch.square(self.encoder.module.fc_sigma.weight))

        return log_truth_batch, kl_div, l2_norm_reg, pos_sum, neg_sum, mu_batch, sigma2_batch

    
class PASEncoder(BaseModel): # PASEncoder
    
    def __init__(self, hidden_act_func, input_dim, arg_emb_dim, hidden_dim, mu_dim, sigma_dim, dropout_type, dropout_prob, avg_non_arg, num_embs):
        super().__init__()
        self.input_dim = input_dim
        self.mu_dim = mu_dim
        self.hidden_dim = hidden_dim
        self.arg_emb_dim = arg_emb_dim
        self.dropout_type = dropout_type
        self.dropout_prob = dropout_prob
        self.avg_non_arg = avg_non_arg

        self.embeds = nn.Embedding(num_embs + 1, input_dim, padding_idx = 0).requires_grad_(True)
        self.binomial_dist = torch.distributions.Bernoulli(dropout_prob) #torch.tensor(dropout_prob, device = self.device))
        # self.fc_pair = nn.Linear(input_dim + self.arg_emb_dim, hidden_dim, bias = False)
        # if hidden_act_func == 'ReLU' and False:
        #     nn.init.kaiming_normal_(self.fc_pair.weight, mode='fan_out', nonlinearity='relu')
        # elif hidden_act_func == 'LeakyReLU' and False:
        #     nn.init.kaiming_normal_(self.fc_pair.weight, a = 0.01, mode='fan_out', nonlinearity='leaky_relu')
        # else:
        #     # pass
        #     nn.init.xavier_uniform_(self.fc_pair.weight)
        self.act_hidden = getattr(nn, hidden_act_func)()
        self.fc_mu = nn.Linear(hidden_dim, mu_dim)
        # nn.init.xavier_uniform_(self.fc_mu.weight)
        self.fc_sigma = nn.Linear(hidden_dim, sigma_dim)
        # nn.init.xavier_uniform_(self.fc_sigma.weight)
    
    def forward(self, pred_func_nodes_ctxt_predargs, pred_func_nodes_ctxt_predargs_len, device, train_mode = 1):

        # dropout on ARG0

        num_nodes, num_pred_args = pred_func_nodes_ctxt_predargs.size()

        keep_arg0s = 1 - self.binomial_dist.sample([num_nodes]) * train_mode
        keep_arg0s = keep_arg0s.to(device)
        keep_arg0s = keep_arg0s.to(torch.int64)

        pred_func_nodes_ctxt_predargs[:,0] *= keep_arg0s

        pred_arg_embs = self.embeds(pred_func_nodes_ctxt_predargs + 1)

        # mean then relu (worse)
        mean_a_padded = torch.mean(pred_arg_embs, dim = 1)
        # rescale padded mean
        mean_a = mean_a_padded * (num_pred_args / pred_func_nodes_ctxt_predargs_len).unsqueeze(-1)
        a = self.act_hidden(mean_a)
        h = a

        mu = self.fc_mu(h)

        mu_batch = mu.unsqueeze(dim = 0)
        log_sigma2 = self.fc_sigma(h)
        log_sigma2_batch = log_sigma2.unsqueeze(dim = 0)
        # print ()

        return mu_batch, log_sigma2_batch

class MyEncoder(BaseModel):
    
    def __init__(self, hidden_act_func, input_dim, arg_emb_dim, hidden_dim, mu_dim, sigma_dim, dropout_type, dropout_prob, avg_non_arg, num_embs):
        super().__init__()
        self.input_dim = input_dim
        self.mu_dim = mu_dim
        self.hidden_dim = hidden_dim
        self.arg_emb_dim = arg_emb_dim
        self.dropout_type = dropout_type
        self.dropout_prob = dropout_prob
        self.avg_non_arg = avg_non_arg

        self.embeds = nn.Embedding(num_embs + 1, input_dim, padding_idx = 0).requires_grad_(True)
        self.arg_embeds = nn.Embedding(10 + 1, self.arg_emb_dim).requires_grad_(True)
        self.binomial_dist = torch.distributions.Bernoulli(dropout_prob) #torch.tensor(dropout_prob, device = self.device))
        self.fc_pair = nn.Linear(input_dim + self.arg_emb_dim, hidden_dim, bias = False)
        if hidden_act_func == 'ReLU' and False:
            nn.init.kaiming_normal_(self.fc_pair.weight, mode='fan_out', nonlinearity='relu')
        elif hidden_act_func == 'LeakyReLU' and False:
            nn.init.kaiming_normal_(self.fc_pair.weight, a = 0.01, mode='fan_out', nonlinearity='leaky_relu')
        else:
            # pass
            nn.init.xavier_uniform_(self.fc_pair.weight)
        self.act_hidden = getattr(nn, hidden_act_func)()
        self.fc_mu = nn.Linear(hidden_dim, mu_dim)
        # nn.init.xavier_uniform_(self.fc_mu.weight)
        self.fc_sigma = nn.Linear(hidden_dim, sigma_dim)
        # nn.init.xavier_uniform_(self.fc_sigma.weight)
    
    def forward(self, pred_func_nodes_ctxt_preds, pred_func_nodes_ctxt_args, device, train_mode = 1):

        # dropout on ARG0

        num_nodes, num_pred_args = pred_func_nodes_ctxt_preds.size()

        keep_arg0s = 1 - self.binomial_dist.sample([num_nodes]) * train_mode
        keep_arg0s = keep_arg0s.to(device)
        keep_arg0s = keep_arg0s.to(torch.int64)

        pred_func_nodes_ctxt_preds[:,0] *= keep_arg0s
        pred_func_nodes_ctxt_args[:,0] *= keep_arg0s

        preds_emb = self.embeds(pred_func_nodes_ctxt_preds + 1)
        args_emb = self.arg_embeds(pred_func_nodes_ctxt_args + 1)

        pred_arg_embs = torch.cat((preds_emb, args_emb), dim = 2) # / (1 - self.dropout_prob / num_nodes)

        pred_arg_embs_lin = self.fc_pair(pred_arg_embs)

        # avg non-arg information
        if self.avg_non_arg:
            # average also args? currently yes
            num_non_zeros = torch.count_nonzero(pred_func_nodes_ctxt_args, dim = 1)
            num_zeros = torch.tensor(num_pred_args, dtype = torch.float32, device = device) - num_non_zeros
            num_zeros = num_zeros.unsqueeze(-1)
            num_non_zeros = num_non_zeros.unsqueeze(-1)
            one = torch.tensor(1.0, dtype = torch.float32, device = device)
            non_arg_mul_factor = torch.where(pred_func_nodes_ctxt_args != 0, one/num_non_zeros, one/num_zeros)
            pred_arg_embs_lin = torch.mul(non_arg_mul_factor.unsqueeze(dim = -1), pred_arg_embs_lin)

        # transform then relu then mean
        a = self.act_hidden(pred_arg_embs_lin)
        # sum_a = torch.sum(a, dim = 1)
        mean_a = torch.mean(a, dim = 1)
        h = mean_a

        # # transform then mean then relu (worse)
        # mean_a = torch.mean(pred_arg_embs_lin, dim = 1)
        # a = self.act_hidden(mean_a)
        # h = a

        mu = self.fc_mu(h)
        # print (mu.shape)
        mu_batch = mu.unsqueeze(dim = 0)
        log_sigma2 = self.fc_sigma(h)
        log_sigma2_batch = log_sigma2.unsqueeze(dim = 0)

        return mu_batch, log_sigma2_batch



class OnePlaceSemFunc(BaseModel):

    def __init__(self, input_dim, hidden_dim, hidden_act_func, output_act_func):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if hidden_act_func == 'ReLU':
            # fc1_normal_dist = torch.distributions.Normal(torch.zeros(hidden_dim), torch.sqrt(2/hidden_dim))
            # fc1_weights = fc2_normal_dist.sample(torch.Size([1, hidden_act_func]))
            # self.fc1 = torch.nn.Parameter(fc2_weights)
            nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        elif hidden_act_func == 'LeakyReLU':
            nn.init.kaiming_normal_(self.fc1.weight, a = 0.01, mode='fan_out', nonlinearity='leaky_relu')
        else:
            pass
        self.act1 = getattr(nn, hidden_act_func)()
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.act2 = getattr(nn, output_act_func)()
        self.log_act2 = getattr(nn, "LogSigmoid")()
        
    def forward(self, x, take_log = False):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        # if take_log:
            # x = self.log_act2(x)
        # else:
        x = self.act2(x)

        return x
    
class TwoPlaceSemFunc(BaseModel):
    
    def __init__(self, input_dim, hidden_dim, hidden_act_func, output_act_func):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if hidden_act_func == 'ReLU':
            # fc1_normal_dist = torch.distributions.Normal(torch.zeros(hidden_dim), torch.sqrt(2/hidden_dim))
            # fc1_weights = fc2_normal_dist.sample(torch.Size([1, hidden_act_func]))
            # self.fc1 = torch.nn.Parameter(fc2_weights)
            nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        elif hidden_act_func == 'LeakyReLU':
            nn.init.kaiming_normal_(self.fc1.weight, a = 0.01, mode='fan_out', nonlinearity='leaky_relu')
        else:
            pass
        self.act1 = getattr(nn, hidden_act_func)()
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.act2 = getattr(nn, output_act_func)()
        self.log_act2 = getattr(nn, "LogSigmoid")()
        
    def forward(self, x, take_log = False):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        # if take_log:
            # x = self.log_act2(x)
        # else:
        x = self.act2(x)
        return x


class SharedSemFunc(BaseModel):
    
    def __init__(self, input_dim, hidden_dim, shared_dim, hidden_act_func, output_act_func, num_sem_funcs):
        super().__init__()
        fc1_tensor = nn.Empty((shared_dim, input_dim, hidden_dim), dtype = torch.float64)
        self.fc1_shared = nn.Parameter(fc1_tensor)
        self.embeds = nn.Embedding(num_sem_funcs + 1, shared_dim, padding_idx = 0).requires_grad_(True)
        if hidden_act_func == 'ReLU':
            # fc1_normal_dist = torch.distributions.Normal(torch.zeros(hidden_dim), torch.sqrt(2/hidden_dim))
            # fc1_weights = fc2_normal_dist.sample(torch.Size([1, hidden_act_func]))
            # self.fc1 = torch.nn.Parameter(fc2_weights)
            nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        elif hidden_act_func == 'LeakyReLU':
            nn.init.kaiming_normal_(self.fc1.weight, a = 0.01, mode='fan_out', nonlinearity='leaky_relu')
        else:
            pass
        self.act1 = getattr(nn, hidden_act_func)()
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.act2 = getattr(nn, output_act_func)()
        self.log_act2 = getattr(nn, "LogSigmoid")()
        
    def forward(self, x, take_log = False):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        # if take_log:
            # x = self.log_act2(x)
        # else:
        x = self.act2(x)
        return x

class OneLayerSemFuncsDecoder(BaseModel):

    def __init__(self, input_dim, freq_sampling, num_negative_samples, sparse_sem_funcs, use_truth, alpha, contrastive_loss, num_sem_funcs, train = True, **kwargs):
        
        super().__init__()

        self.train = train
        self.sparse_sem_funcs = sparse_sem_funcs
        self.use_truth = use_truth
        self.freq_sampling = freq_sampling
        self.num_negative_samples = num_negative_samples
        self.alpha = alpha
        self.contrastive_loss = contrastive_loss
        # matrix sem funcs
        if not self.sparse_sem_funcs:
            self.sem_funcs = torch.empty((num_sem_funcs, input_dim + 1), requires_grad = False)
            for i in range(num_sem_funcs):
                nn.init.xavier_uniform_(self.sem_funcs[i].unsqueeze(0))
                self.sem_funcs[i][-1] = 0.0
            for i in range(input_dim + 1):
                self.sem_funcs[0][i] = 0.0
            # self.sem_funcs[0][-1] = 1.0
            self.sem_funcs = nn.Parameter(self.sem_funcs)

        # emb sem funcs
        else:
            self.sem_funcs = nn.Embedding(num_sem_funcs, input_dim + 1, sparse = True).requires_grad_(False)
            for i in range(num_sem_funcs):
                nn.init.xavier_uniform_(self.sem_funcs.weight[i].unsqueeze(0))
                self.sem_funcs.weight[i][-1] = 0.0
            # self.sem_funcs.weight[0][-1] = 1.0
            self.sem_funcs.requires_grad_(True)
        
        if train:
            if use_truth:
                pred_func2cnt, pred_funcs = kwargs['pred_func2cnt'], kwargs['pred_funcs']
                one_place_pred_func2cnt = {pred_func_ix: cnt for pred_func_ix, cnt in pred_func2cnt.items() if pred_funcs[pred_func_ix].endswith("@ARG0")}
                self.one_place_pred_func_names = list(one_place_pred_func2cnt.keys())
                two_place_pred_func2cnt = {pred_func_ix: cnt for pred_func_ix, cnt in pred_func2cnt.items() if not pred_funcs[pred_func_ix].endswith("@ARG0")}
                self.two_place_pred_func_names = list(two_place_pred_func2cnt.keys())
                self.pred_func_probs = None
                if self.freq_sampling:
                    one_place_freq_sum = sum(one_place_pred_func2cnt.values())
                    two_place_freq_sum = sum(two_place_pred_func2cnt.values())
                    self.one_place_pred_func_probs = [pred_func2cnt[pred_func_name]/one_place_freq_sum for pred_func_name in self.one_place_pred_func_names]
                    self.two_place_pred_func_probs = [pred_func2cnt[pred_func_name]/two_place_freq_sum for pred_func_name in self.two_place_pred_func_names]
                else:
                    self.one_place_pred_func_probs = None
                    self.two_place_pred_func_probs = None
                self.num_neg_sampled = 9999999
                self.regen_one_place_negative_samples()
                self.regen_two_place_negative_samples()
            
            if not use_truth:
                self.pred_ix2arg_num2pred_func_ix = kwargs["pred_ix2arg_num2pred_func_ix"]
                self.arg_num_sum2preds_ix = kwargs["arg_num_sum2preds_ix"]
                self.pred_func_ix2arg_num = kwargs["pred_func_ix2arg_num"]
                # sampling
                self.log_neg_samp_ratio_inv = torch.log(self.pred_ix2arg_num2pred_func_ix.size(0) / torch.tensor(self.num_negative_samples))
                self.num_neg_sampled = 9999999
                self.arg_num_sum2neg_preds_ix_sampled = defaultdict(list)
                self.arg_num_sum2neg_idx = defaultdict(int)
                if self.freq_sampling:
                    pred2cnt = kwargs["pred2cnt"]
                    self.arg_num_sum2pred_probs = defaultdict(list)
                    self.arg_num_sum2neg_pred_ix = defaultdict(list)
                    # print (self.arg_num_sum2preds_ix)
                    for arg_num_sum, preds_ix in self.arg_num_sum2preds_ix.items():
                        preds_freq = [pred2cnt[pred_ix] for pred_ix in preds_ix]
                        preds_freq_sum = sum(preds_freq)
                        self.arg_num_sum2pred_probs[arg_num_sum] = [pred_freq / preds_freq_sum for pred_freq in preds_freq]
                        # self.arg_num_sum2preds_ix[arg_num_sum] = preds_ix
                        self.regen_negative_preds(arg_num_sum)

        # elif not use_truth:
        #     pred2cnt = kwargs['pred2cnt']
        #     self.arg_num_sum2preds_ix = kwargs['arg_num_sum2preds_ix']
        #     self.arg_num_sum2ix = kwargs["arg_num_sum2ix"]
        #     max_arg_num_sum_ix = max(self.arg_num_sum2ix)
        #     self.ix2arg_num_sum = [0 for i in range(max_arg_num_sum_ix + 1)]
        #     for arg_num_sum, ix in enumerate(self.arg_num_sum2ix):
        #         if ix != -1:
        #             self.ix2arg_num_sum[ix] = arg_num_sum
        #     self.ix2arg_num_sum = torch.tensor(self.ix2arg_num_sum, dtype = torch.long)
        #     self.pred_func_ix2arg_num = kwargs["pred_func_ix2arg_num"]
        #     # self.pred_ix2arg_num2pred_func_ix = kwargs["pred_ix2arg_num2pred_func_ix"]
        #     self.pred_ix2arg0_pred_func_ix = kwargs["pred_ix2arg0_pred_func_ix"]
        #     self.pred_ix2non_arg0_num2cnt = kwargs["pred_ix2non_arg0_num2cnt"].to(torch.float32)
        #     self.num_neg_group_sampled = 99
        #     self.num_neg_preds_sampled = self.num_negative_samples * self.num_neg_group_sampled
        #     self.arg_num_sum_ix2neg_preds_gen_chunks = torch.zeros((max_arg_num_sum_ix + 1, self.num_neg_group_sampled, self.num_negative_samples), dtype = torch.int32)
        #     self.arg_num_sum_ix2neg_preds_start = torch.zeros(max_arg_num_sum_ix + 1, dtype = torch.int32)
        #     self.num_neg_pred_funcs_sampled = 9
        #     if self.freq_sampling:
        #         self.arg_num_sum_ix2pred_probs = [0 for i in range(max(self.arg_num_sum2ix + 1))]
        #         for arg_num_sum, preds_ix in self.arg_num_sum2preds_ix.items():
        #             arg_num_sum_ix = self.arg_num_sum2ix[arg_num_sum]
        #             if arg_num_sum_ix < 0:
        #                 continue
        #             preds_freq = [pred2cnt[pred_ix] for pred_ix in preds_ix]
        #             preds_freq_sum = sum(preds_freq)
        #             self.arg_num_sum_ix2pred_probs[arg_num_sum_ix] = [pred_freq / preds_freq_sum for pred_freq in preds_freq]
        #             self.pred_ix2non_arg0_num2pred_func_ix = kwargs["pred_ix2non_arg0_num2pred_func_ix"]
        #     else:
        #         self.arg_num_sum_ix2pred_probs = None
        #         self.pred_func_ix2cnt = None
        #     term_arg_num_sum_list = self.ix2arg_num_sum
        #     term_neg_preds_idx = None
        #     self.regen_negative_preds(term_arg_num_sum_list, term_neg_preds_idx)


    def tsr_to_device(self, device):
        if not self.use_truth and self.train:
            # self.arg_num_sum2preds_ix = self.arg_num_sum2preds_ix.to(device)
            self.pred_func_ix2arg_num = self.pred_func_ix2arg_num.to(device)
            self.pred_ix2arg_num2pred_func_ix = self.pred_ix2arg_num2pred_func_ix.to(device)
            self.log_neg_samp_ratio_inv = self.log_neg_samp_ratio_inv.to(device)
            # self.arg_num_sum2ix = self.arg_num_sum2ix.to(device)
            # self.pred_ix2arg0_pred_func_ix = self.pred_ix2arg0_pred_func_ix.to(device)
            # self.pred_ix2non_arg0_num2cnt = self.pred_ix2non_arg0_num2cnt.to(device)
            # self.ix2arg_num_sum = self.ix2arg_num_sum.to(device)
            # self.arg_num_sum_ix2neg_preds_gen_chunks = self.arg_num_sum_ix2neg_preds_gen_chunks.to(device)
            # self.arg_num_sum_ix2neg_preds_start = self.arg_num_sum_ix2neg_preds_start.to(device)
            # self.pred_ix2non_arg0_num2pred_func_ix = self.pred_ix2non_arg0_num2pred_func_ix.to(device)

    # methods for truth mode
    def regen_one_place_negative_samples(self):
        self.neg_samp_one_place_pred_funcs = np.random.choice(self.one_place_pred_func_names, self.num_neg_sampled, replace = True, p = self.one_place_pred_func_probs)
        self.neg_samp_one_place_idx = 0
    def regen_two_place_negative_samples(self):
        self.neg_samp_two_place_pred_funcs = np.random.choice(self.two_place_pred_func_names, self.num_neg_sampled, replace = True, p = self.two_place_pred_func_probs)
        self.neg_samp_two_place_idx = 0

    def get_neg_sem_funcs(self, is_one_place, pos_sem_func):
        if is_one_place:
            if self.neg_samp_one_place_idx + self.num_negative_samples >= self.num_neg_sampled:
                self.regen_one_place_negative_samples()
            sampled_pred_funcs = self.neg_samp_one_place_pred_funcs[self.neg_samp_one_place_idx : self.neg_samp_one_place_idx + self.num_negative_samples]
            self.neg_samp_one_place_idx += self.num_negative_samples 

            # while pos_sem_func in sampled_pred_funcs:
            #     sampled_pred_funcs.remove(pos_sem_func)
            #     if self.neg_samp_one_place_idx + 1 >= self.num_neg_sampled:
            #         self.regen_one_place_negative_samples()
            #     sampled_pred_funcs += self.neg_samp_one_place_pred_funcs[self.neg_samp_one_place_idx : self.neg_samp_one_place_idx + 1]
            #     self.neg_samp_one_place_idx += 1
        else:
            if self.neg_samp_two_place_idx + self.num_negative_samples >= self.num_neg_sampled:
                self.regen_two_place_negative_samples()
            sampled_pred_funcs = self.neg_samp_two_place_pred_funcs[self.neg_samp_two_place_idx : self.neg_samp_two_place_idx + self.num_negative_samples]
            self.neg_samp_two_place_idx += self.num_negative_samples 

            # while pos_sem_func in sampled_pred_funcs:
            #     sampled_pred_funcs.remove(pos_sem_func)
            #     if self.neg_samp_two_place_idx + 1 >= self.num_neg_sampled:
            #         self.regen_two_place_negative_samples()
            #     sampled_pred_funcs += self.neg_samp_two_place_pred_funcs[self.neg_samp_two_place_idx : self.neg_samp_two_place_idx + 1]
            #     self.neg_samp_two_place_idx += 1
        return sampled_pred_funcs


    # methods for gen mode
    def regen_negative_preds(self, arg_num_sum):
        self.arg_num_sum2neg_preds_ix_sampled[arg_num_sum] = np.random.choice(self.arg_num_sum2preds_ix[arg_num_sum], self.num_neg_sampled, replace = True, p = self.arg_num_sum2pred_probs[arg_num_sum])
        # print ("self.arg_num_sum2neg_preds_ix_sampled[arg_num_sum]", self.arg_num_sum2neg_preds_ix_sampled[arg_num_sum])
        self.arg_num_sum2neg_idx[arg_num_sum] = 0


    def get_term_neg_samples(self, term_pred_funcs_ix_list, term_arg_num_sum_list, device):
        
        term_neg_preds = []
        for arg_num_sum in term_arg_num_sum_list:
            arg_num_sum = arg_num_sum.item()
            if self.arg_num_sum2neg_idx[arg_num_sum] + self.num_negative_samples > self.num_neg_sampled:
                self.regen_negative_preds(arg_num_sum)
            neg_preds = self.arg_num_sum2neg_preds_ix_sampled[arg_num_sum][
                self.arg_num_sum2neg_idx[arg_num_sum]: self.arg_num_sum2neg_idx[arg_num_sum] + self.num_negative_samples
            ]
            term_neg_preds.append(neg_preds)
            self.arg_num_sum2neg_idx[arg_num_sum] += self.num_negative_samples
        term_neg_preds = torch.tensor(np.array(term_neg_preds).astype(np.int64))
        term_pred_funcs_arg_num = F.embedding(term_pred_funcs_ix_list, self.pred_func_ix2arg_num.unsqueeze(-1)).squeeze(-1)
        term_pred_funcs_arg_num = term_pred_funcs_arg_num.unsqueeze(1)
        term_pred_funcs_arg_num = term_pred_funcs_arg_num.expand([-1, self.num_negative_samples, -1])
        term_neg_pred_func_samps = torch.gather(self.pred_ix2arg_num2pred_func_ix[term_neg_preds], -1, term_pred_funcs_arg_num)
        return term_neg_pred_func_samps

    def forward(self, *args, samp_neg, variational = True, device = None, return_sum = True):
        if self.use_truth or not samp_neg:
            return self._forward_truth(*args, samp_neg, variational, device, return_sum)
        else:
            return self._forward_gen(*args, samp_neg, variational, device, return_sum)


    def _forward_truth(self, mu_batch, sigma2_batch, logic_expr_batch, args_vars_batch, samp_neg, variational, device, return_sum = True):
        # currently support batch_size = 1
        # mu_batch.shape = [1, num_nodes, mu_dim]
        # sigma2_batch.shape = [1, num_nodes, 1]
        # logic_expr: list (of sem_func_ix)
        # args_vars: list of two lists, each contain a ix (0 is padded for one-place))

        mu = mu_batch[0]
        sigma2 = sigma2_batch[0]
        pos_sem_funcs = logic_expr_batch[0]
        args_vars = args_vars_batch[0]
        num_nodes, mu_dim = mu.size()

        # mu_zero_pad = torch.zeros((1, mu_dim), device = device)
        # mu_pad = torch.cat((mu_zero_pad, mu))
        mu_pad = F.pad(mu, (0, 0, 1, 0), "constant", 0)
        # signma_zero_pad = torch.zeros((1, 1), device = device)
        # sigma2_pad = torch.cat((signma_zero_pad, sigma2))
        sigma2_pad = F.pad(sigma2, (0, 0, 1, 0), "constant", 0)

        mu_x = torch.index_select(mu_pad, 0, args_vars[0]) 
        mu_y = torch.index_select(mu_pad, 0, args_vars[1]) 
        mu_xy = torch.cat((mu_x, mu_y), dim = 1)
        # mu_xy_bias = torch.cat((mu_xy, torch.ones((num_nodes, 1), device = device)), dim = 1)
        mu_xy_bias = F.pad(mu_xy, (0,1), "constant", 1)

        # diagonal covariance matrix
        sigma2_x = torch.index_select(sigma2_pad, 0, args_vars[0]).expand([-1, mu_dim])
        sigma2_y = torch.index_select(sigma2_pad, 0, args_vars[1]).expand([-1, mu_dim])
        sigma2_xy = torch.cat((sigma2_x, sigma2_y), dim = 1)
        # sigma2_xy_bias = torch.cat((sigma2_xy, torch.zeroes((num_nodes, 1))), dim = 1)
        sigma2_xy_bias = F.pad(sigma2_xy, (0,1), "constant", 0)

        
        neg_sem_funcs = None
        if samp_neg:
            neg_sem_funcs = torch.tensor(np.array([self.get_neg_sem_funcs(var_ix == 0, pos_sem_funcs[idx]) for idx, var_ix in enumerate(args_vars[1])]).flatten(), device = device)
        # pos_neg_sem_funcs = [pos_sem_funcs, neg_sem_funcs]

        # log_truths = torch.tensor([], device = device)
        # log_neg_truths = torch.tensor([], device = device)
        # for itr_idx, sem_funcs in enumerate(pos_neg_sem_funcs):

        if not self.sparse_sem_funcs:
            sem_funcs_w =  torch.index_select(self.sem_funcs, 0, pos_sem_funcs)
        else:
            sem_funcs_w =  self.sem_funcs(pos_sem_funcs)
            
        # mu_a = w^T dot mu
        if variational:
            mu_a = torch.sum(mu_xy_bias * sem_funcs_w, dim = 1) 
            sem_funcs_w2 = sem_funcs_w * sem_funcs_w
            # sigma2_a = w^T cov_xy w
            sigma2_a = torch.sum(sigma2_xy_bias * sem_funcs_w2, dim = 1)

            kappa = (1 / torch.sqrt( 1 + math.pi * sigma2_a / 8) )
            truths = torch.sigmoid(kappa * mu_a)
        else:
            mu_a = torch.sum(mu_xy_bias * sem_funcs_w, dim = 1) 
            truths = torch.sigmoid(mu_a) + sigma2_xy_bias.sum() * 0


        pos_log_truths = torch.log(truths)
        # log_truth += torch.sum(log_truths)

        neg_log_truths = torch.tensor([0])
        if neg_sem_funcs != None:
            

            if not self.sparse_sem_funcs:
                sem_funcs_w =  torch.index_select(self.sem_funcs, 0, neg_sem_funcs)
            else:
                sem_funcs_w =  self.sem_funcs(neg_sem_funcs)

            # mu_a = w^T dot mu
            # repeat for negative samples
            mu_xy_bias = torch.repeat_interleave(mu_xy_bias, self.num_negative_samples, dim = 0)
            mu_a = torch.sum(mu_xy_bias * sem_funcs_w, dim = 1) 
            sem_funcs_w2 = sem_funcs_w * sem_funcs_w
            # sigma2_a = w^T cov_xy w
            # repeat for negative samples
            sigma2_xy_bias = torch.repeat_interleave(sigma2_xy_bias, self.num_negative_samples, dim = 0)
            sigma2_a = torch.sum(sigma2_xy_bias * sem_funcs_w2, dim = 1)
            kappa = (1 / torch.sqrt( 1 + math.pi * sigma2_a / 8) )
            # negate truthness for negative sampels
            if variational:
                truths = torch.sigmoid(- kappa * mu_a)
            else:
                truths = torch.sigmoid(- mu_a) + sigma2_a.sum() * 0
            neg_log_truths = torch.log(truths)
            pos_neg_log_truths = torch.cat((pos_log_truths, neg_log_truths))
            # log_truth += torch.sum(log_truths)
            pos_neg_log_truth = torch.sum(pos_neg_log_truths)

        if not samp_neg:
            pos_neg_log_truths = pos_log_truths
            pos_neg_log_truth = torch.sum(pos_log_truths)
        pos_neg_log_truth_batch = pos_neg_log_truth.unsqueeze(0)

        if return_sum:
            return pos_neg_log_truth_batch, torch.sum(pos_log_truths),  torch.sum(neg_log_truths)
        else:
            return pos_neg_log_truths.unsqueeze(0), torch.sum(pos_log_truths),  torch.sum(neg_log_truths)

    def _forward_gen(self,  mu_batch, sigma2_batch, terms_pred_funcs_ix_list, terms_vars_list, terms_arg_num_sum_list, samp_neg, variational, device, return_sum = True):
        # currently support batch_size = 1
        # mu_batch.shape = [1, num_nodes, mu_dim]
        # sigma2_batch.shape = [1, num_nodes, 1]

        # terms = []
        pos_terms = []
        neg_terms = []

        mu = mu_batch[0]
        sigma2 = sigma2_batch[0]
        num_nodes, mu_dim = mu.size()

        mu_pad = F.pad(mu, (0, 0, 1, 0), "constant", 0)
        sigma2_pad = F.pad(sigma2, (0, 0, 1, 0), "constant", 0)

        for term_idx, term_pred_funcs_ix_list in enumerate(terms_pred_funcs_ix_list):

            term_pred_funcs_ix_list = term_pred_funcs_ix_list.to(device)
            term_vars_list = terms_vars_list[term_idx].to(device)
            term_arg_num_sum_list = terms_arg_num_sum_list[term_idx].to(device)
            # term_num_pred_funcs = terms_num_pred_funcs[term_idx].to(device)
            # print (term_num_pred_funcs, term_arg_num_sum_list)

            # print (term_vars_list, mu_pad)
            mu_x_y = F.embedding(term_vars_list, mu_pad)
            mu_xy =  torch.flatten(mu_x_y, start_dim = 2)
            mu_xy_bias = F.pad(mu_xy, (0,1), "constant", 1)
            
            if variational:
                sigma2_x_y = F.embedding(term_vars_list, sigma2_pad).expand((-1, -1, -1, mu_dim))
                sigma2_xy =  torch.flatten(sigma2_x_y, start_dim = 2)
                sigma2_xy_bias = F.pad(sigma2_xy, (0,1), "constant", 0)

                # positive (observed) samples
                sem_funcs_w = self.sem_funcs(term_pred_funcs_ix_list)
                mu_a = torch.sum(mu_xy_bias * sem_funcs_w, dim = 2) 
                sem_funcs_w2 = sem_funcs_w * sem_funcs_w
                sigma2_a = torch.sum(sigma2_xy_bias * sem_funcs_w2, dim = 2)

                kappa = (1 / torch.sqrt( 1 + math.pi * sigma2_a / 8) )
                truths = torch.sigmoid(kappa * mu_a)
            else:
                # positive (observed) samples
                sem_funcs_w = self.sem_funcs(term_pred_funcs_ix_list)
                mu_a = torch.sum(mu_xy_bias * sem_funcs_w, dim = 2) 
                truths = torch.sigmoid(mu_a) + sigma2_pad.sum() * 0 # unused param

            if not self.contrastive_loss:
                sub_pos_terms = torch.prod(truths, dim = 1)
            else:
                sub_pos_terms = truths.flatten()
            pos_terms.append(sub_pos_terms)

            # negative samples
            if samp_neg:
                term_neg_pred_func_samps = self.get_term_neg_samples(term_pred_funcs_ix_list, term_arg_num_sum_list, device).to(device)

                if variational:
                    # print (self.sem_funcs)
                    mu_xy_bias = mu_xy_bias.unsqueeze(1)
                    sigma2_xy_bias = sigma2_xy_bias.unsqueeze(1)
                    # print (1, term_neg_pred_func_samps)
                    sem_funcs_w = self.sem_funcs(term_neg_pred_func_samps)
                    # print (term_pred_funcs_ix_list.shape)
                    # print (term_arg_num_sum_list.shape)
                    # print (mu_xy_bias.shape, sem_funcs_w.shape)
                    mu_a = torch.sum(mu_xy_bias * sem_funcs_w, dim = 3)
                    sem_funcs_w2 = sem_funcs_w * sem_funcs_w
                    sigma2_a_neg = torch.sum(sigma2_xy_bias * sem_funcs_w2, dim = 3)
                    
                    kappa = (1 / torch.sqrt( 1 + math.pi * sigma2_a_neg / 8) )
                else:
                    mu_xy_bias = mu_xy_bias.unsqueeze(1)
                    sem_funcs_w = self.sem_funcs(term_neg_pred_func_samps)
                    mu_a = torch.sum(mu_xy_bias * sem_funcs_w, dim = 3)
                    kappa = 1 + sigma2_pad.sum() * 0 # unused param

                if not self.contrastive_loss:
                    truths_neg = torch.sigmoid(kappa * mu_a)
                    sub_neg_terms = torch.sum(torch.prod(truths_neg, dim = 2), dim = 1) # *

                else:
                    truths_neg = torch.sigmoid(- kappa * mu_a)
                    sub_neg_terms = truths_neg.flatten()

                neg_terms.append(sub_neg_terms)

        # terms_cat = torch.cat(terms)
        # sum_log_terms_batch = torch.sum(torch.log(terms_cat)).unsqueeze(0)
        neg_terms_cat_sum = 0
        if not self.contrastive_loss:
            pos_terms_cat = torch.cat(pos_terms)
            pos_terms_cat_sum = torch.sum(torch.log(pos_terms_cat))
            if samp_neg:
                neg_terms_cat = torch.cat(neg_terms)
                neg_terms_cat_sum = torch.sum(torch.log(neg_terms_cat))
                sum_log_terms = (self.alpha + 1) * pos_terms_cat_sum - neg_terms_cat_sum - pos_terms_cat.size(0) * self.log_neg_samp_ratio_inv
            else:
                sum_log_terms = pos_terms_cat_sum
            sum_log_terms_batch = sum_log_terms.unsqueeze(0)
        else:
            pos_terms_cat = torch.cat(pos_terms)
            pos_terms_cat_sum = torch.sum(torch.log(pos_terms_cat)) 
            if samp_neg:
                neg_terms_cat = torch.cat(neg_terms)
                neg_terms_cat_sum = torch.sum(torch.log(neg_terms_cat))
                sum_log_terms = pos_terms_cat_sum + neg_terms_cat_sum
            else:
                sum_log_terms = pos_terms_cat_sum
            sum_log_terms_batch = sum_log_terms.unsqueeze(0)


        return sum_log_terms_batch, pos_terms_cat_sum/ pos_terms_cat.size(0), neg_terms_cat_sum / neg_terms_cat.size(0)



