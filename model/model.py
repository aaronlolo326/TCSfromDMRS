from filecmp import dircmp
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


# x = F.dropout(x, training=self.training)
# return F.log_softmax(x, dim=1)
    
class BSGEncoder(BaseModel):
    
    def __init__(self, hidden_act_func, input_dim, hidden_dim, mu_dim, sigma_dim, num_embs):
        super().__init__()
        self.input_dim = input_dim
        self.embeds = nn.Embedding(num_embs, input_dim)
        self.fc_pair = nn.Linear(input_dim * 2, hidden_dim, bias = False)
        self.act_hidden = getattr(nn, hidden_act_func)()
        self.fc_mu = nn.Linear(hidden_dim, mu_dim)
        self.fc_sigma = nn.Linear(hidden_dim, sigma_dim)
    
    def forward(self, targ_preds_ix_batch, lex_preds_ix_batch):

        lex_embeds = self.embeds(lex_preds_ix_batch)
        targ_embeds = self.embeds(targ_preds_ix_batch)

        targ_embeds_exp = targ_embeds.expand(
            targ_embeds.size(dim=0),
            targ_embeds.size(dim=1),
            lex_embeds.size(dim=2),
            self.input_dim)
        lex_embeds_exp = lex_embeds.expand(
            lex_embeds.size(dim=0),
            targ_embeds.size(dim=1),
            lex_embeds.size(dim=2),
            self.input_dim)

        paired = torch.cat((lex_embeds_exp, targ_embeds_exp), dim=3)
        z = self.fc_pair(paired)
        a = self.act_hidden(z)
        sum_a = torch.sum(a, dim = 2)
        mu_batch = self.fc_mu(sum_a)
        log_sigma2_batch = self.fc_sigma(sum_a)

        return mu_batch, log_sigma2_batch
    
class OnePlaceDecoder(BaseModel):
    
    def __init__(self, input_dim, hidden_dim, hidden_act_func, output_act_func):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.act1 = getattr(nn, hidden_act_func)()
        self.act2 = getattr(nn, output_act_func)()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x
    
class TwoPlaceDecoder(BaseModel):
    
    def __init__(self, input_dim, hidden_dim, hidden_act_func, output_act_func):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.act1 = getattr(nn, hidden_act_func)()
        self.act2 = getattr(nn, output_act_func)()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x