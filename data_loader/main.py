import os
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import dataset
from pprint import pprint

#Initialzation
dummy = dataset.dmrsDataset("./../data/dmrs_dummy")
vector_dict={}
all_preds=dummy.get_pred()
noun_preds=dummy.get_pred("x")
verb_preds=dummy.get_pred("e")
verb_connection_list=dummy.get_verb_connection()
function_dict=dummy.get_function_dict()

for pred in all_preds:
    vector_dict[pred]=torch.rand(1,10)
    


function_dict=dummy.get_function_dict()

#the specific function to get all the results

    

#train the function using constructed nouns and verbs
#get a dictionary for verbs: each verb serves as a key to a list of concantenated data
verb_cat_dict={}
for verb_pred in verb_preds:
    verb_cat_dict[verb_pred]=[]
    for verb_connection in verb_connection_list[verb_pred]:
        vector_cat = torch.cat((vector_dict[verb_connection[1]],vector_dict[verb_connection[0]],vector_dict[verb_connection[2]]),1)
        verb_cat_dict[verb_pred].append(vector_cat)
        

        
#use dataloader to iterate over to train the model: '_eat_v_1' will be the best example: let's say verb_cat_dict[verb_pred] is the whole list of (30,1) tensors that has to be trained to 1

#train the functions for verbs and nouns separately
for verb_pred in verb_preds:
    loader = DataLoader(dataset=verb_cat_dict[verb_pred],batch_size=1)
    

