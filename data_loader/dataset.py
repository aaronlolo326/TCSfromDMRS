import os
import json
import torch
from torch.utils.data import Dataset

from pprint import pprint

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm
    
import networkx as nx

from pprint import pprint
from src import util

    
class DmrsDataset(Dataset):
    
    def __init__(self, data_dir, transform = None, sample_only = False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        ## Lazy loading
        # self.data_dir = data_dir
        # data_info_dir = os.path.join(data_dir, "info")
        # with open(os.path.join(data_info_dir, "idx2file_path.json")) as f:
        #     idx2file_path = json.load(f)
        # self.idx2file_path = idx2file_path
        # self.transform = transform
        
        ## Load all at once
        self.data_dir = data_dir
        self.transform = transform
        self.instance_list = []
        for root, dirs, files in tqdm(os.walk(data_dir)):
            for file in files:
                if not util.is_data_json(file):
                    continue
                with open(os.path.join(self.data_dir, file)) as f:
                    idx2instance = json.load(f)
                    self.instance_list = [*self.instance_list, *list(idx2instance.values())]
                    
                if sample_only:
                    break
            if sample_only:
                break
                    
        # with open(os.path.join(data_info_dir, "idx2file_path.json")) as f:
        #     idx2file_path = json.load(f)
        # self.idx2file_path = idx2file_path
        # self.transform = transform       
        

    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, idx):        
        
        ## Lazy loading
#         file_path = self.idx2file_path[str(idx)]
#         with open(os.path.join(self.data_dir, file_path)) as f:
#             idx2instance = json.load(f)
#         sample = idx2instance[str(idx)]

#         if self.transform:
#             sample = self.transform(sample)

        ## Load all at once
        sample = self.instance_list[idx]
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
        

#     def __len__(self):
#         return len(self.json_data)

#     def __getitem__(self, idx):
#         return self.json_data[idx]
#         # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         # image = read_image(img_path)
#         # label = self.img_labels.iloc[idx, 1]
#         # if self.transform:
#         #     image = self.transform(image)
#         # if self.target_transform:
#         #     label = self.target_transform(label)
#         # return image, label
    
#     def get_pred(self,part_of_speech="no_input"):
#         pred_list = set()
#         for data in self.json_data:
#             for node in data["dmrs"]["nodes"]:
#                 if part_of_speech in ["x","e"]:
#                     if node["cvarsot"]==part_of_speech:
#                         pred_list.add(node["predicate"])
#                 else:
#                     if not(node["predicate"] in pred_list):
#                         pred_list.append(node["predicate"])
                         
        
#         return pred_list
    

    
    
    
    
#     def get_verb_connection(self):
        
#         # each verb serve as a key to a list of 3 tuples(the verb plus the two nouns it connects to)
#         pred_verb_list=self.get_pred("e")
#         verb_connection_dict={}
#         for pred_verb in pred_verb_list:
#             verb_connection_dict[pred_verb]=[]
        
#         for data in self.json_data:
#             sub_list=[]
#             verb=0

#             for node in data["dmrs"]["nodes"]:
#                 sub_list.append(node["predicate"])
#                 if node["cvarsot"]=="e":
#                     verb=node["predicate"]
#             verb_connection_dict[verb].append(sub_list)
            
#         return verb_connection_dict

    
    
    
    
#     def get_function_dict(self):
#         # each predicate will work as a key to its corresponding function(not defined)
#         predicate_verb_list=self.get_pred("e")
#         predicate_noun_list=self.get_pred("x")
#         funct_dict={}
#         for pred_verb in predicate_verb_list:

#             funct_dict[pred_verb]=function.Model(30,30)
#         for pred_noun in predicate_noun_list:

#             funct_dict[pred_noun]=function.Model(10,10)

#         return funct_dict