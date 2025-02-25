#%% preliminaries
# run as `python -m data_structures.NaryDrugCombos.process_data` after navigating to BioIFT directory

import os # working directory
import torch as t

import datasets
from datasets import load_dataset

from utils.utils import *
from data_structures.NaryDrugCombos.data_structures import *  

#%%  paths

output_data_path = 'data/processed_data/NaryDrugCombos'

#%% loading data
huggingface_dataset = load_dataset('allenai/drug-combo-extraction')

train_data = huggingface_dataset['train']
test_data = huggingface_dataset['test']

#%% split train into train-valid
split_prop = 0.2
split = train_data.train_test_split(test_size = split_prop)
train_data = split['train']
valid_data = split['test']
#%% make datasets
train_data = NaryDrugCombosDataset(train_data)
valid_data = NaryDrugCombosDataset(valid_data)
test_data = NaryDrugCombosDataset(test_data)

#%% saving datasets
try:
    os.mkdir(output_data_path)
except:
    pass

t.save(train_data, f'{output_data_path}/train_data')    
t.save(valid_data, f'{output_data_path}/valid_data')    
t.save(test_data, f'{output_data_path}/test_data')    



