#%% preliminaries
# run as `python -m data_structures.CDR.process_data` after navigating to BioIFT
# ^^^allows absolute imports using project as root

import numpy as np # linear algebra
import os # working directory
import json
import torch as t

import bioc
from bioc import biocxml, biocjson

# from utils.utils import *
from data_structures.CDR.data_structures import *  


#%% settings
raw_data_dir = 'data/raw_data/CDR/CDR.Corpus.v010516'
output_data_dir = 'data/processed_data/CDR'

#raw_data_dir = '/Users/avivbrokman/Documents/Kentucky/Grad School/NLP/datasets/CDR/CDR.Corpus.v010516'
#output_data_dir = '/Users/avivbrokman/Documents/Kentucky/Grad School/NLP/projects/bioIFT/CDR'

#%% loading data
train_file = raw_data_dir + '/' + 'CDR_TrainingSet.BioC.xml'
with open(train_file, 'r') as fp:
    train_collection = biocxml.load(fp)
    
valid_file = raw_data_dir + '/' + 'CDR_DevelopmentSet.BioC.xml'
with open(valid_file, 'r') as fp:
    valid_collection = biocxml.load(fp)
    
test_file = raw_data_dir + '/' + 'CDR_TestSet.BioC.xml'
with open(test_file, 'r') as fp:
    test_collection = biocxml.load(fp)

#%% useful functions
train_data = CDRDataset(train_collection)
valid_data = CDRDataset(valid_collection)
test_data = CDRDataset(test_collection)

t.save(train_data, output_data_dir + '/' + 'train_data')    
t.save(valid_data, output_data_dir + '/' + 'valid_data')    
t.save(test_data, output_data_dir + '/' + 'test_data')    



