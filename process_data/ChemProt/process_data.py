#%% preliminaries
# run as `python -m data_structures.ChemProt.process_data` after navigating to BioIFT directory

import pandas as pd
import os # working directory
import torch as t


from utils.utils import *
from data_structures.ChemProt.data_structures import *  

#%%  paths
raw_data_path = 'data/raw_data/ChemProt'
train_path = f'{raw_data_path}/chemprot_training'
valid_path = f'{raw_data_path}/chemprot_development'
test_path = f'{raw_data_path}/chemprot_test_gs'

output_data_path = 'data/processed_data/ChemProt'

#%% loading data
# training data
train_text_df = pd.read_table(f"{train_path}/chemprot_training_abstracts.tsv", 
                                   header = None, keep_default_na = False, 
                                   names = ['pmid', 'title', 'abstract'], 
                                   encoding = 'utf-8')

train_mentions_df = pd.read_table(f"{train_path}/chemprot_training_entities.tsv", 
                                   header = None, keep_default_na = False, 
                                   names = ['pmid', 'mention_id', 'entity_type_string', 'start', 'end', 'string'], 
                                   encoding = 'utf-8')

train_relations_df = pd.read_table(f"{train_path}/chemprot_training_gold_standard.tsv", 
                                   header = None, keep_default_na = False, 
                                   names = ['pmid', 'predicate_short_string', 'head_mention_id', 'tail_mention_id'], 
                                   encoding = 'utf-8')

# validation data
valid_text_df = pd.read_table(f"{valid_path}/chemprot_development_abstracts.tsv", 
                                   header = None, keep_default_na = False, 
                                   names = ['pmid', 'title', 'abstract'], 
                                   encoding = 'utf-8')

valid_mentions_df = pd.read_table(f"{valid_path}/chemprot_development_entities.tsv", 
                                   header = None, keep_default_na = False, 
                                   names = ['pmid', 'mention_id', 'entity_type_string', 'start', 'end', 'string'], 
                                   encoding = 'utf-8')

valid_relations_df = pd.read_table(f"{valid_path}/chemprot_development_gold_standard.tsv", 
                                   header = None, keep_default_na = False, 
                                   names = ['pmid', 'predicate_short_string', 'head_mention_id', 'tail_mention_id'], 
                                   encoding = 'utf-8')

# test data
test_text_df = pd.read_table(f"{test_path}/chemprot_test_abstracts_gs.tsv", 
                                   header = None, keep_default_na = False, 
                                   names = ['pmid', 'title', 'abstract'], 
                                   encoding = 'utf-8')

test_mentions_df = pd.read_table(f"{test_path}/chemprot_test_entities_gs.tsv", 
                                   header = None, keep_default_na = False, 
                                   names = ['pmid', 'mention_id', 'entity_type_string', 'start', 'end', 'string'], 
                                   encoding = 'utf-8')

test_relations_df = pd.read_table(f"{test_path}/chemprot_test_gold_standard.tsv", 
                                   header = None, keep_default_na = False, 
                                   names = ['pmid', 'predicate_short_string', 'head_mention_id', 'tail_mention_id'], 
                                   encoding = 'utf-8')

#%% make datasets
train_data = ChemProtDataset(train_text_df, train_mentions_df, train_relations_df)
valid_data = ChemProtDataset(valid_text_df, valid_mentions_df, valid_relations_df)
test_data = ChemProtDataset(test_text_df, test_mentions_df, test_relations_df)


#%% saving datasets
try:
    os.mkdir(output_data_path)
except:
    pass

t.save(train_data, f'{output_data_path}/train_data')    
t.save(valid_data, f'{output_data_path}/valid_data')    
t.save(test_data, f'{output_data_path}/test_data')    



