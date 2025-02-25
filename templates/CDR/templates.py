#%% libraries
import re
from templates.general_templates import *
from data_structures.CDR.data_structures import *
from utils.utils import *

#%% template 
class E2eReTemplate1(E2eReTemplate):
    dataset_name = 'CDR'
    label_name = 'relations'

    null_target = 'No chemical-induced diseases are established.'
    
    string_chooser = 'longest'
    
    def __init__(self, text, relations):
       
        self.relations = relations
        self.text = text

    @classmethod
    def from_example(cls, example):
        return cls(example.text, example.relations)
                        
    def _make_relation_target(self, head, tail):
        
        return f'The relation between {head} and {tail} exists'
    
    def _make_mini_target(self, relation):
        head = relation.output_head(self.string_chooser)
        tail = relation.output_tail(self.string_chooser)
        
        return self._make_relation_target(head, tail)
    
    def make_target_sequence(self):
        
        target = ''
        
        mini_targets = [self._make_mini_target(el) for el in self.relations]
                                                     
        target = f"{'; '.join(mini_targets)}."
        
        return target   
    
    def _extract_mini_prediction(self, string):
        match = re.search(r"relation between (.*) and (.*?) exists", string)
        
        if match:
            head = match.group(1)
            tail = match.group(2)
            
            return dict(head = head, tail = tail)
#%%
class E2eReHardTemplate1(E2eReTemplate1):
    is_trainable = False
    
    instruction = "Find all instances of relations between chemicals and diseases in the following passage:"
    
    def make_source_sequence(self):
    
       return f'{self.instruction}\n\n{self.text}\n\n'
    
         
class E2eReSoftTemplate1(E2eReTemplate1):
    is_trainable = True
    prompt_tokens = [f'<prompt_{i}>' for i in range(9)]
    prompt_string = ''.join(prompt_tokens)
    
    def make_source_sequence(self):
        
        return f'{self.text}{self.prompt_string}'
#%%    
class E2eReTupleTemplate(E2eReTemplate1):
    is_trainable = False
    
    instruction = "Find all instances of relations between chemicals and diseases in the following passage:"
                        
    def _make_relation_target(self, head, tail):
        
        return f'({head},{tail})' 
    
    def make_source_sequence(self):
    
       return f'{self.instruction}\n\n{self.text}\n\n'

    def _extract_mini_prediction(self, string):
        match = re.search(r'\(([^,]+),\s*([^)]+)\)', string)

        if match:
            head = match.group(1).strip()
            tail = match.group(2).strip()  
            
            return dict(head = head, tail = tail)
#%%
#%% NER
class NerTemplate1(MulticlassNerTemplate):
    dataset_name = 'CDR'
    label_name = 'mentions'


    class2text = {'Chemical': 'chemicals',
                  'Disease': 'diseases'}

class NerHardTemplate1(NerTemplate1):
    is_trainable = False
    
    instruction = "Find all chemicals and diseases discussed in the following passage:" + ' \n '
    
    def make_source_sequence(self):
    
        return f'{self.instruction}{self.text} \n '
    
class NerSoftTemplate1(NerTemplate1):
    is_trainable = True
    prompt_tokens = [f'<prompt_{i}>' for i in range(9)]
    prompt_string = ''.join(prompt_tokens)

    def make_source_sequence(self):
        
        return f'{self.text}{self.prompt_string}'
    




