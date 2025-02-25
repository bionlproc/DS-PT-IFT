#%% RE template 
import re

from data_structures.DDI.data_structures import *
from utils.utils import *
from templates.general_templates import *

#%%
class E2eReTemplate1(E2eReTemplate):    
    
    dataset_name = 'DDI'
    label_name = 'relations'

    null_target = 'No drug-drug interactions are established.'
    
    class2text = {'INT': 'interaction', 
                  'MECHANISM': 'mechanism', 
                  'EFFECT': 'effect', 
                  'ADVISE': 'advice'}
    
    text2class = QueryDict(reverse_dict(class2text))
    
    string_chooser = 'longest'

    def __init__(self, text, relations):
       
        self.text = text
        self.relations = relations        

    # @classmethod
    # def from_example(cls, example):
    #     return cls(example.text, getattr(example, cls.label_name))

    def _make_relation_target(self, head, tail, predicate):

        return f'the relation between {head} and {tail} is {predicate}'
    
    def _make_mini_target(self, relation):
        
        head = relation.head.string
        tail = relation.tail.string

        predicate = self.class2text[relation.predicate]
        
        return self._make_relation_target(head, tail, predicate)

    def _extract_mini_prediction(self, string):
        # print('regex')
        match = re.search(r"relation between (.*) and (.*) is (.*)", string)
        
        # print('extract head and tail')
        if match:
            head = match.group(1)
            tail = match.group(2)
            predicate = match.group(3)
            predicate = self.text2class[predicate]
            
            return dict(head = head, tail = tail, predicate = predicate)

class E2eReTemplate2(E2eReTemplate):    
    
    null_target = 'No drug-drug interactions are established.'
    
    class2text = {'INT': 'interaction', 
                  'MECHANISM': 'mechanism', 
                  'EFFECT': 'effect', 
                  'ADVISE': 'advice'}
    
    text2class = QueryDict(reverse_dict(class2text))
    
    def __init__(self, text, relations):
       
        self.text = text
        self.relations = relations        
                                
    def _make_relation_target(self, head, tail, predicate):
        if predicate == 'MECHANISM':
            return f'the pharmacokinetic relation between {head} and {tail} is described'
        elif predicate == 'EFFECT':
            return f'the effect of administering {head} and {tail} in combination is described'
        elif predicate == 'ADVISE':
            return f'advice regarding administering {head} and {tail} in combination is described'
        elif predicate == 'INT':
            return f'the interaction between {head} and {tail} was discussed, but additional information was not provided'

    def _make_mini_target(self, relation):
        head = relation.head.string
        tail = relation.tail.string
        predicate = self.class2text[relation.predicate]
        
        return self._make_relation_target(head, tail, predicate)
    
    def _extract_relation_regex(self, raw_string, sequence, predicate):
        match = re.search(raw_string, sequence)
        
        if match:
            head = match.group(1)
            tail = match.group(2)
            
            return dict(head = head, tail = tail, predicate = predicate)
    
    def _extract_mini_prediction(self, sequence):
        if 'pharmacokinetic' in sequence:
            return self._extract_relation_regex(r"pharmacokinetic relation between (.*) and (.*) is described", sequence, 'MECHANISM')
        elif 'effect' in sequence:
            return self._extract_relation_regex(r"effect of administering (.*) and (.*) in combination is described", sequence, 'EFFECT')
        elif 'advice' in sequence:
            return self._extract_relation_regex(r"advice regarding administering (.*) and (.*) in combination is described", sequence, 'ADVISE')
        elif 'interaction' in sequence:
            return self._extract_relation_regex(r"interaction between (.*) and (.*) was discussed, but additional information was not provided", sequence, 'INT')
    
    

#%%

class E2eReHardTemplate1(E2eReTemplate1):
    is_trainable = False
    
    instruction = "Find all instances of drug-drug relations in the following passage:"
    
    def make_source_sequence(self):
    
        return f'{self.instruction} \n {self.text} \n '
    
         
class E2eReSoftTemplate1(E2eReTemplate1):
    is_trainable = True
    
    num_tokens = 9
    prompt_tokens = [f'<prompt{i}>' for i in range(num_tokens)]
    prompt_string = ''.join(prompt_tokens)

    def make_source_sequence(self):
        
        return f'{self.text}{self.prompt_string}'
            

#%% NER template 
class NerTemplate1(MulticlassNerTemplate):
    
    class2text = {'BRAND': 'drug brands', 
                  'DRUG': 'drugs', 
                  'DRUG_N': 'nonmedical chemicals', 
                  'GROUP': 'groups of drugs'}
                                 
    text2class = QueryDict(reverse_dict(class2text))
                             
#%% 
class NerHardTemplate1(NerTemplate1):
    is_trainable = False
    
    instruction = "Find all drugs named in the following text: "
    
    def make_source_sequence(self):
    
        return f'{self.instruction} \n {self.text} \n '
    
         
class NerSoftTemplate1(NerTemplate1):
    is_trainable = True
    
    num_tokens = 9
    prompt_tokens = [f'<prompt{i}>' for i in range(num_tokens)]
    prompt_string = ''.join(prompt_tokens)

    def make_source_sequence(self):
        
        return f'{self.text}{self.prompt_string}'
            
