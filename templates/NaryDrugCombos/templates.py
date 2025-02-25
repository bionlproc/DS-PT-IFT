#%% To Do

#%% libraries
from data_structures.NaryDrugCombos.data_structures import *
from templates.general_templates import *
from utils.utils import *

#%% templates
class E2eReTemplate1(E2eReTemplate):
    dataset_name = 'NaryDrugCombos'
    label_name = 'relations'

    null_target = 'No drug combinations are established.'
    
    class2text = {'POS': 'positive',
                  'COMB': ''}
    
    def __init__(self, text, relations):
       
        self.relations = relations
        self.text = text['text']
        self.context = text['context']
    
    @classmethod
    def from_example(cls, example):
        return cls(example.text, example.relations)

    def _make_relation_target(self, mention_strings, predicate_string):
                
        return f'{mention_strings} participate in a {self.class2text[predicate_string]} drug combination'
        
    def _make_mini_target(self, relation):
                
        mentions = set(el.string for el in relation.mentions)
        mention_strings = comma_separated_string(mentions)
                        
        predicate_string = relation.predicate
                
        target = self._make_relation_target(mention_strings, predicate_string) 
        
        return target
    
    def _remove_whitespace_around_hyphens(self, mentions):
        
        def workhorse(mention):
            return mention.replace(' - ', '-')

        return [workhorse(el) for el in mentions]


    def _extract_mini_prediction(self, sequence):
        
        splits = sequence.split('participate')
        if len(splits) == 2:
            mentions_string, label_string = splits
            mentions = re.split(', and |, | and ', mentions_string)
            mentions = [el.lstrip().rstrip() for el in mentions]
            mentions = self._remove_whitespace_around_hyphens(mentions)
            mentions = frozenset(mentions)
            


            if 'positive' in label_string:
                predicate = 'POS'
            else:
                predicate = 'COMB'
                
            return dict(mentions = mentions, predicate = predicate)




            
class E2eReTemplate2(E2eReTemplate):
    dataset_name = 'NaryDrugCombos'
    label_name = 'relations'
    
    null_target = 'No drug interactions are established.'
    
    def __init__(self, text, relations):
       
        self.relations = relations
        self.text = text['text']
        self.context = text['context']
    
    @classmethod
    def from_example(cls, example):
        return cls(example.text, example.relations)

    def _make_relation_target(self, mention_strings, predicate_string):
        
        if predicate_string == 'POS':
            return f'{mention_strings} form an effective drug combination'
        elif predicate_string == 'COMB':
            return f'{mention_strings} were administered in combination'
                
    def _make_mini_target(self, relation):
                
        mentions = set(el.string for el in relation.mentions)
        mention_strings = comma_separated_string(mentions)
                        
        predicate_string = relation.predicate
                
        target = self._make_relation_target(mention_strings, predicate_string) 
        
        return target
    
    def _remove_whitespace_around_hyphens(self, mentions):
        
        def workhorse(mention):
            return mention.replace(' - ', '-')

        return [workhorse(el) for el in mentions]

    def _extract_mini_prediction(self, sequence):
        
        if 'effective' in sequence:
            splits = sequence.split(' form')
            if len(splits) == 2:
                mentions_string, _ = splits
                mentions = self._separate_list(mentions_string)
                mentions = self._remove_whitespace_around_hyphens(mentions)
                mentions = frozenset(mentions)
                
                predicate = 'POS'
        
                return dict(mentions = mentions, predicate = predicate)
        
        elif 'administered' in sequence:
            splits = ' were'
            if len(splits) == 2:
                mentions_string, _ = splits
                mentions = self._separate_list(mentions_string)
                mentions = self._remove_whitespace_around_hyphens(mentions)
                mentions = frozenset(mentions)
                
                predicate = 'COMB'
            
                return dict(mentions = mentions, predicate = predicate)            
        
                
#%%
class E2eReHardTemplate1(E2eReTemplate1):
    is_trainable = False
    
    instruction = "In the following sentence which appears in the following passage, find all instances of drugs used in combination and whether this combination had a positive impact on health"
    
    def make_source_sequence(self):
    
        return f'{self.instruction} \n Sentence: {self.text} \n Passage: {self.context} \n '

class E2eReHardTemplate1_sentence(E2eReTemplate1):
    is_trainable = False
    
    instruction = "Find all instances of drugs used in combination and whether this combination had a positive impact on health in the following sentence:\n\n"
    
    def make_source_sequence(self):
    
        return f'{self.instruction}\n\n{self.text}\n\n'

class E2eReHardTemplate2_sentence(E2eReTemplate2):
    is_trainable = False
    
    instruction = "Find all instances of drugs used in combination and whether this combination had a positive impact on health in the following sentence:\n\n"
    
    def make_source_sequence(self):
    
        return f'{self.instruction}\n\n{self.text}\n\n'
         
class E2eReSoftTemplate1(E2eReTemplate1):
    is_trainable = True
    
    num_tokens1 = 9
    num_tokens2 = 9
    prompt_tokens1 = [f'<prompt{i}>' for i in range(num_tokens1)]
    prompt_tokens2 = [f'<prompt{i}>' for i in range(num_tokens1, num_tokens2)]
    
    prompt_tokens = prompt_tokens1 + prompt_tokens2
    
    prompt_string1 = ''.join(prompt_tokens1)
    prompt_string2 = ''.join(prompt_tokens2)
    
    def make_source_sequence(self):
        
        return f'{self.text}{self.prompt_string1}{self.context}{self.prompt_string2}'

class E2eReHardTemplate2a(E2eReTemplate2):
    is_trainable = False
    
    instruction = "Find all instances of drugs used in combination and whether this combination had a positive impact on health in the following sentence, which appears in the ensuing passage:"
    
    def make_source_sequence(self):
    
        return f'{self.instruction} \n Sentence: {self.text} \n Passage: {self.context} \n '

class E2eReHardTemplate2b(E2eReTemplate2):
    is_trainable = False
    
    instruction = "Find all instances of drugs used in combination and whether this combination had a positive impact on health in the sentence enclosed in '$':"
    
    def make_source_sequence(self):
        
        augmented_context = self.context.replace(self.text, f'${self.text}$')
        
        return f'{self.instruction} \n {augmented_context} \n '

class E2eReSoftTemplate2(E2eReTemplate2):
    is_trainable = True
    
    num_tokens1 = 9
    num_tokens2 = 9
    prompt_tokens1 = [f'<prompt{i}>' for i in range(num_tokens1)]
    prompt_tokens2 = [f'<prompt{i}>' for i in range(num_tokens1, num_tokens2)]
    
    prompt_tokens = prompt_tokens1 + prompt_tokens2
    
    prompt_string1 = ''.join(prompt_tokens1)
    prompt_string2 = ''.join(prompt_tokens2)

    
    def make_source_sequence(self):
        
        return f'{self.text}{self.prompt_string1}{self.context}{self.prompt_string2}'
    
#%%
#class NerTemplate1():
#    
#        
#    def __init__(self, text, mentions):
#       
#        self.text = text['text']
#        self.context = text['context']
#        self.mentions = mentions
#        
#    def make_target_sequence(self):
#                
#        mentions = set(el.string for el in self.mentions)
#        target = comma_separated_string2(mentions)
#        
#        return target
#        
#    def make_sequence(self):
#        
#        return f'{self.make_source_sequence()}{self.make_target_sequence()}'
#    
#    @classmethod    
#    def _extract_prediction(cls, string):
#        
#        mentions = separate_list(string)
#        mentions = frozenset(mentions)
#        
#        return mentions
    
    
#%% 
class NerHardTemplate1(UniclassNerTemplate):
    is_trainable = False
    
    instruction = "Find all drugs named in the following text:"
    
    def __init__(self, text, mentions):
       
        self.text = text['text']
        self.mentions = mentions
    
    def make_source_sequence(self):
    
        return f'{self.instruction} \n {self.text} \n '
        
class NerSoftTemplate1(UniclassNerTemplate):
    is_trainable = True
    
    num_tokens = 9
    prompt_tokens = [f'<prompt{i}>' for i in range(num_tokens)]
    prompt_string = ''.join(prompt_tokens)

    def __init__(self, text, mentions):
       
        self.text = text['text']
        self.mentions = mentions
    
    def make_source_sequence(self):
        
        return f'{self.text}{self.prompt_string}'
     
  