#%% RE template 
import re
#print('importing spacy')
import spacy
#print('importing scispacy')
import scispacy

from templates.general_templates import *
from data_structures.ChemProt.data_structures import *
from utils.utils import *

spacy_model = spacy.load('en_core_sci_lg')
spacy_model.add_pipe("sentencizer")

#%%
class E2eReTemplate1(E2eReTemplate):
    dataset_name = 'ChemProt'
    label_name = 'relations'

    null_target = 'No drug-gene relationships are established.'
    
    #predicate_map = {'upregulator': 'CPR:3',
    #                 'activator': 'CPR:3',
    #                 'indirect upregulator': 'CPR:3',
    #                 'upregulator, activator, or indirect upregulator': 'CPR:3',
    #                 
    #                 'downregulator': 'CPR:4',
    #                 'inhibitor': 'CPR:4',
    #                 'indirect downregulator': 'CPR:4',
    #                 'downregulator, inhibitor, or indirect downregulator': 'CPR:4',
    #                 
    #                 'agonist': 'CPR:5',
    #                 'agonist—activator': 'CPR:5',
    #                 'agonist—inhibitor': 'CPR:5',
    #                 'agonist, agonist—activator, agonist—inhibitor': 'CPR:5',
    #                 
    #                 'antagonist': 'CPR:6',
    #                 
    #                 'substrate': 'CPR:9',
    #                 'product': 'CPR:9',
    #                 'substrate product of': 'CPR:9'
    #                 }
                    
    class2text = {'CPR:3': 'upregulation',
                  'CPR:4': 'downregulation',
                  'CPR:5': 'agonist',
                  'CPR:6': 'antagonist',
                  'CPR:9': 'substrate and/or product'
                  }
                        
    text2class = QueryDict({'upregulation': 'CPR:3',
                            'downregulation': 'CPR:4',
                            'agonist': 'CPR:5',
                            'antagonist': 'CPR:6',
                            'substrate and/or product': 'CPR:9',
                            'substrate': 'CPR:9',
                            'product': 'CPR:9'
                            })
    def __init__(self, text, relations):
       
        self.text = text
        self.relations = relations   
        
    @classmethod
    def from_example(cls, example):
        return cls(example.text, example.relations)

    def _make_predicate_string(self, predicate):
        print('predicate: ', predicate)
        print('type: ', type(predicate))
        print(dir(predicate))
        
        return self.class2text[predicate.short_string]
             
    def _make_relation_target(self, head, tail, predicate):
        
        #predicate_string = self._make_predicate_string(predicate)
        
        target = f'the relation between {head} and {tail} is {predicate}'
        
        return target
    
    def _make_mini_target(self, relation):
        
        head = relation.head.string
        tail = relation.tail.string
        #predicate = self._make_predicate_string(relation.predicate)
        predicate = self.class2text[relation.formal_predicate]
    
        return self._make_relation_target(head, tail, predicate)
        
    
    def _extract_mini_prediction(cls, string):
        # print('regex')
        match = re.search(r"relation between (.*) and (.*) is (.*)", string)
        
        # print('extract head and tail')
        if match:
            head = match.group(1)
            tail = match.group(2)
            predicate = match.group(3)
            formal_predicate = cls.text2class[predicate]
            # print('predicate: ', predicate)
            # print('CPR: ', formal_predicate)
            
            return dict(head = head, tail = tail, predicate = formal_predicate)
            
    def _is_intrasentence_relation(self, relation):
        head = relation['head']
        tail = relation['tail']
        
        doc = spacy_model(self.text)
        sentences = [sent.text for sent in doc.sents]
        for el in sentences:
            if head in el and tail in el:
                return True
        else: 
            return False
        
    def _extract_prediction(self):        
       predicted_relations = super()._extract_prediction()
       
       predicted_relations = [self._is_intrasentence_relation(el) for el in predicted_relations]
       
       return predicted_relations
   
#%%

class E2eReHardTemplate1(E2eReTemplate1):
    is_trainable = False
    
    instruction = "Find all instances of relations between chemicals and proteins in the following passage:"
    
    def make_source_sequence(self):
    
        return f'{self.instruction}\n\n{self.text}\n\n'
    
         
class E2eReSoftTemplate1(E2eReTemplate1):
    is_trainable = True
    
    num_tokens = 9
    prompt_tokens = [f'<prompt{i}>' for i in range(num_tokens)] 
    prompt_string = ''.join(prompt_tokens)

    def make_source_sequence(self):
        
        return f'{self.text}{self.prompt_string}'
            

#%% NER template 
class NerTemplate1(MulticlassNerTemplate):
                
    class2text = {'CHEMICAL': 'chemicals',
                  'GENE': 'genes'}
                            
    text2class = QueryDict(reverse_dict(class2text))
                  
    
class NerHardTemplate1(NerTemplate1):
    is_trainable = False
    
    instruction = "Find all chemicals and genes discussed in the following passage:" + ' \n '
    
    def make_source_sequence(self):
    
        return f'{self.instruction}{self.text} \n '
    
class NerSoftTemplate1(NerTemplate1):
    is_trainable = True
    num_tokens = 9
    prompt_tokens = [f'<prompt{i}>' for i in range(num_tokens)] 
    prompt_string = ''.join(prompt_tokens)

    def make_source_sequence(self):
        
        return f'{self.text}{self.prompt_string}'    





















            
#%% RE-only template
class HardReTemplate1(E2eReTemplate1):
    # RE template: <passage>\n\n The entities w,x,y, and z are mentioned in the passage.  Which pairs of entities exhibit relations, and what are the relations?\n\n
    
    is_trainable = False
    
    instruction = lambda chemicals_string, proteins_string: f"Find all instances of relations between chemicals {chemicals_string} and the proteins {proteins_string} in the following passage:"
    
    
    def __init__(self, text, mentions, relations):
        
        self.text = text
        self.mentions = mentions
        self.relations = relations
    
        
    def _make_entity_type_string(self, entity_type):
        strings_list = [el.string for el in self.mentions if el.entity_type == entity_type]
        string = comma_separated_string2(chemicals)
        
        return string
        
    def _make_filled_instruction(self):
        self.filled_instruction = self.instruction(self._make_entity_type_string('CHEMICAL'),
                                                   self._make_entity_type_string('GENE'))
         
    def make_source_sequence(self):
    
        return f'{self.filled_instruction} \n {self.text} \n '
    
         
class SoftReTemplate1(E2eReTemplate1):
    is_trainable = True
    
    num_tokens = 9
    prompt_tokens1 = [f'<prompt{i}>' for i in range(num_tokens)] 
    prompt_tokens = [f'<prompt{i}>' for i in range(num_tokens)] 
    prompt_tokens = [f'<prompt{i}>' for i in range(num_tokens)] 
    
    def __init__(self, text, mentions, relations):
        
        self.text = text
        self.mentions = mentions
        self.relations = relations
    
        
    def _make_entity_type_string(self, entity_type):
        strings_list = [el.string for el in self.mentions if el.entity_type == entity_type]
        string = ','.join(strings_list)
        
        return string
    
    def make_source_sequence(self):
        
        return f"{self.text}{self.prompt_tokens1}{self._make_entity_type_string('CHEMICAL')}{self.prompt_tokens2}{self._make_entity_type_string('GENE')}{self.prompt_tokens3}"
         
    
    
    
    
    
    
    