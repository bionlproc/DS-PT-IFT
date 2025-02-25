#%% libraries
import re

from utils.utils import *

#%% General Template

class Template():
    
    @classmethod
    def from_example(cls, example):
        return cls(example.text, getattr(example, cls.label_name))

    def make_sequence(self):
        
        return f'{self.make_source_sequence()}{self.make_target_sequence()}'
    
    def _strip_period(self, text):
        if len(text) > 0:
            if text[-1] == '.':
                text = text[:-1]
        return text
    
    def _separate_by_semicolon(self, text):
        return text.split('; ')
    
    def _separate_by_period(self, text):
        return text.split('. ')
    
    def _separate_list(self, string):
        elements = re.split(', and |, | and ', string)
        elements = [el.lstrip().rstrip() for el in elements]    
        return elements
    
    def _is_match_permissable(self, offsets, text):
        left_idx = offsets[0] - 1
        right_idx = offsets[1]
        
        if left_idx >= 0:
            if text[left_idx].isalpha():
                return False
        
        if right_idx < len(text):
            if text[right_idx].isalpha():
                return False
        
        return True        
        
    
    def _find_all_substring_offsets(self, substring, text):
        substring = substring.lower()
        text = text.lower()
        
        substring_length = len(substring)
        offsets = []
        start = 0
        while True:
            start = text.find(substring, start)
            if start == -1:
                break
            
            offsets_temp = (start, start + substring_length)
            if self._is_match_permissable(offsets_temp, text):
                offsets.append(offsets_temp)
            start += 1
        return offsets    
    
    def _remove_prefix_newlines(self, string):
        string = string.lstrip()
        has_newline = True # just to get while loop started
        while has_newline:
            if string[:2] == '\n':
                string = string[2:]
                continue
            if string[:3] == '\n':
                string = string[3:]
                continue
            has_newline = False
        string = string.lstrip().rstrip()
        return string
    
    def _unique_strings(self, strings, case_sensitive = False):
        if case_sensitive:
            unique = set(strings)
        else:
            unique = set()
            for el in strings:
                if el.lower() not in set(el_uniq.lower() for el_uniq in unique):
                    unique.add(el)
        return unique

    def _choose_string(self, strings, criterion):
        if criterion == 'longest':
            return max(strings, key = len)
        else:
            ValueError('Invalid value for criterion argument')


            
        
#%% NER template 
class UniclassNerTemplate(Template):
    task = 'NER'
    '''
    A hard template for this class should have a source sequence that ends in 
    something like " \n The drugs mentioned in this passage are: "
    '''
    
    def __init__(self, text, mentions):
       
        self.text = text
        self.mentions = mentions

    @classmethod
    def from_example(cls, example):
        return cls(example.text, example.mentions)

    def make_target_sequence(self):
                
        mentions = [el.string for el in self.mentions]
        mentions = self._unique_strings(mentions)
        target = comma_separated_string2(mentions)
        
        return target
        
    def _extract_entities(self):
        
        entities = separate_list(self.generated_sequence)
        entities = [self._remove_prefix_newlines(el) for el in entities]
        entities = frozenset(entities)
        return entities
    
    def _extract_mentions_from_entity(self, entity):
        
        return self._find_all_substring_offsets(entity, self.text)
        
    def extract_prediction(self):
        predicted_entities = self._extract_entities()
        #print('predicted entities: ', predicted_entities)
        predicted_mentions = [self._extract_mentions_from_entity(el) for el in predicted_entities]
        #print('predicted mentions: ', predicted_mentions)
        predicted_mentions = unlist(predicted_mentions)
        #print('unlisted predicted mentions: ', predicted_mentions)
        
        #predicted_mentions = unique_dicts(predicted_mentions) # is this necessary?
        # print('predicted_mentions: ', predicted_mentions)
        return predicted_mentions
    
#%% Soft uniclass NER Template
class SoftUniclassNerTemplate1(UniclassNerTemplate):
    is_trainable = True
    
    num_tokens = 9
    prompt_tokens = [f'<prompt{i}>' for i in range(num_tokens)]
    
    def make_source_sequence(self):
        
        return f'{self.text}{self.prompt_tokens}'
                
#%% General multiclass NER Template
class MulticlassNerTemplate(Template):
    task = 'NER'
    ''' This template requires a `class2text` from the name of 
    the entity type according to the dataset and the plural form that
    you wish to be a part of the `prefix` in the template. Example: 
    in the ChemProt dataset, the entity type strings are 'CHEMICAL'
    and 'GENE'.  We want them to be in the prefix as 'chemicals' 
    and 'genes', so our mapping would be 
    
    `class2text = {'CHEMICAL': 'chemicals',
                   'GENE': 'genes'}`
    '''
    
    def __init__(self, text, mentions):
        
        self.text = text
        #print(self.text)
        self.mentions = mentions
    
    @classmethod
    def from_example(cls, example):
        return cls(example.text, example.mentions)

    def _make_entity_type_target(self, mentions, entity_type_string):
        
        mentions_string = comma_separated_string2(mentions)
        
        prefix = f'the {entity_type_string} in the passage are'
        
        if mentions_string == '':
            return None
        else:
            return f"{prefix} {mentions_string}"
        
    def make_target_sequence(self):
        
        mention_strings_by_entity_type = {value: self._unique_strings([el.string for el in self.mentions if el.entity_type_string == key]) for key, value in self.class2text.items()}
        
        target_by_entity_type = {el_type: self._make_entity_type_target(el_mens, el_type) for el_type, el_mens in mention_strings_by_entity_type.items() if el_mens}
        
        target = '; '.join(target_by_entity_type.values())
        target += '.'
        
        #target[0] = target[0].upper()

        return target
    
    def _separate_entity_types(self, text):
        return self._separate_by_semicolon(text)
    
    def _extract_entities_within_type(self, string):
        
        entities = separate_list(string)
        entities = set(entities)
        
        return entities
        
    def _extract_entities_of_type(self, sequence):
        
        match = re.search(r"[Tt]he (.*) in the passage are (.*)", sequence)
        
        if match and len(match.groups()) == 2:
            entity_type = match.group(1)
            entity_type = self.text2class[entity_type]
            if entity_type in self.class2text.keys():
                entities_string = match.group(2)
                entities = self._extract_entities_within_type(entities_string)
                
                return {entity_type: entities}
            

    def _extract_entities(self):
        # separates entity types
        generated_sequence = self._strip_period(self.generated_sequence)
        
        generated_sequences = self._separate_entity_types(generated_sequence)
        
        #
        predicted_entities = {}
        for el in generated_sequences:
            type_entities = self._extract_entities_of_type(el)
            if type_entities:
                predicted_entities.update(type_entities)
    
        return predicted_entities
    
    def _extract_mentions_from_entity(self, entity_string, entity_type):
        
        offsets = self._find_all_substring_offsets(entity_string, self.text)

        mentions = [{'offsets': el, 'entity_type': entity_type} for el in offsets]
        
        return mentions
    
    def _extract_mentions_from_entities_of_type(self, entities, entity_type):
        print('extract mentions from entities of type')
        print('entities: ', entities)
        mentions = []
        for el in entities:
            print('el: ', el)
            mentions_temp = self._extract_mentions_from_entity(el, entity_type)
            if mentions_temp:
                mentions += mentions_temp
        
        return mentions
        
    def _extract_mentions_from_entities(self, entity_dict):
        
        mentions = [self._extract_mentions_from_entities_of_type(value_entities, key_type) for key_type, value_entities in entity_dict.items()]
        mentions = unlist(mentions)
        return mentions
        
    def extract_prediction(self):
        
        predicted_entities = self._extract_entities()
        print('predicted entities: ', predicted_entities)

        predicted_mentions = self._extract_mentions_from_entities(predicted_entities)
        print('predicted mentions: ', predicted_mentions)

        predicted_mentions = unique_dicts(predicted_mentions) # is this necessary?

        return predicted_mentions
        
#%%
class SoftMulticlassNerTemplate1(MulticlassNerTemplate):
    is_trainable = True
    
    num_tokens = 9
    prompt_tokens = [f'<prompt{i}>' for i in range(num_tokens)]
    prompt_string = ''.join(prompt_tokens)
    
    def make_source_sequence(self):
        
        return f'{self.text}{self.prompt_string}'

#%%
class E2eReTemplate(Template):
    task = 'E2ERE'
    '''
    Need to assign the attribute `null_target` that is the target if there are no relations.
    '''
    
    def __init__(self, text, relations):
       
        self.relations = relations
        self.text = text
                        
    def _make_relation_target(self):
        '''
        This method gets defined for each fully realized template.  It should 
        take in the components of a relation, in the ideal format (i.e., 
        strings, lists of strings, etc.), and output a string of the target 
        sequence for a single relation.
        '''
        pass

    def _make_mini_target(self, sequence):
        '''
        This method gets defined for each template.  It prepares a relation 
        to be in the correct format for self._make_mini_targets.
        '''
        pass
    
    def _make_no_rel_target(self):
        '''
        This method gets defined for each template.  It is the sequence that
        should appear if there are no relations.
        '''
        pass
    
    def make_target_sequence(self):


        if len(self.relations) > 0:
            mini_targets = [self._make_mini_target(el) for el in self.relations]
        
            target = f"{'; '.join(mini_targets)}."
        else:
            target = self.null_target
        
        return target   
    
    def _separate_relations(self, text):
        #print('splitting multiple relations (if multiple)')
        return text.split(';')
    
    def _extract_mini_prediction(self, string):

        '''
        This method gets defined for each template.  It is the workhorse 
        method that extract a relation from a text containing a predicted 
        relation.  It takes as input a string, and output a dictionary of the 
        necessary information to define a relation.
        '''
        pass

    def extract_prediction(self):
        #print(text, type(text))
        
        # print(f'generated_sequence: {self.generated_sequence}')

        text = self._strip_period(self.generated_sequence)
        
        # print('period stripped: ', text)
        
        #print('splitting multiple relations, if possible')
        mini_sequences = self._separate_relations(text)
        
        # print(f'split_sequence: {text}')

        #print('extracting predicted relations')
        predicted_relations = []
        for el in mini_sequences:
            relation_temp = self._extract_mini_prediction(el)
            if relation_temp:
                predicted_relations.append(relation_temp)
        
        predicted_relations = unique_dicts(predicted_relations)
        
        # print(f'preds: {predicted_relations}')
        # print(len(predicted_relations))

        return predicted_relations

#%% Direct NER
# =============================================================================
# class DirectUniclassNerTemplate(Template):
#     
#     start_char = '*'
#     end_char = '$'
#     
#     def __init__(self, text, mentions):
#         self.text = text
#         self.mentions = mentions
#     
#     def make_target_sequence(self):
#         
#         displacement = 0
#         for el in self.mentions:
#             
#             prior_text = self.text[:el.offsets[0]]
#             post_text = self.text[el.offsets[1]:]
#             mention_text = self.text[el.offsets[0]:el.offsets[1]]
#             
#             text = f'{prior_text}{self.start_char}{self.mention_text}{self.end_char}{post_text}'
#             
#             displacement += 1
#             el.offsets[0] += displacement
#             displacement += 1
#             el.offsets[1] += displacement
#         
#         return text
# =============================================================================
    
  
        