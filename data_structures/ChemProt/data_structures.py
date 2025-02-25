#%% preliminaries
from collections import defaultdict
import pandas as pd
import re
from dataclasses import dataclass
from statistics import mode

#from performance_utils import *

#%%
class Predicate():
    def __init__(self, predicate_short_string, predicate_long_string, predicate_id):
        self.short_string = predicate_short_string
        self.id = predicate_id
        self.long_string = predicate_long_string
        
        
class EntityType():
    def __init__(self, entity_type_string, entity_type_id):
        self.id = entity_type_id
        self.string = entity_type_string

class PredicateConverter():
    def __init__(self, predicates):
        
        self.predicates = predicates
        
        self._process()
        
    def _get_anyformat2predicate(self):
    
        anyformat2predicate = dict()
        
        for el in self.predicates:
            anyformat2predicate[el.id] = el
            anyformat2predicate[el.short_string] = el
            anyformat2predicate[el.long_string] = el
            
        self.anyformat2predicate = anyformat2predicate
    
    def _process(self):
        self._get_anyformat2predicate()
    
    def get(self, input_):
        return self.anyformat2predicate[input_]

    def convert(self, input_, desired_format):
        return getattr(self.anyformat2predicate[input_], desired_format)
    
    def __call__(self, input_, desired_format):
        return self.convert(input_, desired_format)
        
class EntityTypeConverter():
    def __init__(self, entity_types):
        
        self.entity_types = entity_types
        
        self._process()
        
    def _get_anyformat2entity_type(self):
    
        anyformat2entity_type = dict()
        
        for el in self.entity_types:
            anyformat2entity_type[el.id] = el
            anyformat2entity_type[el.string] = el
            
        self.anyformat2entity_type = anyformat2entity_type
    
    def _process(self):
        self._get_anyformat2entity_type()
    
    def get(self, input_):
        return self.anyformat2entity_type[input_]

    def convert(self, input_, desired_format):
        return getattr(self.anyformat2entity_type[input_], desired_format)    

#%% mentions
class Mention():
    def __init__(self, mentions_df_row, parent_example):
        self.row = mentions_df_row
        self.parent_example = parent_example

        self._process()    
    
    def _get_mention_id(self):
        self.mention_id = self.row.mention_id
    
# =============================================================================
#     def _get_type_string(self):
#         self.entity_type_string = self.mention_df.entity_type_string[0]
# =============================================================================
    
    def _get_type(self):
        self.entity_type = ChemProtDataset.entity_type_converter.get(self.row.entity_type_string)
        self.entity_type_string = self.entity_type.string
    def _get_string(self):
        # self.string = self.row.string
        self.string = self.row.string.lower()
    
    def _get_span(self):
        self.span = (self.row.start, self.row.end)
    
    def _process(self):
        
        self._get_mention_id()
# =============================================================================
#         self._get_type_string()
# =============================================================================
        self._get_type()
        self._get_string()
        self._get_span()
    
    @classmethod
    def check_candidate_veracity(cls, predicted_mention, true_mentions):
        for el in true_mentions:
            if predicted_mention['string'] == el.string:
                if predicted_mention['entity_type'] == el.entity_type.string:
                    return True
                
        else:
            return False

#%% relation class 
class Relation():
    def __init__(self, relations_df_row, parent_example):
        
        self.row = relations_df_row
        self.parent_example = parent_example
    
        self._process()
    
    def _get_pmid(self):
        self.pmid = self.row.pmid
    
    def _get_mentions(self):
        
        head_id = self.row.head_mention_id
        tail_id = self.row.tail_mention_id
        
        self.head = self.parent_example.mention_id2mention[head_id]
        self.tail = self.parent_example.mention_id2mention[tail_id]
    
    def _get_predicate(self):
        self.predicate = ChemProtDataset.predicate_converter.get(self.row.predicate_short_string)
        self.formal_predicate = self.predicate.short_string
        
    def _get_output_dict(self):
        predicate = self.predicate.short_string
        head = self.head.string
        tail = self.tail.string
        
        self.output_dict = dict(head = head, tail = tail, predicate = predicate)
    
    def output_head(self):
        return self.head.string
    
    
    def output_tail(self):
        return self.tail.string
    
    def output_predicate(self):
        return self.predicate.long_string
    
    def output_head_type(self):
        return self.head.entity_type.string
    
    def output_tail_type(self):
        return self.tail.entity_type.string
    
    def _process(self):
        self._get_pmid
        self._get_mentions()
        self._get_predicate()
        self._get_output_dict()
    
    def __eq__(self, other):
        if self.head.string == other.head.string:
            if self.tail.string == other.tail.string:
                if self.predicate.short_string == other.predicate.short_string:
                    return True
        return False

    def __hash__(self):
        return hash((self.head.string, self.tail.string, self.predicate.short_string))

    @classmethod
    def check_candidate_veracity(cls, candidate_relation, true_relations):
        for el in true_relations:
            if candidate_relation['head'] == el.head.string:
                if candidate_relation['tail'] == el.tail.string:
                    if candidate_relation['predicate'] == el.predicate.short_string:
                        return True
        else: 
            return False


#####


# @dataclass
#     class Triple:
#         head: str
#         tail: str
#         predicate: str


#     def __eq__(self, other):
#         return (self.head.lower() == other.head.lower() and
#                 self.tail.lower() == other.tail.lower() and 
#                 self.predicate.lower() == other.predicate.lower()
#                 )
    
#     def key(self):
#         return (self.head.lower(), self.tail.lower(), self.predicate.lower())
    
#     @classmethod
#     def group_equal_instances(cls, triples):
#         groups = {}
#         for instance in triples:
#             key = instance.key()
#             groups.setdefault(key, []).append(instance)
        
#         return list(groups.values())
        
#     @classmethod
#     def get_representative_instances(cls, triples_list):
#         def get_representative_instance(triples):
#             heads = [el.head for el in triples]
#             tails = [el.tail for el in triples]

#             head_mode = mode(heads)
#             tail_mode = mode(tails)
#             predicate = triples[0].predicate
            
#             return {'head': head_mode, 'tail': tail_mode, 'predicate': predicate}
    
#         unique_triples = [get_representative_instance(el) for el in triples_list]
#         return unique_triples



    
#     @classmethod
#     def from_relation(cls, relation, class2text):
#         head = relation.head.string
#         tail = relation.tail.string
#         predicate = class2text[relation.formal_predicate]

#         return Triple(head, tail, predicate)
    


    
#     def deduplicate(self): 
#         triples = [Triple.from_relation(el, self.class2text) for el in self.relations]
#         groups = Triple.group_equal_instances(triples)
        
    


    ########################3    
# =============================================================================
#     
#     @classmethod
#     def _dict2frozenset(cls, relation_dict):
#         for key, value in relation_dict:
#             relation_dict[key] = dict_list2frozenset_set(value)
#         return relation_dict
#     
#     @classmethod
#     def _convert_relations_to_dict(cls, relations_list, all_predicates):
#         relations_dict = {el: [] for el in all_predicates}
#         
#         for el in relations_dict:
#             relations_dict[el.predicate.short_string].append(el.output_dict)
#             
#         return relations_dict
#      
#     @classmethod
#     def _change_keys(cls, dictionary, key_map):
#         return {key_map.get(key, key): value for key, value in dictionary.items()}
#    
#     @classmethod
#     def _get_true_dict(cls, true_relations, template):
#         
#         all_predicates = [el.short_string for el in ChemProtDataset.predicates]
#         true_dict = cls._convert_relations_to_dict(true_relations, all_predicates)
#         true_dict = cls._change_keys(true_dict, template.relation_string_map)
#         
#         return true_dict
#         
#     @classmethod
#     def _get_performance_counts_example_workhorse(cls, true_dict, candidate_dict):
#         
#         true_dict = cls._dict2frozenset(true_dict)
#         candidate_dict = cls._dict2frozenset(candidate_dict)
#         
#         performance_dict = {el: set_performance_counts(true_dict[el], 
#                                                        candidate_dict[el]) for el in true_dicts.keys()}
#         
#         return performance_dict
#     
#     @classmethod
#     def _get_performance_counts_example(cls, true_relations, candidate_dict, template):
#         true_dict = cls._get_true_dict(true_relations, template.predicate_converter)
#         return true_dict    
#     
#     @classmethod
#     def get_performance(cls, batch_output_list, template):
#         epoch_output = unlist(batch_output_list)
#         
#         example_counts_list = [cls._get_performance_counts_example(el['true'], el['pred'], template) for el in epoch_output]
#         
#         performance_counts = multiclass_dataset_performance_counts(example_counts_list)
#         
#         return multiclass_performance(performance_counts) 
# =============================================================================

#%% example class
class ChemProtExample():
    def __init__(self, 
                 pmid, parent_dataset,
                 text_df, mentions_df, relations_df):
        
        self.pmid = pmid
        self.parent_dataset = parent_dataset
        self.text_df = text_df
        self.mentions_df = mentions_df
        self.relations_df = relations_df
        
        self.templates = {}
        
        self._process()
        
    ######## organizing information
    def _get_text(self):
        self.title_text = self.text_df.title.iloc[0]
        self.abstract_text = self.text_df.abstract.iloc[0]
        self.text = self.title_text + ' ' + self.abstract_text
        
    def _get_mentions(self):
        if self.mentions_df is None:
            self.mentions = []
        else:
            self.mentions = [Mention(self.mentions_df.iloc[i], self) for i in range(len(self.mentions_df))]
            
    def _get_mention_id2mention(self):
        
        self.mention_id2mention = dict()
        for el in self.mentions:
            self.mention_id2mention[el.mention_id] = el
        
    def _get_relations(self):
        if self.relations_df is None:
            self.relations = []
        else:
            self.relations = [Relation(self.relations_df.iloc[i], self) for i in range(len(self.relations_df))]
            self.relations = list(set(self.relations))
    ############ processing
    def _process(self):
        self._get_text()
        self._get_mentions()
        self._get_mention_id2mention()
        self._get_relations()
    
    def _templatify(self, template):
        self.templates[str(template)] = template(text = self.text,
                                                 relations = self.relations
                                                 )
    def _use_template(self, template, stage):
        template = self.templates[str(template)]
        
        return template.make_sequence(stage)
    
        

#%%
class ChemProtDataset():
    
    predicates = [Predicate('CPR:3', 'UPREGULATOR|ACTIVATOR|INDIRECT_UPREGULATOR', 0),
                  Predicate('CPR:4', 'DOWNREGULATOR|INHIBITOR|INDIRECT_DOWNREGULATOR', 1),
                  Predicate('CPR:5', 'AGONIST|AGONIST—ACTIVATOR|AGONIST—INHIBITOR', 2),
                  Predicate('CPR:6', 'ANTAGONIST', 3),
                  Predicate('CPR:9', 'SUBSTRATE|PRODUCT_OF|SUBSTRATE_PRODUCT_OF', 4),
                  ]

    predicate_converter = PredicateConverter(predicates)

    entity_types = [EntityType('CHEMICAL', 0),
                    EntityType('GENE', 1)]

    entity_type_converter = EntityTypeConverter(entity_types)

    
    
    def __init__(self, text_df, mentions_df, relations_df):
        self.text_df = text_df
        self.mentions_df = mentions_df
        self.relations_df = relations_df
        
        self.templates = {}
        
        self._process()
    
    ##### preprocessing
    def _remove_normalization_tags_from_entity_types(self):
        
        def remove(string):
            if string.endswith("-N") or string.endswith("-Y"):
                string = string[:-2]
            return string
        
        cleaned_entity_type_string = self.mentions_df.entity_type_string.apply(remove)        
        
        self.mentions_df.entity_type_string = cleaned_entity_type_string      
    
    def _remove_Arg_tags_from_relation_mentions(self):
        
        def remove(string):
            if string.startswith("Arg"):
                string = string[5:]
            return string
        
        cleaned_head_mention_id = self.relations_df.head_mention_id.apply(remove)        
        cleaned_tail_mention_id = self.relations_df.tail_mention_id.apply(remove)        
        
        self.relations_df.head_mention_id = cleaned_head_mention_id     
        self.relations_df.tail_mention_id = cleaned_tail_mention_id     
    ######
    
    def _get_pmid2df_defaultdict(self, dataframe):
        
        pmid2df = defaultdict(lambda : None)

        grouping = dataframe.groupby('pmid')

        for el_id, el_group in grouping:
            pmid2df[el_id] = dataframe.loc[el_group.index]
            
        return pmid2df
    
    def _get_examples(self):
        unique_pmids = self.text_df.pmid
        
        pmid2text_df = self._get_pmid2df_defaultdict(self.text_df)
        pmid2mentions_df = self._get_pmid2df_defaultdict(self.mentions_df)
        pmid2relations_df = self._get_pmid2df_defaultdict(self.relations_df)
        
        unique_pmids = self.text_df.pmid
        
        self.examples = [ChemProtExample(el, self,
                                          pmid2text_df[el], 
                                          pmid2mentions_df[el], 
                                          pmid2relations_df[el]) for el in unique_pmids]
        
        

            
    def _process(self):
        self._remove_normalization_tags_from_entity_types()
        self._remove_Arg_tags_from_relation_mentions()
        
        self._get_examples()
    
    
    def __add__(self, other):
        new_text_df = self.text_df + other.text_df
        new_mentions_df = self.mentions_df + other.mentions_df
        new_relations_df = self.relations_df + other.relations_df
                
        return ChemProtDataset(new_text_df, new_mentions_df, new_relations_df)
        
        
    def add_template(self, template):
        
        self.templates[str(template)] = template
        
        for el in self.examples:
            el._templatify(template)
    
    def use_template(self, template, stage):
        return [el._use_template(template, stage) for el in self.examples]
        

