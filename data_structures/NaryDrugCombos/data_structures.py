#%% libraries

#%%
class Mention():
    def __init__(self, annotation):
        self.annotation = annotation
        
        self._process()
        
    def _get_string(self):
        self.string = self.annotation['text']
        
    def _get_offsets(self):
        self.offsets = (self.annotation['start'], self.annotation['end'])
        
    def _process(self):
        self._get_string()
        self._get_offsets()
        
    @classmethod
    def check_candidate_string_veracity(cls, predicted_mention_string, true_mentions):
        return predicted_mention_string in [el.string for el in true_mentions]
    
    @classmethod
    def check_candidate_veracity(cls, predicted_mention_offsets, true_mentions):
        return predicted_mention_offsets in [el.offsets for el in true_mentions]

#%% predicate
# =============================================================================
# class Predicate():
#     def __init__(self, string, predicate_id):
#         self.string = string
#         self.id = predicate_id
# 
# class PredicateConverter():
#     def __init__(self, predicates):
#         
#         self.predicates = predicates
#         
#         self._process()
#         
#     def _get_anyformat2predicate(self):
#     
#         anyformat2predicate = dict()
#         
#         for el in self.predicates:
#             anyformat2predicate[el.id] = el
#             anyformat2predicate[el.string] = el
#             
#         self.anyformat2predicate = anyformat2predicate
#     
#     def _process(self):
#         self._get_anyformat2predicate()
#     
#     def get(self, input_):
#         return self.anyformat2predicate[input_]
# 
#     def convert(self, input_, desired_format):
#         return getattr(self.anyformat2predicate[input_], desired_format)
#     
#     def __call__(self, input_, desired_format):
#         return self.convert(input_, desired_format)
# =============================================================================
    
#%%
class Relation():
    def __init__(self, annotation, parent_example):
        self.annotation = annotation
        self.parent_example = parent_example
        
        self._process()
        
    def _get_mentions(self):
        self.mentions = [self.parent_example.mentions[el] for el in self.annotation['spans']]
    
    def _get_predicate(self):
        self.original_predicate = self.annotation['class']
        
        if self.original_predicate == 'NEG':
            self.predicate = 'COMB'
        else:
            self.predicate = self.original_predicate
        
        self.formal_predicate = self.predicate
            
    def _get_is_context_needed(self):
        self.is_context_needed = self.annotation['is_context_needed']
    
    def _get_mention_strings(self):
        return set(el_men.string for el_men in self.mentions)
    
    def _process(self):
        self._get_mentions()
        self._get_predicate()
        self._get_is_context_needed()
        
    @classmethod
    def check_candidate_veracity(cls, predicted_relation, true_relations):
        for el_rel in true_relations:
            if predicted_relation['mentions'] == el_rel._get_mention_strings():
                if predicted_relation['predicate'] == el_rel.predicate:
                    return True
        else:
            return False
    
    @classmethod
    def _jaccard_index(cls, set1, set2):
        return (set1 & set2)/(set1 | set2)
    
    @classmethod
    def get_highest_overlap(cls, predicted_relation, true_relations):
        
        jaccards = [cls._jaccard_index(predicted_relation['drugs'], 
                                       el_rel._get_mention_strings()
                                       ) for el_rel in true_relations]
        
        return max(jaccards)
            
#%% example
class Example():
    def __init__(self, annotation):
        self.annotation = annotation
    
        self._process()   
    
    def _get_text(self):
        self.text = self.annotation['sentence']
        self.context = self.annotation['paragraph']
        self.text = {'text': self.text, 'context': self.context}
        
    def _get_mentions(self):
        self.mentions = [Mention(el) for el in self.annotation['spans']]
        
    def _get_relations(self):
        self.relations = frozenset(Relation(el, self) for el in self.annotation['rels'])

    def _process(self):
        self._get_text()
        self._get_mentions()
        self._get_relations()
        
#%% Dataset
class NaryDrugCombosDataset():
    
# =============================================================================
#     predicates = [Predicate('POS', 0),
#                   Predicate('COMB', 1),
#                   Predicate('NEG',  2),
#                   ]
# 
#     predicate_converter = PredicateConverter(predicates)
# =============================================================================
    
    predicate_converter = {'POS': 'positive'}

    def __init__(self, annotation):
        self.annotation = annotation
        
        self._process()
        
    def _get_examples(self):
        self.examples = [Example(el) for el in self.annotation]
        print('num examples: ', len(self.examples))
        #self.examples = [el for el in self.examples if len(el.text['context']) < 2000]
        print('num examples: ', len(self.examples))
    def _process(self):
        self._get_examples()
    
