#%% libraries

#%%
class Mention():
    def __init__(self, annotation):
        self.annotation = annotation
        
        self._process()
    
    def _get_id(self):
        self.id = self.annotation['id']
    
    def _get_string(self):
        self.string = self.annotation['text']
    
    def _get_type(self):
        self.type = self.annotation['type']
        self.entity_type_string = self.type
    def _get_offsets(self):
        self.offsets = tuple(self.annotation['offsets'])
        
    def _process(self):
        self._get_id()
        self._get_string()
        self._get_type()
        self._get_offsets()
        
    @classmethod
    def check_candidate_string_veracity(cls, predicted_mention_string, true_mentions):
        return predicted_mention_string in [el.string for el in true_mentions]
    
    @classmethod
    def check_candidate_veracity(cls, predicted_mention, true_mentions):
        print('predicted_mention: ', predicted_mention)
        for el_true in true_mentions:
            if el_true.offsets == predicted_mention['offsets']:
                if el_true.entity_type_string == predicted_mention['entity_type']:
                    return True
        else:
            return False
        
  
#%%
class Relation():
    def __init__(self, annotation, parent_example):
        self.annotation = annotation
        self.parent_example = parent_example
        
        self._process()
     
    def _get_head(self):
        mention_id = self.annotation['head']['ref_id']
        self.head = self.parent_example.mention_converter[mention_id]
    
    def _get_tail(self):
        mention_id = self.annotation['tail']['ref_id']
        self.tail = self.parent_example.mention_converter[mention_id]
    
    def _get_predicate(self):
        self.predicate = self.annotation['type']
        self.formal_predicate = self.predicate
            
    def _process(self):
        self._get_head()
        self._get_tail()
        self._get_predicate()
        
    @classmethod
    def check_candidate_veracity(cls, predicted_relation, true_relations):
        for el_rel in true_relations:
            if predicted_relation['head'] == el_rel.head.string:
                if predicted_relation['tail'] == el_rel.tail.string:
                    if predicted_relation['predicate'] == el_rel.predicate:
                        return True
        else:
            return False
        
    
#%% example
class Example():
    def __init__(self, annotation):
        self.annotation = annotation
    
        self._process()   
    
    def _get_id(self):
        self.id = self.annotation['document_id']
    
    def _get_text(self):
        self.text = self.annotation['text']
    
    def _get_mentions(self):
        self.mentions = frozenset(Mention(el) for el in self.annotation['entities'])
    
    def _get_mention_converter(self):
        ids = [el.id for el in self.mentions]
        self.mention_converter = dict(zip(ids, self.mentions))
    
    def _get_relations(self):
        self.relations = frozenset(Relation(el, self) for el in self.annotation['relations'])

    def _process(self):
        self._get_id()
        self._get_text()
        self._get_mentions()
        self._get_mention_converter()
        self._get_relations()
    
    
#%% Dataset
class DdiDataset():
    def __init__(self, annotation):
        self.annotation = annotation
        
        self._process()
        
    def _get_examples(self):
        self.examples = [Example(el) for el in self.annotation]
        
        #self.examples = [el for el in self.examples if len(el.text) < 4500]
    def _process(self):
        self._get_examples()
        