#%% preliminaries
from collections import Counter
from bioc import biocxml, biocjson
from copy import deepcopy

from utils.utils import *

#%% mentions
class Mention():
    def __init__(self, annotation, parent_passage):
        self.annotation = annotation
        self.parent_passage = parent_passage
        self.parent_document = self.parent_passage.parent_example.document
        self._process()
        
    def _get_type_string(self):
        
        self.entity_type_string = self.annotation.infons['type']
        
    def _get_type_id(self):
        entity_type = self.parent_passage.parent_example.parent_collection.entity_type_string2id[self.entity_type_string]
        self.entity_type = entity_type
    
    def _get_string(self):
        self.string = self.annotation.text
        
    def _get_ids(self):
        self.ids = self.annotation.infons['MESH']
        
        if self.ids == '-':
            self.ids = None
        else:
            self.ids = self.ids.split("|")
    
    def _no_id(self):
        return self.ids is None
                            
    def _process(self):
        
        self._get_type_string()
        self._get_type_id()
        self._get_string()
        self._get_ids()
        
    
#%% title and abstract class
class Passage():
    def __init__(self, passage, parent_example):
        
        # basic initialization
        self.passage = passage
        self.parent_example = parent_example
                
        self._process()
        
    def _get_text(self):
        self.text = self.passage.text
    
    def _get_mentions(self):
        self.mentions = [Mention(el, self) for el in self.passage.annotations] 
    
    def _process(self):
        
        self._get_mentions()
        self._get_text()
        
#%% Entity class
class DocEntity():
    def __init__(self, entity_id, parent_example):
        
        self.id = entity_id
        self.parent_example = parent_example
        
        self._process()
    
    def _get_mentions(self):
        
        self.mentions = [el for el in self.parent_example.mentions if self.id in el.ids]
    
    def _get_type(self):
        self.type = self.mentions[0].entity_type
        
    def _get_type_string(self):
        self.type_string = self.mentions[0].entity_type_string
    
    def _get_strings(self):
        self.strings = [el.string for el in self.mentions if el.string]
        
    def _get_unique_strings(self):
        self.unique_strings = list(dict.fromkeys(self.strings)) 
    
    def longest(self):
        return max(self.strings)
    
    def shortest(self):
        return min(self.strings)
    
    def commonest(self):
        count = Counter(self.strings)
        
        return max(count, key = count.get)
    
    def first(self):
        return self.strings[0]
   
    def _process(self):
        self._get_mentions()
        self._get_type()
        self._get_type_string()
        self._get_strings()
        self._get_unique_strings()
        
#%% relation class 
class Relation():
    def __init__(self, document, parent_example):
        
        self.document = document
        self.parent_example = parent_example
    
        self._process()
        
    def _get_entities(self):
        infons = self.document.infons
        
        head_id = infons['Chemical']
        tail_id = infons['Disease']
        
        self.head = self.parent_example.id2entity[head_id]
        self.tail = self.parent_example.id2entity[tail_id]
    
    def _process(self):
        self._get_entities()
        
    #def output_head(self, string_chooser):
    def output_head(self, string_chooser = None):
        
        if string_chooser:
            return getattr(self.head, string_chooser)()
        else:
            return self.head.unique_strings()
        
    
    def output_tail(self, string_chooser):
        if string_chooser:
            return getattr(self.tail, string_chooser)()
        else:
            return self.tail.unique_strings()
    
    @classmethod
    def check_candidate_veracity(cls, candidate_relation, true_relations):
        for el in true_relations:
            if candidate_relation['head'] in el.head.strings:
                if candidate_relation['tail'] in el.tail.strings:
                    return True
        else: 
            return False
    
#%% example class
class CDRExample():
    def __init__(self, document,
                 parent_collection):
        
        self.document = document
        self.parent_collection = parent_collection
        self.templates = {}
        
        self._process()
        
    ######## organizing information
    def _get_pubmed_id(self):
        self.pubmed_id = self.document.id
    
    def _get_passages(self):
        
        passages = [Passage(el, self) for el in self.document.passages]
        
        passages[0].passage_type = 'title'
        passages[1].passage_type = 'abstract'
        
        self.passages = passages
    
    def _get_text(self):
        self.title_text = self.passages[0].text
        self.abstract_text = self.passages[1].text
        self.text = self.title_text + ' ' + self.abstract_text
        
    def _get_mentions(self):
        self.mentions = self.passages[0].mentions + self.passages[1].mentions
    
    def _get_example_entities(self):
        unique_ids = set(unlist([el.ids for el in self.mentions if el.ids is not None]))    
        self.entities = [DocEntity(el, self) for el in unique_ids]
        self.id2entity = dict(zip(unique_ids, self.entities))
        
    def _get_relations(self):
        self.relations = [Relation(el, self) for el in self.document.relations]
    
    ############ processing
    def _process(self):
        self._get_pubmed_id()
        self._get_passages()
        self._get_text()
        self._get_mentions()
        self._get_example_entities()
        self._get_relations()
    
    def _templatify(self, template):
        self.templates[str(template)] = template(text = self.text,
                                                 relations = self.relations
                                                 )
    def _use_template(self, template, stage):
        template = self.templates[str(template)]
        
        return template.make_sequence(stage)
    
        
        
#%% dataset class
class CDRDataset():
    def __init__(self, collection):
        self.collection = collection
        self.templates = {}
        
        self._process()
        
    def _get_docs(self):
                
        self.documents = self.collection.documents
        
    def _get_entity_type_converters(self):
        
        self.entity_type_string2id = {'Chemical': 0,
                                      'Disease': 1}
        
        self.entity_type2string = reverse_dict(self.entity_type_string2id)
    
    def _get_examples(self):
        self.examples = [CDRExample(el, self) for el in self.documents]
        
    def _process(self):
        self._get_docs()
        self._get_entity_type_converters()
        self._get_examples()
    
    
    def __add__(self, other):
        full_collection = deepcopy(self.collection)
        full_collection.documents += other.collection.documents
        
        return CDRDataset(full_collection)
        
        
    def add_template(self, template):
        
        self.templates[str(template)] = template
        
        for el in self.examples:
            el._templatify(template)
    
    def use_template(self, template, stage):
        return [el._use_template(template, stage) for el in self.examples]
        
