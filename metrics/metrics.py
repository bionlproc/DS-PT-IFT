#%% libraries
from collections import defaultdict


#%% metrics

class Metric:
    def _compute_precision(self):
        if self.TP + self.FP == 0:
            return 0
        else:
            return self.TP/(self.TP + self.FP)
    
    def _compute_recall(self):
        if self.TP + self.FN == 0:
            return 0
        else:
            return self.TP/(self.TP + self.FN)
    
    def _compute_F1(self):
        if self.TP + self.FP + self.FN == 0:
            return 0
        else:
            return 2 * self.TP/(2 * self.TP + self.FP + self.FN)

    def compute(self):
        output = dict(precision = self._compute_precision(),
                      recall = self._compute_recall(),
                      F1 = self._compute_F1()
                      )
        return output

    def reset(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0

class E2ePredicatelessReMetrics(Metric):
    
    def __init__(self, relation_class):
        
        self.relation_class = relation_class
        
        self.TP = 0
        self.FP = 0
        self.FN = 0
        
        
    def update(self, true, preds):
        TP = 0
        FP = 0
        
        for el in preds:
            if self.relation_class.check_candidate_veracity(el, true):
                TP += 1
            else:
                FP += 1
                
        FN = len(true) - TP
        
        self.TP += TP
        self.FP += FP
        self.FN += FN

#%%
class UniclassNerMetrics(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    
    def __init__(self, mention_class):
        super().__init__()
        
        self.mention_class = mention_class
        
        self.TP = 0
        self.FP = 0
        self.FN = 0

    def update(self, true, preds):
        TP = 0
        FP = 0
        
        for el in preds:
            if self.mention_class.check_candidate_veracity(el, true):
                TP += 1
            else:
                FP += 1
                
        FN = len(true) - TP
        
        self.TP += TP
        self.FP += FP
        self.FN += FN
    
    
#%%
class MulticlassIeMetrics(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False 
    
    def __init__(self, classes):
        super().__init__()
        
        self.classes = classes
        
        self.TP = 0
        self.FP = 0
        self.FN = 0
        
        for el in classes:
            setattr(self, f'TP_{el}', 0)
            setattr(self, f'FP_{el}', 0)
            setattr(self, f'FN_{el}', 0)
            
            
    def _compute_class_counts(self):
        self.TP = sum(getattr(self, f'TP_{el}') for el in self.classes)
        self.FP = sum(getattr(self, f'FP_{el}') for el in self.classes)
        self.FN = sum(getattr(self, f'FN_{el}') for el in self.classes)
    
    def _compute_class_F1(self, class_):
        TP = getattr(self, f'TP_{class_}')
        FP = getattr(self, f'FP_{class_}')
        FN = getattr(self, f'FN_{class_}')
        
        if TP + FP + FN == 0:
            return 0
        else:
            return 2 * TP/(2 * TP + FP + FN)
    
    def _compute_macro_F1(self):
        return sum(self._compute_class_F1(el) for el in self.classes)/len(self.classes)
    
    def compute(self):
        self._compute_class_counts()
        
        output = dict(micro_precision = self._compute_precision(),
                      micro_recall = self._compute_recall(),
                      micro_F1 = self._compute_F1(),
                      
                      macro_F1 = self._compute_macro_F1()
                      )
        
        return output
        
        
class E2ePredicatefulReMetrics(MulticlassIeMetrics):
    
    def __init__(self, relation_class, classes#, predicate_converter
                 ):
        
        super().__init__(classes)
        self.relation_class = relation_class
        #self.predicate_converter = predicate_converter
    
    def _update_for_micro_scores(self, true, preds):
                        
        TP = 0
        FP = 0
        
        for el in preds:
            if self.relation_class.check_candidate_veracity(el, true):
                TP += 1
            else:
                FP += 1
                
        FN = len(true) - TP
        
        self.TP += TP
        self.FP += FP
        self.FN += FN
        
    def _update_for_macro_scores(self, true, preds):
        
        counts_dict = {el: {'TP': 0, 'FP': 0, 'FN': 0} for el in self.classes}
        
        for el in preds:
            predicate = el['predicate']
            if self.relation_class.check_candidate_veracity(el, true):
                counts_dict[predicate]['TP'] += 1
            else: 
                if predicate in self.classes: # only increment FP's if a valid predicate is predicted.  This is okay because we're dealing with a finite set of valid predicates.
                    counts_dict[predicate]['FP'] += 1
                
        # get FN's
        true_predicate_counts = defaultdict(lambda: 0)
        for el in true:
            true_predicate_counts[el.formal_predicate] += 1
        
        for el in self.classes:
            counts_dict[el]['FN'] = true_predicate_counts[el] - counts_dict[el]['TP']
        
        # updating metric states
        for el in self.classes:
            
            setattr(self, 
                    f'TP_{el}', 
                    getattr(self, f'TP_{el}') + counts_dict[el]['TP'])
 
            setattr(self, 
                    f'FP_{el}', 
                    getattr(self, f'FP_{el}') + counts_dict[el]['FP'])

            setattr(self, 
                    f'FN_{el}', 
                    getattr(self, f'FN_{el}') + counts_dict[el]['FN'])

    def update(self, true, preds):
        
        self._update_for_micro_scores(true, preds)
        self._update_for_macro_scores(true, preds)
        
#%%
class MulticlassNerMetrics(MulticlassIeMetrics):
    
    def __init__(self, mention_class, classes):
        super().__init__(classes)
        self.mention_class = mention_class

    def _update_for_micro_scores(self, true, preds):
                        
        TP = 0
        FP = 0
        
        for el in preds:
            if self.mention_class.check_candidate_veracity(el, true):
                TP += 1
            else:
                FP += 1
                
        FN = len(true) - TP
        
        self.TP += TP
        self.FP += FP
        self.FN += FN
        
    def _update_for_macro_scores(self, true, preds):
        
        counts_dict = {el: {'TP': 0, 'FP': 0, 'FN': 0} for el in self.classes}
        
        for el in preds:
            entity_type = el['entity_type']
            if self.mention_class.check_candidate_veracity(el, true):
                counts_dict[entity_type]['TP'] += 1
            else:
                if entity_type in self.classes:
                    counts_dict[entity_type]['FP'] += 1
                
        # get FN's
        true_class_counts = defaultdict(lambda: 0)
        for el in true:
            true_class_counts[el.entity_type_string] += 1
        
        for el in self.classes:
            counts_dict[el]['FN'] = true_class_counts[el] - counts_dict[el]['TP']
        
        # updating metric states
        for el in self.classes:
            
            setattr(self, 
                    f'TP_{el}', 
                    getattr(self, f'TP_{el}') + counts_dict[el]['TP'])
 
            setattr(self, 
                    f'FP_{el}', 
                    getattr(self, f'FP_{el}') + counts_dict[el]['FP'])

            setattr(self, 
                    f'FN_{el}', 
                    getattr(self, f'FN_{el}') + counts_dict[el]['FN'])

    def update(self, true, preds):
        
        self._update_for_micro_scores(true, preds)
        self._update_for_macro_scores(true, preds)    
        
     