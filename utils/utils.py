import os
from collections import Counter
import importlib
import re


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

def unlist(nested_list):
    unlisted = [subel for el in nested_list for subel in el]
    return unlisted

def sum_dictionaries(dictionaries):
    counter = Counter()
    for el in dictionaries:
        counter.update(el)
         
    result = dict(counter)
    return result

def reverse_dict(input_dict):
    if len(input_dict) > len(set(input_dict.values())):
        return 'dict is not 1-1'
    else:
        new_dict = dict(zip(input_dict.values(), 
                            input_dict.keys()))
        
        return new_dict
    
def dict2frozenset(dictionary):
    return frozenset(dictionary.items())

def frozenset2dict(froze):
    return dict(froze)

def dict_list2frozenset_set(dict_list):
    return set(dict2frozenset(el) for el in dict_list)

def frozenset_set2dict_list(frozenset_set):
    return [frozenset2dict(el) for el in frozenset_set]
    
def unique_dicts(dict_list):
    frozenset_list = [frozenset(el.items()) for el in dict_list]
    unique_frozensets = set(frozenset_list)
    unique_dicts = [dict(el) for el in unique_frozensets]
    
    return unique_dicts
    
def avg_dicts(dict_list):
    keys = dict_list[0].keys()
    return {el_key: sum([el_dict[el_key] for el_dict in dict_list])/len(dict_list) for el_key in keys}

def list_of_dicts2dict_of_lists(list_of_dicts):
    keys = list_of_dicts[0].keys()
    return {el_key: [el_dict[el_key] for el_dict in list_of_dicts] for el_key in keys}
    
def comma_separated_string(elements):
    elements = list(elements)
    if len(elements) == 0:
        return None
    elif len(elements) == 1:
        return elements[0]
    elif len(elements) == 2:
        return f'{elements[0]} and {elements[1]}'
    elif len(elements) >= 3:
        return f'{", ".join(elements[:-1])}, and {elements[-1]}'

def comma_separated_string2(elements):
    elements = list(elements)
    if len(elements) == 0:
        return ''
    elif len(elements) == 1:
        return elements[0]
    elif len(elements) == 2:
        return f'{elements[0]} and {elements[1]}'
    elif len(elements) >= 3:
        return f'{", ".join(elements[:-1])}, and {elements[-1]}'

      
def separate_list(string):
    elements = re.split(', and |, | and ', string)
    elements = [el.lstrip().rstrip() for el in elements]    
    return elements
        
def import_all_from_file(file_path:str)->None:
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals().update(module.__dict__)

class QueryDict(dict):
    def __missing__(self, key):
        return key







    
    
    
    
    