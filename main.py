#%% libraries
import logging
import os
from dataclasses import dataclass, field
from typing import Optional
from importlib import import_module
from copy import deepcopy
import json
import csv


import torch
import numpy as np
from datasets import Dataset

from deepspeed.utils import safe_get_full_fp32_param
from peft import prepare_model_for_kbit_training, get_peft_model, PeftType


import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
    TrainerCallback
)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

#%% my libraries
from metrics import metrics
from utils.utils import unlist, list_of_dicts2dict_of_lists


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_auth_token: bool = field(default = False)

    load_in_8bit: bool = field(default = False)
    load_in_4bit: bool = field(default = False)

    peft_config_file: Optional[str] = field(default = None)
    lora_config_file: Optional[str] = field(default = None)   
    adalora_config_file: Optional[str] = field(default = None)
    ia3_config_file: Optional[str] = field(default = None)
    
    def _resolve_precision_conflicts(self):
        if self.load_in_4bit and self.load_in_8bit:
            self.load_in_8bit = False

    def _model_structure_logic(self):

        # determine if model is decoder-only or encoder-decoder
        if 'Llama-2' in self.model_name_or_path:
            model_structure = 'decoder'
        else:
            # config = AutoConfig.from_pretrained(self.model_name_or_path, use_auth_token = True, trust_remote_code = True)
            config = AutoConfig.from_pretrained(self.model_name_or_path)
            model_structure = 'encoder-decoder' if config.is_encoder_decoder else 'decoder'    

        model_structure2automodel = {'decoder': AutoModelForCausalLM, 
                                     'encoder-decoder': AutoModelForSeq2SeqLM}
        model_structure2train_preprocess_function = {'decoder': preprocess_train_example_decoder_v3, 
                                                        'encoder-decoder': preprocess_example_encoder_decoder}
        model_structure2eval_preprocess_function = {'decoder': preprocess_eval_example_decoder_v3, 
                                                        'encoder-decoder': preprocess_example_encoder_decoder}
        model_structure2padding_side = {'decoder': 'left',
                                        'encoder-decoder': 'right'}
        

        self.model_structure = model_structure
        self.automodel_class = model_structure2automodel[model_structure]
        self.train_preprocess_function = model_structure2train_preprocess_function[model_structure]
        self.eval_preprocess_function = model_structure2eval_preprocess_function[model_structure]
        self.padding_side = model_structure2padding_side[model_structure]

    def _peft_logic(self):
        from peft import LoraConfig, AdaLoraConfig, IA3Config

        model_structure2peft_task_type = {'decoder': 'CAUSAL_LM',
                                        'encoder-decoder': 'SEQ_2_SEQ_LM'}
        


        peft_type2config = {PeftType.LORA: LoraConfig,
                            PeftType.ADALORA: AdaLoraConfig,
                            PeftType.IA3: IA3Config}
        
        with open(self.peft_config_file, "r") as file:
            peft_config_dict = json.load(file)

        peft_config_class = peft_type2config[peft_config_dict['peft_type']]
        peft_config_dict['task_type'] = model_structure2peft_task_type[self.model_structure]        

        self.peft_config = peft_config_class(**peft_config_dict)
            
    
    def __post_init__(self):
        self._resolve_precision_conflicts()
        self._model_structure_logic()
        if self.peft_config_file:
            self._peft_logic()
       
@dataclass
class DataArguments():
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: str
    template: str

    base_data_path: Optional[str] = field(default = None)
    train_file: Optional[str] = field(default = None)
    valid_file: Optional[str] = field(default = None)
    test_file: Optional[str] = field(default = None)

    max_source_length: Optional[int] = field(
        default = 512,
        metadata = {
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default = 256,
        metadata = {
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default = None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default = False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_examples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_valid_examples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_test_examples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
   
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    
    def _imports(self):
        self.data_structures = import_module(f'data_structures.{self.dataset_name}.data_structures')
        self.templates = import_module(f'templates.{self.dataset_name}.templates')

    def _get_template(self):
        self.template = getattr(self.templates, self.template)

    def _get_task(self):
        self.task = self.template.task

    def _data_file_logic(self):
        if self.base_data_path is not None:
            self.train_file = os.path.join(self.base_data_path, self.dataset_name, 'train_data')
            self.valid_file = os.path.join(self.base_data_path, self.dataset_name, 'valid_data')
            self.test_file = os.path.join(self.base_data_path, self.dataset_name, 'test_data')

    def _get_val_max_target_length(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length
    
    def _metric_logic(self):
        
        if self.dataset_name == 'CDR':
            self.metrics = metrics.E2ePredicatelessReMetrics(self.data_structures.Relation)
            self.monitor_metric = 'F1'
        elif self.dataset_name == 'ChemProt':
            self.metrics = metrics.E2ePredicatefulReMetrics(self.data_structures.Relation, ['CPR:3', 'CPR:4', 'CPR:5', 'CPR:6', 'CPR:9'])
            self.monitor_metric = 'micro_F1'
        elif self.dataset_name == 'DDI':
            self.metrics = metrics.E2ePredicatefulReMetrics(self.data_structures.Relation, ['INT', 'MECHANISM', 'EFFECT', 'ADVISE'])
            self.monitor_metric = 'micro_F1'
        elif self.dataset_name == 'NaryDrugCombos':
            self.metrics = metrics.E2ePredicatefulReMetrics(self.data_structures.Relation, ['POS', 'COMB'])
            self.monitor_metric = 'micro_F1'

    def __post_init__(self):
        self._imports()
        self._get_template()
        self._get_task() 
        self._data_file_logic()
        self._get_val_max_target_length()
        self._metric_logic()


#%% useful functions
def preprocess_train_example_decoder_v1(template, tokenizer):
    sequence = template.make_sequence()
    model_inputs = tokenizer(sequence)
    model_inputs['labels'] = deepcopy(model_inputs.input_ids)
    
    return model_inputs

def preprocess_train_example_decoder_v2(template, tokenizer):

    source_sequence = template.make_source_sequence()
    target_sequence = template.make_target_sequence()

    source_ids = tokenizer.encode(source_sequence, add_special_tokens = False)
    target_ids = tokenizer.encode(target_sequence, add_special_tokens = False)

    def _does_begin_with_special():
        return tokenizer.encode('just a placeholder')[0] == tokenizer.bos_token_id
    def _does_end_with_special():
        return tokenizer.encode('just a placeholder')[-1] == tokenizer.eos_token_id
    
    labels = [-1] * len(source_ids) + target_ids

    if _does_begin_with_special():
        source_ids = [tokenizer.bos_token_id] + source_ids
    if _does_end_with_special():
        target_ids = target_ids + [tokenizer.eos_token_id]
    

    sequence_ids = source_ids + target_ids
    labels = [-100] * len(source_ids) + target_ids
    mask = [1] * len(sequence_ids)

    model_inputs = {'input_ids': sequence_ids,
                    'attention_mask': mask,
                    'labels': labels}
        
    return model_inputs

def preprocess_train_example_decoder_v3(template, tokenizer):

    source_sequence = template.make_source_sequence()
    target_sequence = template.make_target_sequence()

    source_ids = tokenizer.encode(source_sequence, add_special_tokens = False)
    target_ids = tokenizer.encode(target_sequence, add_special_tokens = False)

    def _does_begin_with_special():
        return tokenizer.encode('just a placeholder')[0] == tokenizer.bos_token_id
    def _does_end_with_special():
        return tokenizer.encode('just a placeholder')[-1] == tokenizer.eos_token_id
    
    labels = [-1] * len(source_ids) + target_ids

    if _does_begin_with_special():
        source_ids = [tokenizer.bos_token_id] + source_ids
    if _does_end_with_special():
        target_ids = target_ids + [tokenizer.eos_token_id]
    
    if tokenizer.model_max_length is not None:
        if len(source_ids + target_ids) > tokenizer.model_max_length:
            print('before: ', len(source_ids + target_ids))
            source_ids = source_ids[0:tokenizer.model_max_length - len(target_ids)]
            print('after: ', len(source_ids + target_ids))


    sequence_ids = source_ids + target_ids
    labels = [-100] * len(source_ids) + target_ids
    mask = [1] * len(sequence_ids)

    model_inputs = {'input_ids': sequence_ids,
                    'attention_mask': mask,
                    'labels': labels}
        
    return model_inputs

def preprocess_eval_example_decoder(template, tokenizer):
    sequence = template.make_source_sequence()
    model_inputs = tokenizer(sequence)
    model_inputs['labels'] = deepcopy(model_inputs.input_ids)


    return model_inputs

def preprocess_eval_example_decoder_v3(template, tokenizer):
    sequence = template.make_source_sequence()
    model_inputs = tokenizer(sequence, truncation = True)
    model_inputs['labels'] = deepcopy(model_inputs.input_ids)

    return model_inputs

def preprocess_example_encoder_decoder(template, tokenizer):
    source_sequence = template.make_source_sequence()
    target_sequence = template.make_target_sequence()
    
    source_inputs = tokenizer(source_sequence, max_length = 512, truncation = True)
    target_inputs = tokenizer(target_sequence, max_length = 512, truncation = True)

    model_inputs = source_inputs
    model_inputs['labels'] = target_inputs.input_ids

    return model_inputs

def datasetify_dataset(raw_dataset, template, preprocess_function, tokenizer):
    def datasetify_example(example):
        filled_template = template.from_example(example)
        model_inputs = preprocess_function(filled_template, tokenizer)
        return model_inputs
    
    def list_of_dicts2dict_of_lists(list_of_dicts):
        keys = list_of_dicts[0].keys()
        return {el_key: [el_dict[el_key] for el_dict in list_of_dicts] for el_key in keys}

    examples = raw_dataset.examples
    processed_examples = [datasetify_example(el) for el in examples]
    dataset_dict = list_of_dicts2dict_of_lists(processed_examples)
    processed_dataset = Dataset.from_dict(dataset_dict)
    
    return processed_dataset



#%% Trainer
class MySeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, metrics, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = metrics
        self.compute_metrics = self.my_compute_metrics
        # print('compute_metrics: ', self.compute_metrics)


    def my_compute_metrics(self, eval_preds):
        
        generated_ids, input_ids = eval_preds

        # Replace -100s used for padding as we can't decode them
        generated_ids = np.where(generated_ids != -100, generated_ids, self.tokenizer.pad_token_id)
        generated_sequences = self.tokenizer.batch_decode(generated_ids, skip_special_tokens = True)
        

 
        true_list = []
        prediction_list = []
        sequence_list = []
        counter = 0
        for el_template, el_seq in zip(self.templates, generated_sequences):
            counter += 1
            el_template.generated_sequence = el_seq

            predictions = el_template.extract_prediction()
            true = getattr(el_template, el_template.label_name)
            
            self.metrics.update(true, predictions)

            true_list.append(true)
            sequence_list.append(el_seq)
            prediction_list.append(predictions)
        
        metric_values = deepcopy(self.metrics.compute())
        print('performance: ', metric_values)
        self.metrics.reset()
        
        return metric_values



#%% callbacks
class LogValidationMetricsCallback(TrainerCallback):
    def __init__(self):
        self.metrics = []

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.metrics.append(metrics)
        with open(os.path.join(args.output_dir, 'valid_performance.csv'), 'w') as file:
            writer = csv.DictWriter(file, fieldnames = self.metrics[0].keys())
            writer.writeheader()
            writer.writerows(self.metrics)
    def on_train_end(self, args, state, control, **kwargs):
        with open(os.path.join(args.output_dir, 'valid_performance.csv'), 'w') as file:
            writer = csv.DictWriter(file, fieldnames = self.metrics[0].keys())
            writer.writeheader()
            writer.writerows(self.metrics)

#%% program
def main():
    ### command line argument parsing
    parser = HfArgumentParser((DataArguments, ModelArguments, Seq2SeqTrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()
    data_args.train_preprocess_function = model_args.train_preprocess_function
    data_args.eval_preprocess_function = model_args.eval_preprocess_function
    
    # Set seed before initializing model.
    set_seed(training_args.seed)  
   
    # download model & tokenizer
    # config =  AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                              padding_side = model_args.padding_side
                                              )
    model = model_args.automodel_class.from_pretrained(model_args.model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token":"<pad>"})
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))

    # add trainable tokens (if applicable)
    if data_args.template.is_trainable:
        tokenizer.add_tokens(data_args.template.prompt_tokens)
        model.resize_token_embeddings(len(tokenizer))

    # peft
    if model_args.peft_config_file is not None:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, model_args.peft_config)

    # model_config
    model.generation_config.do_sample = False
 
    # data
    raw_train_dataset = torch.load(data_args.train_file)
    raw_valid_dataset = torch.load(data_args.valid_file)
    raw_test_dataset = torch.load(data_args.test_file)

    train_dataset = datasetify_dataset(raw_train_dataset, 
                                        data_args.template, 
                                        model_args.train_preprocess_function, 
                                        tokenizer)
    valid_dataset = datasetify_dataset(raw_valid_dataset, 
                                        data_args.template, 
                                        model_args.eval_preprocess_function, 
                                        tokenizer)
    test_dataset = datasetify_dataset(raw_test_dataset, 
                                        data_args.template, 
                                        model_args.eval_preprocess_function,
                                        tokenizer)
    
    def filter_long_examples(example):
        return len(example['input_ids']) <= tokenizer.model_max_length
    
    if model_args.model_structure == 'decoder':
        train_dataset = train_dataset.filter(filter_long_examples)
    
    # gets templates for valid and test sets
    def templatify_dataset(raw_dataset, template):
        def templatify_example(example):
            return template.from_example(example)

        return [templatify_example(el) for el in raw_dataset.examples]
    
    valid_templates = templatify_dataset(raw_valid_dataset, data_args.template)
    test_templates = templatify_dataset(raw_test_dataset, data_args.template)

    # making datasets smaller for quick code testing
    if data_args.max_train_examples is not None:
        max_train_examples = min(data_args.max_train_examples, len(train_dataset))
        train_dataset = train_dataset.shuffle(seed = training_args.data_seed).select(range(max_train_examples))
    if data_args.max_valid_examples is not None:   
        max_valid_examples = min(data_args.max_valid_examples, len(valid_dataset))
        valid_dataset = valid_dataset.shuffle(seed = training_args.data_seed).select(range(max_valid_examples))
        valid_templates = valid_templates[0:max_valid_examples]
    if data_args.max_test_examples is not None:
        max_test_examples = min(data_args.max_test_examples, len(test_dataset))
        test_dataset = test_dataset.shuffle(seed = training_args.data_seed).select(range(max_test_examples))
        test_templates = test_templates[0:max_test_examples]


    # Data collator
    label_pad_token_id = -100
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # callbacks
    log_validation_metrics_callback = LogValidationMetricsCallback()


    # Initialize our Trainer
    trainer = MySeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset = train_dataset if training_args.do_train else None,
        eval_dataset = valid_dataset if training_args.do_eval else None,
        tokenizer = tokenizer,
        data_collator = data_collator,
        compute_metrics = None,
        metrics = data_args.metrics,
        callbacks = [log_validation_metrics_callback] if training_args.do_eval else []
    )

    ##########
    # training
    if training_args.do_train:
        trainer.templates = valid_templates

        print('training')
        train_result = trainer.train()

        eval_metrics = train_result.metrics

        trainer.log_metrics("eval", eval_metrics)

        trainer.save_metrics("eval", eval_metrics)

        trainer.save_state()

        # print('valid metric: ', log_validation_metrics_callback.metrics)
        # torch.save(log_validation_metrics_callback.metrics, os.path.join(training_args.output_dir, 'validation_metrics.save'))
        
    # testing
    if training_args.do_predict:
        print('testing')
        trainer.templates = test_templates

        test_result = trainer.predict(test_dataset)

        torch.save(test_result, os.path.join(training_args.output_dir, 'test_result'))

        metrics = test_result.metrics

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    


if __name__ == "__main__":
    main()
