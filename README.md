# How Important is Domain Specificity in Language Models and Instruction Finetuning for Biomedical Relation Extraction?
---
# Installation
Installation directions are for a Linux machine with Python &ge; 3.8.  Download the repository from the command line using 

```git clone https://anonymous.4open.science/r/domain_specificity-BCB8.git```

## Setting up environment
To set up the `conda` environment to run our code, use the `DomainSpecificity.yaml` definition file in our repository:

1. Navigate to the cloned repository, e.g., `cd domain_specificity`
2. Create the conda environment: `conda env create -f DomainSpecificy.yaml`

## Additional setup
To run our code for the ChemProt dataset, we need to install the `en_core_sci_lg` model from [https://allenai.github.io/scispacy/](SciSpaCy):

1. First, activate the conda environment: `conda activate DomainSpecificity`
2. Install the model: `python -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz`

# Dataset processing
Before any training, raw datasets must first be converted into natural language via: `bash create_datasets.sh`

# Running main code
## Setting variables

To run our code, first set the following environmental variables:

- MODEL: the name of the model we use, available on `huggingface`.
- DATASET: the name of the dataset we use.
- TEMPLATE: the template used to convert raw dataset examples into natural language sequences.
- MAX_LENGTH: the maximum output (generated) sequence length allowed.

Based on whether you want to run full-dataset or few-shot finetuning, set additional environmental variables:

- Full-dataset finetuning
  - EPOCHS: 30.  We train each model for 30 epochs, and pick the "best model"
  - METRIC: For full-dataset finetuning, a performance metric is used to choose the "best" epoch's model to conduct inference with; so a metric must be supplied.
- Few-shot finetuning
  - EPOCHS: We use no validation set to choose the "best number of training epochs, so we set a hard number of epochs.
  - SEED: For few-shot finetuning, we sample training examples using a random seed environmental variable SEED.  We used the values 0, 1, 2, 3, and 4.

We designed our code to be easily extensible, but we provide default METRIC and TEMPLATE for each dataset, as well as defaults for maximum output length (MAX_LENGTH) and number of EPOCH trained in few-shot finetuning (few-shot experiments were not performed with decoder-only models, as full-dataset finetuning performance was poor with them) for each MODEL.

| DATASET  | TEMPLATE                    | METRIC   |
|----------|-----------------------------|----------|
| CDR      | E2eReHardTemplate1          | F1       |
| DCE      | E2eReHardTemplate1_sentence | micro_F1 |
| ChemProt | E2eReHardTemplate1          | micro_F1 |
| DDI      | E2eReHardTemplate1          | micro_F1 |

| MODEL                       | 16-shot EPOCHS | 64-shot EPOCHS | MAX_LENGTH |
|-----------------------------|----------------|----------------|------------|
| t5-small                    | 160            | 60             | 512        |
| t5-base                     | 100            | 40             | 512        |
| t5-large                    | 50             | 30             | 512        |
| google/flan-t5-small        | 160            | 60             | 512        |
| google/flan-t5-base         | 100            | 40             | 512        |
| google/flan-t5-large        | 50             | 30             | 512        |
| razent/SciFive-base-Pubmed  | 100            | 40             | 512        |
| razent/SciFive-large-Pubmed | 50             | 30             | 512        |
| facebook/bart-base          | 130            | 50             | 512        |
| facebook/bart-large         | 75             | 35             | 512        |
| GanjinZero/biobart-v2-base  | 130            | 50             | 512        |
| GanjinZero/biobart-v2-large | 75             | 35             | 512        |
| cogint/in-boxbart           | 130            | 50             | 512        |
| gpt2-medium                 | -              | -              | 1024       |
| gpt2-xl                     | -              | -              | 1024       |
| EleutherAI/gpt-neo-2.7B     | -              | -              | 1024       |
| microsoft/biogpt            | -              | -              | 1024       |
| microsoft/BioGPT-Large      | -              | -              | 1024       |
| stanford-crfm/BioMedLM      | -              | -              | 1024       |

## Run
We use the deepspeed launcher to run our code, regardless of whether any deepspeed strategies like sharding or CPU offload are used.  

### Full-dataset finetuning
To run full-dataset finetuning without deepspeed strategies, run:

```
deepspeed main.py \
--model_name_or_path $MODEL \
--generation_max_length $MAX_LENGTH \
--num_train_epochs $EPOCHS \
--dataset_name $DATASET \
--template $TEMPLATE \
--metric_for_best_model $METRIC \
--per_device_train_batch_size 1 \
--do_train \
--do_eval \
--do_predict \
--predict_with_generate \
--base_data_path data/processed_data \
--evaluation_strategy epoch \
--warmup_ratio 0.1 \
--gradient_accumulation 32 \
--save_strategy epoch \
--load_best_model_at_end \
--save_total_limit 1 \
--report_to none \
--eval_delay 10 \
--ddp_find_unused_parameters false
--output_dir output
```

### Few-shot finetuning
To run **64**-shot finetuning without deepspeed strategies, run:

```
deepspeed main.py \
--model_name_or_path $MODEL \
--generation_max_length $MAX_LENGTH \
--num_train_epochs $EPOCHS \
--dataset_name $DATASET \
--template $TEMPLATE \
--metric_for_best_model $METRIC \
--per_device_train_batch_size 1 \
--do_train \
--do_predict \
--predict_with_generate \
--base_data_path data/processed_data \
--evaluation_strategy no \
--warmup_ratio 0.1 \
--gradient_accumulation 32 \
--report_to none \
--ddp_find_unused_parameters false \
--max_train_examples 64 \
--data_seed $SEED
--output_dir output
```
To run **16**-shot finetuning, change `max_train_examples` to `16`.

### CPU offloading
To use CPU offloading for larger models, add another optional argument:

``` --deepspeed deepspeed_configs/ds_config_stage3.json ```

### Output
Output will be found in the `output` directory.  Change the `--output dir` option to choose another location.
