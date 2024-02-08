
# Language learning with context

This repo provides code for finetuning language models with a context, as used in [Conditional language learning with context](link pending).

The medical textbooks data can be downloaded from the MedQA [repo](https://github.com/jind11/MedQA).

We release the medical QA data used in the paper on the Huggingface hub [repo](https://huggingface.co/datasets/winder-hybrids/MedicalTextbook_QA).


## Training
Finetune llama-2 7b with `Following is an excerpt from a medical textbook.` as context (on one A100 80GB GPU)
```
python run.py --config-name finetune_text_context.yaml
```
`dataset`, `model` and `context` can be spedified in the config file.

Finetune llama-2 7b with learned soft prompt as context 
- First, learn a soft context with prompt tuning
```
python run_learn_context.py --config-name learn_context.yaml
```
- Then, finetune llama-2 7B with the learned context
```
python run.py --config-name finetune_learned_context.yaml
```


## Evaluation
Evaluate finetuned model on language modeling
```
python run.py --config-name eval_lm.yaml
```

Evaluate finetuned model on medical QA
```
python lm-evaluation-harness/main.py --model hf-causal-experimental --model_args pretrained=save/full_model_text_context,use_accelerate=True --no_cache --tasks Anatomy_Gray --num_fewshot 5 --batch_size auto:3
```
replace `save/full_model_text_context` with the path to the model to evaluate and replace `Anatomy_Gray` with the name of the QA dataset to evaluate on. See [here](https://huggingface.co/datasets/winder-hybrids/MedicalTextbook_QA) for all the subjects in the medical QA dataset.


## Configuration
Hyperparameters, models and training/evaluation datasets are specified using config files in `config/`.


## Citation
```
pending
```