import random
import numpy as np
import torch

from datasets import load_dataset
import datasets
from torch.utils.data.dataset import Dataset

def get_c4(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
         "json", data_files={'train': '/share/public/hanling/c4/en/c4-train.00000-of-01024.json.gz'}
    )['train']
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len )
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0)

def get_bookcorpus(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'bookcorpus', split='train'
    )
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0)

def get_alpaca_bi_chat(tokenizer, n_samples, seq_len):
    random.seed(42)
    traindata = load_dataset('json', data_files='/share/public/hanling/DecompLLM/alpaca_conversation.jsonl', split='train')
    traindata = traindata['messages']
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            print(traindata[i])
            tokenized_sample = tokenizer.apply_chat_template(traindata[i], tokenize=True, add_generation_prompt=True, return_tensors="pt")
            if tokenized_sample.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        tokenized_samples.append(tokenized_sample[:, :seq_len])
    return torch.cat(tokenized_samples, dim=0)

def get_alpaca_chat(tokenizer, n_samples, seq_len):
    random.seed(42)
    traindata = load_dataset('json', data_files='/share/public/hanling/DecompLLM/alpaca_conversation.jsonl', split='train')
    traindata = traindata['messages']
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, 52048)
            i = i * 2 + 1
            print(traindata[i])
            tokenized_sample = tokenizer.apply_chat_template(traindata[i], tokenize=True, add_generation_prompt=True, return_tensors="pt")
            if tokenized_sample.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        tokenized_samples.append(tokenized_sample[:, :seq_len])
    print(torch.cat(tokenized_samples, dim=0).size())
    return torch.cat(tokenized_samples, dim=0)

def get_alpaca_bi(tokenizer, n_samples, seq_len):
    random.seed(42)
    traindata = load_dataset("silk-road/alpaca-data-gpt4-chinese")
    traindata = traindata['train']
    combined_instruction = traindata['instruction'] + traindata['instruction_zh']
    combined_input = traindata['input'] + traindata['input_zh']
    combined_output = traindata['output'] + traindata['output_zh']
    traindata = datasets.Dataset.from_dict({'instruction':combined_instruction, 
                                    'input': combined_input,
                                    'output':combined_output})

    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['output'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len )
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0)

def get_alpaca(tokenizer, n_samples, seq_len):
    random.seed(42)
    traindata = load_dataset("silk-road/alpaca-data-gpt4-chinese")
    traindata = traindata['train']
    combined_instruction = traindata['instruction']
    combined_input = traindata['input']
    combined_output = traindata['output']
    traindata = datasets.Dataset.from_dict({'instruction':combined_instruction, 
                                    'input': combined_input,
                                    'output':combined_output})

    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['output'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len )
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0)

def get_examples(dataset, tokenizer, n_samples, seq_len = 128):
    if dataset == 'c4':
        return get_c4(tokenizer, n_samples, seq_len)
    elif dataset == 'bookcorpus':
        return get_bookcorpus(tokenizer, n_samples, seq_len)
    elif dataset == 'alpaca':
        return get_alpaca(tokenizer, n_samples, seq_len)
    elif dataset == 'alpaca_bi':
        return get_alpaca_bi(tokenizer, n_samples, seq_len)
    elif dataset == 'alpaca_chat':
        return get_alpaca_chat(tokenizer, n_samples, seq_len)
    elif dataset == 'alpaca_bi_chat':
        return get_alpaca_bi_chat(tokenizer, n_samples, seq_len)
    else:
        raise NotImplementedError
