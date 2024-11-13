# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import os
import wandb
from peft import get_peft_model
from datasets import load_dataset
from torch import nn
# from syne_tune import Reporter

from finetune_utils.finetune_prep import get_dataset, config_prep, create_alpaca_prompt
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
# from unsloth import FastLanguageModel
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["https_proxy"]="http://10.10.20.100:1089"
os.environ['HF_HOME'] = "/mnt/public/hanling/cache"
os.environ['TRANSFORMERS_CACHE'] = "/share/public/hanling/cache"
os.environ['HF_DATASETS_CACHE'] = "/share/public/hanling/dataset_cache"
#os.environ["WANDB_PROJECT"] = "alpaca_ft"  # name your W&B project
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# TODO: add unsloth 

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/Qwen2-7B-BD6")
    parser.add_argument("--dataset", type=str, default="llamafactory/alpaca_en")
    parser.add_argument("--output_dir", type=str, default="/ft")
    parser.add_argument("--use_lora", type=bool, default=False)
    # parser.add_argument("--use_unsloth", type=bool, default=True)

    args = parser.parse_args()

    # if args.use_unsloth:
        # model, tokenizer = FastLanguageModel.from_pretrained(args.model)
    # else: 
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code = True)
    # for asvd recovery finetuning, freeze layers that not decomposed
    # module_dict = {name: module for name, module in model.named_modules()}
    # full_name_dict = {module: name for name, module in model.named_modules()}
    # linear_info = {}
    # modules = [model]
    # while len(modules) > 0:
    #     submodule = modules.pop()
    #     for name, raw_linear in submodule.named_children():
    #         if "lm_head" in name:
    #             continue
    #         if isinstance(raw_linear, nn.Linear) and name not in ['ALinear_no_train','ALinear_train','BLinear_no_train', 'BLinear_train']:
    #             print(name)
    #             raw_linear.weight.requires_grad = False
    #             if raw_linear.bias is not None:
    #                 raw_linear.bias.requires_grad = False
    #         else:
    #             modules.append(raw_linear)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code = True)
    if "llama" in args.model.lower():
        print("Llama model detected, set pad token to <|finetune_right_pad_id|>")
        tokenizer.pad_token = '<|finetune_right_pad_id|>'
    else:
        print("using original pad token")

    train_dataset = load_dataset('json', data_files='alpaca_conversation.jsonl', split='train')

    training_args, peft_config = config_prep(args.use_lora, args.output_dir)

    #if not args.use_lora:
    #    peft_config = None
    #else:
        #if args.use_unsloth:
        #    peft_model = FastLanguageModel.get_peft_model(kwargs=peft_config)
        #else:
    #    peft_model = get_peft_model(model, peft_config)
    #    peft_model.print_trainable_parameters()

    if args.use_lora:
        print("LoRA is used")
        trainer = SFTTrainer(
            model,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            max_seq_length=1024, # maximum packed length 
            args=training_args,
            peft_config=peft_config,
        )
    else:
        trainer = SFTTrainer(
            model,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            max_seq_length=1024, # maximum packed length 
            args=training_args,
        )
    trainer.train()

    # if trainer.is_fsdp_enabled:
    #     trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

if __name__ == "__main__":
    main()