# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import os
import wandb
from peft import get_peft_model
# from syne_tune import Reporter

from finetune_utils.finetune_prep import get_dataset, config_prep, create_alpaca_prompt
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
# from unsloth import FastLanguageModel
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["https_proxy"]="http://10.10.20.100:1089"
os.environ['HF_HOME'] = "/mnt/public/hanling/cache"
os.environ['TRANSFORMERS_CACHE'] = "/mnt/public/hanling/cache"
os.environ['HF_DATASETS_CACHE'] = "/mnt/public/hanling/dataset_cache"
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
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code = True)
    if "llama" in args.model.lower():
        print("Llama model detected, set pad token to <|finetune_right_pad_id|>")
        tokenizer.pad_token = '<|finetune_right_pad_id|>'
    elif "phi" in args.model.lower():
        tokenizer.pad_token_id = 0
    else:
        print("using original pad token")

    train_dataset, eval_dataset = get_dataset(args.dataset)

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
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            packing=True,
            max_seq_length=2048, # maximum packed length 
            args=training_args,
            peft_config=peft_config,
            formatting_func=create_alpaca_prompt, # format samples with a model schema
        )
    else:
        trainer = SFTTrainer(
            model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            packing=True,
            max_seq_length=2048, # maximum packed length 
            args=training_args,
            formatting_func=create_alpaca_prompt, # format samples with a model schema
        )
    trainer.train()

    # if trainer.is_fsdp_enabled:
    #     trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

if __name__ == "__main__":
    main()