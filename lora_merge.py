import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default="", help='base model name')
    parser.add_argument('--save_path', type=str, default="", help='save path')
    args = parser.parse_args()

    config = PeftConfig.from_pretrained(args.model_id)
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
    lora_model = PeftModel.from_pretrained(base_model, args.model_id)
    print("Start weight merge...")
    merged_model = lora_model.merge_and_unload()
    print("Save model...")
    merged_model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print("LoRA weight merge finished")
