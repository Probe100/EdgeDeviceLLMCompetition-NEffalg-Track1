from transformers import AutoModelForCausalLM, AutoTokenizer
from decomp_llm.decomp_linear import TrainableDecompLinear, SVDLinear
from tqdm import tqdm
import time
import argparse
import torch
import json

def main(args):
    prompt = [{'role': 'system', 'content': 'You are a helpful AI assistant'},
              {'role': 'user', 'content': 'What day is today？'}]
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True).to('cuda', torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    input_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to('cuda')
    print("model output")
    outputs = model.generate(input_ids, max_new_tokens=256) 
    print(tokenizer.decode(outputs[0]))
    print("finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="",
        help="Pretrained model ID",
    )
    args = parser.parse_args()
    main(args)