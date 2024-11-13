import torch
import argparse
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, LlamaTokenizer
from decomp_llm.evaluate_utils import evaluate_model, evaluate_perplexity
from transformers.modeling_utils import no_init_weights

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
args = parser.parse_args()
model = AutoModelForCausalLM.from_pretrained(args.path, trust_remote_code=True, torch_dtype=torch.float16).to("cuda")
# model = AutoModelForCausalLM.from_pretrained(args.path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(args.path, trust_remote_code=True)
evaluate_model(model, tokenizer, "name", "", "wikitext2")