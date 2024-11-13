import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import AutoModelForCausalLM, AutoTokenizer
from decomp_llm.decomp_linear import SVDLinear, TrainableDecompLinear
from tqdm import tqdm
import time
import argparse
import torch
import json
from config_variant.llm_pruner_asvd.modeling_llama import ASVDLinear
import os
def change_train_frac(model):
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            # print(name)
            # get all ASVDLinear in the model
            if hasattr(raw_linear, "ALinear_no_train"):
                full_name = full_name_dict[raw_linear]
                print(full_name)
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)
    st = time.time()
    for raw_linear, info in tqdm(linear_info.items()):
        # set ratio
        print("------------------")
        print(info["full_name"])
        print("------------------")
        svd_linear = SVDLinear.from_trainable_decomp_linear(
            raw_linear
        )
        if svd_linear.ALinear.bias != None:
                bias = svd_linear.ALinear.bias.data
        else:
            bias = None
        svd_linear = TrainableDecompLinear(
            svd_linear.ALinear.weight.data,
            svd_linear.BLinear.weight.data,
            svd_linear.truncation_rank,
            bias,
            args.train_frac_beta,
        )
        raw_linear.to("cpu")
        setattr(info["father"], info["name"], svd_linear)
        print(f"transform {info['full_name']}")
    ed = time.time()
    print(f"transform time: {ed-st}")
    return model


def main(args):
    prompt = [{'role': 'system', 'content': 'You are a helpful AI assistant'},
              {'role': 'user', 'content': 'Calculate the total surface area of a cube with a side length of 5 cm.'}]
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True).to('cuda', torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    input_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to('cuda')
    print("model output before merge")
    outputs = model.generate(input_ids, max_new_tokens=128) 
    print(tokenizer.decode(outputs[0]))
    model = change_train_frac(model)

    print("model output after merge")
    outputs = model.generate(input_ids, max_new_tokens=128) 
    print(tokenizer.decode(outputs[0]))
    print("save model...")    
    config = model.config.to_dict()

    for name, module in model.named_modules():
        if isinstance(module, TrainableDecompLinear):
            config["truncation_ranks"][name] = module.truncation_rank

    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    json.dump(config, open(args.save_path + "/config.json", "w"), indent=2)
    # os.system("cp ./config_variant/width_prune_asvd_eval/configuration_llama.py ./config_variant/width_prune_asvd_eval/modeling_llama.py ./" + args.save_path)

    print("finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="model_for_eval",
        help="model after combining train and no_train parts",
    )
    parser.add_argument(
        "--train_frac_beta",
        type=float,
        default=0.1,
        help="model after combining train and no_train parts",
    )
    args = parser.parse_args()
    main(args)