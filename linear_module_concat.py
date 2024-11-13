import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import json
import argparse


def model_transform(model, model_id, qkv_only):
    if "llama" in model_id.lower():
        bias = False
    elif "qwen" in model_id.lower():
        bias = True
    elif "phi" in model_id.lower():
        bias = True
    for i in range(model.config.num_hidden_layers):
        layer = model.model.layers[i]

        # combine q_proj, k_proj, v_proj
        qkv_weight = torch.cat(
            (
                layer.self_attn.q_proj.weight.data,
                layer.self_attn.k_proj.weight.data,
                layer.self_attn.v_proj.weight.data,
            ),
            dim=0,
        )
        qkv_proj = torch.nn.Linear(in_features=qkv_weight.size()[1], out_features=qkv_weight.size()[0], bias=bias)
        qkv_proj.weight.data = qkv_weight
        if bias:
            qkv_bias = torch.cat(
                (layer.self_attn.q_proj.bias.data, layer.self_attn.k_proj.bias.data, layer.self_attn.v_proj.bias.data),
                dim=0,
            )
            qkv_proj.bias.data = qkv_bias
        setattr(layer.self_attn, f"qkv_proj", qkv_proj)
        delattr(layer.self_attn, f"q_proj")
        delattr(layer.self_attn, f"k_proj")
        delattr(layer.self_attn, f"v_proj")

        if not qkv_only:
            # combine up_proj, gate_proj
            gate_up_weight = torch.cat((layer.mlp.gate_proj.weight.data, layer.mlp.up_proj.weight.data), dim=0)
            gate_up_proj = torch.nn.Linear(
                in_features=gate_up_weight.size()[1], out_features=gate_up_weight.size()[0], bias=False
            )
            gate_up_proj.weight.data = gate_up_weight
            setattr(layer.mlp, f"gate_up_proj", gate_up_proj)
            delattr(layer.mlp, f"gate_proj")
            delattr(layer.mlp, f"up_proj")
    return model


def model_save(model, model_id, tokenizer, save_path, qkv_only):
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    config = model.config.to_dict()
    folder_name = "concat"
    if qkv_only:
        folder_name = "qkv_" + folder_name

    if "llama" in model_id.lower():
        os.system(
            f"cp ./config_variant/{folder_name}/configuration_llama.py ./config_variant/{folder_name}/modeling_llama.py ./"
            + save_path
        )
        config["auto_map"] = {
            "AutoConfig": "configuration_llama.LlamaConfig",
            "AutoModelForCausalLM": "modeling_llama.LlamaForCausalLM",
        }
        config["architectures"] = ["LlamaForCausalLM"]
        json.dump(config, open(save_path + "/config.json", "w"), indent=2)
    elif "qwen" in model_id.lower():
        os.system(
            f"cp ./config_variant/{folder_name}/configuration_qwen2.py ./config_variant/{folder_name}/modeling_qwen2.py ./"
            + save_path
        )
        config["auto_map"] = {
            "AutoConfig": "configuration_qwen2.Qwen2Config",
            "AutoModelForCausalLM": "modeling_qwen2.Qwen2ForCausalLM",
        }
        config["architectures"] = ["Qwen2ForCausalLM"]
        json.dump(config, open(save_path + "/config.json", "w"), indent=2)
    elif "phi" in model_id.lower():
        os.system(
            f"cp ./config_variant/{folder_name}/configuration_phi.py ./config_variant/{folder_name}/modeling_phi.py ./"
            + save_path
        )
        config["auto_map"] = {
            "AutoConfig": "configuration_phi.PhiConfig",
            "AutoModelForCausalLM": "modeling_phi.PhiForCausalLM",
        }
        config["architectures"] = ["PhiForCausalLM"]
        json.dump(config, open(save_path + "/config.json", "w"), indent=2)



def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, trust_remote_code=True).to(
        "cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(model.device)
    prompt = [{'role': 'system', 'content': 'You are a helpful AI assistant'},
              {'role': 'user', 'content': 'What day is today？'}]
    input_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to(model.device)
    # Generate text
    output = model.generate(input_ids, max_new_tokens=256)
    # Decode the generated text character by character
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("prompt:")
    print(args.prompt)
    print("text generated before linear concat: ")
    print(generated_text)

    model = model_transform(model, args.model_id, args.qkv_only)
    model_save(model, args.model_id, tokenizer, args.save_path, args.qkv_only)

    # post-merge check
    model = AutoModelForCausalLM.from_pretrained(args.save_path, trust_remote_code=True, torch_dtype=torch.float16).to(
        "cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.save_path)
    # input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(model.device)
    input_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to(model.device)
    # Generate text
    output = model.generate(input_ids, max_new_tokens=256)
    # Decode the generated text character by character
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-1.3b",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--qkv_only",
        action="store_true",
        help="concat qkv only or not",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="Llama3.1_combined",
        help="save path of the model after concatenation",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="I'm starting with the man in the mirror, ",
        help="prompt for pre & post concatenation output",
    )
    args = parser.parse_args()
    main(args)
