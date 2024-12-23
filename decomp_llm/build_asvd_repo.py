import sys

sys.path.append(".")
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM
from transformers.models.opt.configuration_opt import OPTConfig

# from evaluate_utils import evaluate_model
from decomp_llm.datautils import get_calib_data

# from decomp_llm.act_aware_utils import calib_input_distribution, calib_fisher_info
from sensitivity import (
    calib_sensitivity_ppl,
    calib_sensitivity_stable_rank,
    calib_sensitivity_ppl_greedy,
    calib_sensitivity_ppl_gradually_update,
#    calib_sensitivity_ppl_hadamard,
#     calib_sensitivity_ppl_hadamard_both_side,
)
from topk_compress import iteratively_topk_sensitivity_compress

# from quantization import rtn_quant_sequential
from binary_search import binary_search_truncation_rank, naive_decomp
from decomp_linear import SVDLinear, TrainableDecompLinear
import os
from modelutils import profile_svdllm_low_resource
from decomp_llm.evaluate_utils import evaluate_model


def main(args):
    model_id = args.model_id

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )

    # sensitivity calibration
    calib_loader = get_calib_data(args.calib_dataset, tokenizer, model_id, 128)
    if args.hadamard or args.hadamard_both:
        print("skip whitening")
    else:
        if args.whitening_profiling_path is not None:
            profiling_mat = torch.load(args.whitening_profiling_path, map_location="cpu")
        else:
            model.seqlen = 256
            profiling_mat = profile_svdllm_low_resource(args.model_id, model, calib_loader, args.DEV, args.norm, args.norm_coef)
            if args.save_path is not None:
                add_norm = ""
                if args.norm:
                    add_norm = "_is_norm"
                torch.save(
                    profiling_mat,
                    args.save_path
                    + "/"
                    + args.model_id.replace("/", "_").replace("-", "_")
                    + "_profiling_"
                    + args.calib_dataset
                    + "_"
                    + str(args.seed)
                    + add_norm
                    + ".pt",
                )
                print("finish whitening")
                exit(0)

            # profiling_mat = torch.load(
            #     args.save_path
            #     + "/"
            #     + args.model_id.replace("/", "_").replace("-", "_")
            #     + "_profiling_"
            #     + args.calib_dataset
            #     + "_"
            #     + str(args.seed)
            #     + add_norm
            #     + ".pt",
            #     map_location="cpu"
            # )

        layers = model.model.layers

        for i in range(len(layers)):
            layer = layers[i]
            for name, module in layer.named_modules():
                if name in profiling_mat[i]:
                    # print(f"Profiling matrix found for {name}")
                    if args.norm:
                        whitening_matrix = profiling_mat[i][name][0]
                        scaling_diag_matrix2 = profiling_mat[i][name][1]
                        module.whitening_matrix = whitening_matrix
                        module.scaling_diag_matrix2 = scaling_diag_matrix2
                    else:
                        whitening_matrix = profiling_mat[i][name]
                        module.whitening_matrix = whitening_matrix


    if args.naive_truncation:
        naive_decomp(model, args)
    else:
        if args.sensitivity_metric == "ppl" and not args.hadamard and not args.hadamard_both:
            sensitivity = calib_sensitivity_ppl(model, calib_loader, args, args.use_cache)
        # elif args.sensitivity_metric == "stable_rank":
        #     sensitivity = calib_sensitivity_stable_rank(model, calib_loader, args, args.use_cache)
        # if args.sensitivity_metric == "ppl_greedy":
        #    sensitivity = calib_sensitivity_ppl_greedy(model, calib_loader, args, args.use_cache)
        elif args.sensitivity_metric == "topk":
            iteratively_topk_sensitivity_compress(model, calib_loader, args.delta, args.topk, args, use_cache=False)
        elif args.hadamard:
            print("start hadamard transform")
            raise NotImplementedError
        elif args.hadamard_both:
            print("start hadamard transform both side")
            raise NotImplementedError
        # search best truncation rank for each layer
        if args.sensitivity_metric != "topk":
            binary_search_truncation_rank(model, sensitivity, calib_loader, args)

    # build huggingface model
    # assert args.act_aware
    assert args.alpha == 0.5
    # assert args.calib_dataset == "c4"
    # assert args.scaling_method == "abs_mean"
    # assert args.sensitivity_metric == "ppl"
    # assert args.use_cache
    assert args.weight_quant == "none"
    assert not args.eval_mmlu

    save_path = (
        "huggingface_repos/"
        + model_id.split("/")[-1]
        + f"-{args.calib_dataset}"
        + f"-{args.train_frac_beta}"
        + f"-param-ratio-{args.param_ratio_target}"
    )
    if args.hadamard:
        save_path = save_path + "-hadamard"
    elif args.hadamard_both:
        save_path = save_path + "-hadamard_both"
    if args.sensitivity_metric == "topk":
        save_path = save_path + f"-top{args.topk}_delta{-args.delta}"
    if args.naive_truncation:
        save_path = save_path + "-naive_truncation"
    print("model_id: ", model_id)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    config = model.config.to_dict()
    config["truncation_ranks"] = {}
    for name, module in model.named_modules():
        if isinstance(module, TrainableDecompLinear):
            config["truncation_ranks"][name] = module.truncation_rank
    if "opt" in model_id:
        config["auto_map"] = {
            "AutoConfig": "configuration_asvd_opt.ASVDOPTConfig",
            "AutoModelForCausalLM": "modeling_asvd_opt.ASVDOPTForCausalLM",
        }
        config["architectures"] = ["ASVDOPTForCausalLM"]
        os.system(
            "cp ./huggingface_repos/configuration_asvd_opt.py ./huggingface_repos/modeling_asvd_opt.py ./" + save_path
        )
    elif "llama" in model_id or "Llama" in model_id:
        # config["auto_map"] = {
        #     "AutoConfig": "configuration_asvd_llama.ASVDLlamaConfig",
        #     "AutoModelForCausalLM": "modeling_asvd_llama.ASVDLlamaForCausalLM",
        # }
        # config["architectures"] = ["ASVDLlamaForCausalLM"]
        os.system(
            "cp ./config_variant/concat_asvd/configuration_llama.py ./config_variant/concat_asvd/modeling_llama.py ./"
            + save_path
        )
    elif "Qwen" in model_id:
        config["auto_map"] = {
            "AutoConfig": "configuration_asvd_qwen2.ASVDQwen2Config",
            "AutoModelForCausalLM": "modeling_asvd_qwen2.ASVDQwen2ForCausalLM",
        }
        config["architectures"] = ["ASVDQwen2ForCausalLM"]
        os.system(
            "cp ./huggingface_repos/configuration_asvd_qwen2.py ./huggingface_repos/modeling_asvd_qwen2.py ./"
            + save_path
        )
    elif "phi" in model_id:
        config["auto_map"] = {
            "AutoConfig": "configuration_asvd_phi.ASVDPhiConfig",
            "AutoModelForCausalLM": "modeling_asvd_phi.ASVDPhiForCausalLM",
        }
        config["architectures"] = ["ASVDPhiForCausalLM"]
        os.system(
            "cp ./huggingface_repos/configuration_asvd_phi.py ./huggingface_repos/modeling_asvd_phi.py ./" + save_path
        )
    import json

    json.dump(config, open(save_path + "/config.json", "w"), indent=2)

    print("Done building huggingface model")

    evaluate_model(model, tokenizer, model_id, "", "wikitext2")

    del model
    del tokenizer
    # if args.push:
    #     # load
    #     hub_name = model_id.split("/")[-1] + f"-asvd{int(args.param_ratio_target*100)}"
    #     tokenizer = AutoTokenizer.from_pretrained(save_path, trust_remote_code=True)
    #     model = AutoModelForCausalLM.from_pretrained(
    #         save_path,
    #         device_map="cpu",
    #         torch_dtype=torch.float16,
    #         trust_remote_code=True,
    #     )
    #     tokenizer.push_to_hub(hub_name)
    #     model.push_to_hub(hub_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-1.3b",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--save_path",
        type=str,
    )
    parser.add_argument(
        "--whitening_profiling_path",
        type=str,
    )
    parser.add_argument(
        "--ppl_target",
        type=float,
        default=-1,
        help="target ppl",
    )
    parser.add_argument(
        "--param_ratio_target",
        type=float,
        default=-1,
        help="target param ratio",
    )
    parser.add_argument(
        "--act_aware",
        action="store_true",
        help="use act aware svd (ASVD)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="hyper-parameter alpha for ASVD",
    )
    parser.add_argument(
        "--n_calib_samples",
        type=int,
        default=32,
        help="number of samples used for calibration",
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="c4",
        choices=["wikitext2", "c4", "ptb", "alpaca", "alpaca_bi_chat"],
        help="calibration dataset",
    )
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="abs_mean",
        choices=["abs_mean", "abs_max", "fisher", "fisher_abs_mean"],
        help="scaling method",
    )
    parser.add_argument(
        "--sensitivity_metric",
        type=str,
        default="ppl",
        choices=["ppl", "stable_rank", "topk"],
        help="search metric",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="use cached calibration results",
    )
    parser.add_argument(
        "--weight_quant",
        type=str,
        default="none",
        choices=["none", "rtn_int8", "rtn_int6"],
        help="weight quantization method",
    )
    parser.add_argument(
        "--eval_mmlu",
        action="store_true",
        help="evaluate mmlu",
    )
    parser.add_argument(
        "--sigma_fuse",
        type=str,
        default="UV",
        help="sigma fuse method",
        choices=["U", "V", "UV"],
    )
    # parser.add_argument(
    #     "--push",
    #     action="store_true",
    #     help="push to hub",
    # )

    parser.add_argument(
        "--compress_kv_cache",
        action="store_true",
        help="compress kv cache by asvd for k_proj and v_proj",
    )
    parser.add_argument(
        "--rank_align",
        type=int,
        default=1,
        help="align rank in SVD",
    )
    parser.add_argument("--DEV", type=str, default="cuda", help="device")
    parser.add_argument("--model_seq_len", type=int, default=2048, help="the default sequence length of the LLM")
    # parser.add_argument(
    #     "--ppl_thres",
    #     type=float,
    #     default=10,
    #     help="increase in ppl",
    # )
    parser.add_argument("--seed", type=int, default=233, help="seed")
    parser.add_argument("--train_frac_beta", type=float, default=0.2, help="trainable ratio")
    parser.add_argument("--hadamard", type=bool, default=False, help="apply hadamard transform or not")
    parser.add_argument(
        "--hadamard_both", type=bool, default=False, help="apply hadamard transform on both side or not"
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--delta", type=float, default=-0.1)
    parser.add_argument("--naive_truncation", type=bool, default=False)
    parser.add_argument("--norm", type=bool, default=False)
    parser.add_argument("--norm_coef", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
