import os
import torch
import torch.nn as nn
from decomp_linear import SVDLinear
from evaluate_utils import evaluate_model, evaluate_perplexity
from tqdm import tqdm
import numpy as np


@torch.no_grad()
def calib_sensitivity_ppl(model, calib_loader, args, use_cache=True, lm_head=True):
    model_id = model.config._name_or_path
    norm = ""
    if args.norm:
        norm = "_is_norm"
    cache_file = f"cache/{model_id.replace('/','_')}_sensitivity_{args.scaling_method}_{args.alpha}_{args.n_calib_samples}_{args.calib_dataset}{norm}.pt"
    if os.path.exists(cache_file) and use_cache:
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict
    model.eval()

    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if lm_head and "lm_head" in name:
                continue
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_dict = {}
    if args.compress_kv_cache:
        param_ratio_candidates = [0.1 * i for i in range(1, 20)]
    else:
        param_ratio_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    print(f"input_ids.shape={input_ids.shape}")
    pbar = tqdm(total=len(linear_info) * len(param_ratio_candidates))
    for raw_linear, info in linear_info.items():
        sensitivity_dict[info["full_name"]] = {}
        raw_linear.is_calibration_stage = True
        for param_ratio in param_ratio_candidates:
            svd_linear = SVDLinear.from_linear(
                raw_linear,
                param_ratio=param_ratio,
                alpha=args.alpha,
                act_aware=True,
                rank_align=args.rank_align,
            )
            setattr(info["father"], info["name"], svd_linear)

            ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
            sensitivity_dict[info["full_name"]][param_ratio] = ppl
            print(f"{info['full_name']} {param_ratio} {ppl}")
            pbar.update(1)
        raw_linear.is_calibration_stage = False
        raw_linear.cached_svd = None
        setattr(info["father"], info["name"], raw_linear)
    torch.save(sensitivity_dict, cache_file)
    return sensitivity_dict


@torch.no_grad()
def calib_sensitivity_stable_rank(model, calib_loader, args, use_cache=True):
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/','_')}_sensitivity_stable_rank_{args.scaling_method}_{args.alpha}_{args.n_calib_samples}_{args.calib_dataset}.pt"
    if os.path.exists(cache_file) and use_cache:
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict
    model.eval()

    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_dict = {}
    param_ratio_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    print(f"input_ids.shape={input_ids.shape}")
    pbar = tqdm(total=len(linear_info) * len(param_ratio_candidates))
    for raw_linear, info in linear_info.items():
        sensitivity_dict[info["full_name"]] = {}

        # stable rank is defined to be the ratio between squared Frobenius norm and the squared spectral norm of a matrix
        w = raw_linear.weight
        w = w  # *raw_linear.scaling_diag_matrix.view(1,-1)**args.alpha
        w_fro = torch.norm(w, p="fro") ** 2
        _, singular_values, _ = torch.svd(w.float(), compute_uv=False)
        spectral_norm = torch.max(singular_values)
        w_spec = spectral_norm**2
        sr = (w_fro / w_spec) ** 0.5

        for param_ratio in param_ratio_candidates:
            sensitivity_dict[info["full_name"]][param_ratio] = -sr * param_ratio**0.1
            pbar.update(1)
    torch.save(sensitivity_dict, cache_file)
    return sensitivity_dict


@torch.no_grad()
def calib_sensitivity_ppl_greedy(model, calib_loader, args, use_cache=True, lm_head=True):
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/','_')}_sensitivity_{args.scaling_method}_{args.alpha}_{args.n_calib_samples}_{args.calib_dataset}_greedy_{args.ppl_thres}.pt"
    if os.path.exists(cache_file) and use_cache:
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict
    model.eval()

    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    module_count = 0
    while len(modules) > 0:
        submodule = modules.pop(0)
        for name, raw_linear in submodule.named_children():
            if lm_head and "lm_head" in name:
                continue
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                print(full_name)
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
                module_count += 1
            else:
                modules.append(raw_linear)

    sensitivity_dict = {}

    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    print(f"input_ids.shape={input_ids.shape}")
    pbar = tqdm(total=len(linear_info))
    module_thres = args.ppl_thres / module_count
    search_limit = 4
    tot_params = 0
    compress_params = 0
    start_ppl = init_ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
    print(f"Start ppl: {start_ppl}")
    for raw_linear, info in linear_info.items():
        ratio_low = 0
        ratio_high = 1
        if args.compress_kv_cache:
            ratio_high = 2
        ratio_mid = (ratio_high + ratio_low) / 2
        sensitivity_dict[info["full_name"]] = {}
        raw_linear.is_calibration_stage = True
        init_ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
        for i in range(search_limit):
            svd_linear = SVDLinear.from_linear(
                raw_linear,
                param_ratio=ratio_mid,
                alpha=args.alpha,
                act_aware=True,
                rank_align=args.rank_align,
            )
            setattr(info["father"], info["name"], svd_linear)

            ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
            # abs value
            if ppl - init_ppl < module_thres:
                ratio_high = ratio_mid
            else:
                ratio_low = ratio_mid
            ratio_mid = (ratio_high + ratio_low) / 2
            del svd_linear
        pbar.update(1)
        print(info["full_name"])
        print(f"Ratio: {ratio_high}")
        svd_linear = SVDLinear.from_linear(
            raw_linear,
            param_ratio=ratio_high,
            alpha=args.alpha,
            act_aware=True,
            rank_align=args.rank_align,
        )
        tot_params += raw_linear.weight.numel()
        compress_params += raw_linear.weight.numel() * ratio_high
        raw_linear.is_calibration_stage = False
        raw_linear.cached_svd = None
        raw_linear.to("cpu")
        setattr(info["father"], info["name"], svd_linear)
        sensitivity_dict[info["full_name"]] = ratio_high
        del svd_linear
        torch.cuda.empty_cache()
        # raw_linear.is_calibration_stage = False
        # raw_linear.cached_svd = None
        # setattr(info["father"], info["name"], raw_linear)
    print(f"Total params: {tot_params}")
    print(f"After compression params: {compress_params}")
    print(f"Total compression ratio for thres {args.ppl_thres}: {compress_params / tot_params}")
    torch.save(sensitivity_dict, cache_file)
    return sensitivity_dict


@torch.no_grad()
def calib_sensitivity_ppl_gradually_update(
    model, calib_loader, args, use_cache=True, lm_head=True, topk=10, update_limit=30, step=0.1
):
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/','_')}_sensitivity_{args.scaling_method}_{args.alpha}_{args.n_calib_samples}_{args.calib_dataset}_top{topk}.pt"
    if os.path.exists(cache_file) and use_cache:
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        module_dict = {name: module for name, module in model.named_modules()}
        # modify

        return sensitivity_dict
    model.eval()

    full_name_dict = {module: name for name, module in model.named_modules()}
    module_dict = {name: module for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    module_count = 0
    ratio_dict = {}
    while len(modules) > 0:
        submodule = modules.pop(0)
        for name, raw_linear in submodule.named_children():
            if lm_head and "lm_head" in name:
                continue
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                print(full_name)
                linear_info[raw_linear] = {"father": submodule, "name": name, "full_name": full_name}
                ratio_dict[full_name] = 1
                if args.compress_kv_cache:
                    linear_info[raw_linear]["ratio"] = 2
                module_count += 1
            else:
                modules.append(raw_linear)

    sensitivity_dict = {}

    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    print(f"input_ids.shape={input_ids.shape}")
    pbar = tqdm(total=len(linear_info) * update_limit)
    init_ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
    print("start_ppl: ", init_ppl)
    for i in range(update_limit):
        for raw_linear, info in linear_info.items():
            sensitivity_dict[info["full_name"]] = init_ppl
            raw_linear.is_calibration_stage = False
            svd_linear = SVDLinear.from_linear(
                raw_linear,
                param_ratio=ratio_dict[info["full_name"]] - step,
                alpha=args.alpha,
                act_aware=True,
                rank_align=args.rank_align,
            )
            setattr(info["father"], info["name"], svd_linear)

            ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
            sensitivity_dict[info["full_name"]] = ppl
            raw_linear.is_calibration_stage = False
            raw_linear.cached_svd = None
            setattr(info["father"], info["name"], raw_linear)
            pbar.update(1)
        topk_keys = sorted(sensitivity_dict, key=sensitivity_dict.get)[:topk]
        print(sensitivity_dict)
        # breakpoint()
        for k in topk_keys:
            ratio_dict[k] -= step
            print(k, ratio_dict[k])
            k.is_calibration_stage = True
            del k.cached_svd
            print(hasattr(k, "cached_svd"))
            if isinstance(module_dict[k], SVDLinear):
                module_dict[k].update_svdlinear(ratio_dict[k])
            else:
                svd_linear = SVDLinear.from_linear(
                    module_dict[k],
                    param_ratio=ratio_dict[k],
                    alpha=args.alpha,
                    act_aware=True,
                    rank_align=args.rank_align,
                )
            setattr(linear_info[k]["father"], linear_info[k]["name"], svd_linear)
    for raw_linear, r in sensitivity_dict.items():
        raw_linear.cached_svd = None
        raw_linear.to("cpu")
    print(f"Total compression ratio: {update_limit*step*topk/len(linear_info)}")
    torch.save(ratio_dict, cache_file)
    return

