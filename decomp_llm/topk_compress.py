import torch
import torch.nn as nn
from decomp_linear import SVDLinear, TrainableDecompLinear
from evaluate_utils import evaluate_model, evaluate_perplexity
from typing import List
import os
import logging
import time

logging.basicConfig(
    level=logging.DEBUG,  # 设置日志级别为 DEBUG，这样会记录所有级别的日志
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # 设置日志格式
    filename=str(int(time.time())) + "topk.log",  # 将日志写入到文件中
    filemode="w",  # 文件模式，'w' 表示每次运行程序时覆盖文件内容，'a' 表示追加
)

# 获取一个日志记录器
logger = logging.getLogger("topk_logger")


def test_sensitivity(model, layers, input_ids, delta_param_ratio, args, lm_head=True):
    for layer in layers:
        target_param_ratio = layer.param_ratio + delta_param_ratio
        if target_param_ratio <= 0.05:
            print("target ratio < 0")
            target_param_ratio = 0
            layer.sensitivity_ppl = 999
        else:
            if isinstance(layer, SVDLinear):
                svd_linear = layer.change_param_ratio(target_param_ratio)
            else:
                svd_linear = SVDLinear.from_linear(
                    layer,
                    param_ratio=target_param_ratio,
                    alpha=args.alpha,
                    act_aware=True,
                    rank_align=args.rank_align,
                )
            father = layer._father[0]
            name = layer._name
            setattr(father, name, svd_linear)
            ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
            layer.sensitivity_ppl = ppl
            setattr(father, name, layer)
            del svd_linear
    return layers


def get_topk_sensitivity_layers(layers, topk):
    # sort layers by sensitivity_ppl
    layers = sorted(layers, key=lambda x: x.sensitivity_ppl)
    topk_layers = layers[:topk]
    return topk_layers


def transform_topk_layers(topk_layers: List[SVDLinear], delta_param_ratio, args):
    for layer in topk_layers:
        target_param_ratio = layer.param_ratio + delta_param_ratio
        if target_param_ratio < 0:
            target_param_ratio = 0.01
        if isinstance(layer, nn.Linear):
            svd_linear = SVDLinear.from_linear(
                layer,
                param_ratio=target_param_ratio,
                alpha=args.alpha,
                act_aware=True,
                rank_align=args.rank_align,
            )
        else:
            svd_linear = layer.change_param_ratio(target_param_ratio)
        logger.info(f"update {layer._name} ratio {target_param_ratio}")
        father = layer._father[0]
        name = layer._name
        svd_linear._father = layer._father
        svd_linear._name = layer._name
        setattr(father, name, svd_linear)
        if isinstance(layer, nn.Linear):
            del layer.weight


def load_from_cache(model, param_ratio_dict, args):
    # get full_name
    full_name_dict = {module: name for name, module in model.named_modules()}
    model.eval()
    layers = get_layers(model, [])

    for layer in layers:
        target_param_ratio = layer.param_ratio
        if isinstance(layer, nn.Linear):
            svd_linear = SVDLinear.from_linear(
                layer,
                param_ratio=target_param_ratio,
                alpha=args.alpha,
                act_aware=True,
                rank_align=args.rank_align,
            )
        else:
            svd_linear = layer.change_param_ratio(target_param_ratio)
        father = layer._father[0]
        name = layer._name
        setattr(father, name, svd_linear)
    return layers


def get_layers(module, layer_list):
    for name, child in module.named_children():
        if "lm_head" in name:
            print("skipped ", name)
            continue
        elif isinstance(child, (nn.Linear)):
            layer_list.append(child)
            child.param_ratio = 1.0
            child._father = [module]
            child._name = name
        elif isinstance(child, SVDLinear):
            layer_list.append(child)
        else:
            get_layers(child, layer_list)
    return layer_list


def iteratively_topk_sensitivity_compress(model, calib_loader, delta_param_ratio, topk, args, use_cache=False):
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/','_')}_sensitivity_{args.scaling_method}_{args.alpha}_{args.n_calib_samples}_{args.calib_dataset}_top{topk}_delta{-delta_param_ratio}_.pt"
    if use_cache:
        if os.path.exists(cache_file) and use_cache:
            param_ratio_dict = torch.load(cache_file, map_location="cpu")
            # layers = load_from_cache(model, param_ratio_dict, args)
            
    else:
        model.eval()
        layers = []

        input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
        print(f"calibration input_ids.shape={input_ids.shape}")

    model_param_ratio_target = args.param_ratio_target

    raw_tot_params = sum([_.numel() for _ in model.parameters()])
    iter_count = 0
    while True:
        print(f"start {iter_count} round update ...")
        logger.info(f"start {iter_count} round update ...") 
        layers = get_layers(model, [])
        layers = test_sensitivity(model, layers, input_ids, delta_param_ratio, args)
        topk_layers = get_topk_sensitivity_layers(layers, topk)
        print(f"===== compress layers {[_._name for _ in topk_layers]} =====")
        transform_topk_layers(topk_layers, delta_param_ratio, args)
        now_model_params = sum([_.numel() for _ in model.parameters()])

        print(
            f"finish {iter_count} round compress, now_model_params={now_model_params}, raw_tot_params={raw_tot_params}, ratio={now_model_params / raw_tot_params}"
        )
        ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)

        logger.info(
            f"finish {iter_count} round compress, current ppl {ppl}, now_model_params={now_model_params}, raw_tot_params={raw_tot_params}, ratio={now_model_params / raw_tot_params}"
        )
        iter_count += 1
        if now_model_params / raw_tot_params < model_param_ratio_target:
            break

    full_name_dict = {module: name for name, module in model.named_modules()}
    param_ratio_dict = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                param_ratio_dict[full_name] = 1
            elif isinstance(raw_linear, SVDLinear):
                full_name = full_name_dict[raw_linear]
                param_ratio_dict[full_name] = raw_linear.param_ratio
            else:
                modules.append(raw_linear)

    torch.save(param_ratio_dict, cache_file)

    print("transform SVDLinear layers to TrainableDecompLinear layers")
    layers = get_layers(model, [])
    for layer in layers:
        if isinstance(layer, SVDLinear):
            if layer.ALinear.bias != None:
                bias = layer.ALinear.bias.data
            else:
                bias = None
            svd_linear = TrainableDecompLinear(
                layer.ALinear.weight.data,
                layer.BLinear.weight.data,
                layer.truncation_rank,
                bias,
                args.train_frac_beta,
            )
            father = layer._father[0]
            name = layer._name
            setattr(father, name, svd_linear)

# 
def iteratively_topk_sensitivity_compress_with_recovery(model, calib_loader, delta_param_ratio, topk, args, use_cache=False):
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/','_')}_sensitivity_{args.scaling_method}_{args.alpha}_{args.n_calib_samples}_{args.calib_dataset}_top{topk}_delta{-delta_param_ratio}_.pt"
    if use_cache:
        if os.path.exists(cache_file) and use_cache:
            param_ratio_dict = torch.load(cache_file, map_location="cpu")
            layers = load_from_cache(model, param_ratio_dict, args)
    else:
        model.eval()
        layers = []

        input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
        print(f"calibration input_ids.shape={input_ids.shape}")

    model_param_ratio_target = args.param_ratio_target

    raw_tot_params = sum([_.numel() for _ in model.parameters()])
    iter_count = 0
    while True:
        print(f"start {iter_count} round update ...")
        logger.info(f"start {iter_count} round update ...")
        layers = get_layers(model, [])
        layers = test_sensitivity(model, layers, input_ids, delta_param_ratio, args)
        topk_layers = get_topk_sensitivity_layers(layers, topk)
        print(f"===== compress layers {[_._name for _ in topk_layers]} =====")
        transform_topk_layers(topk_layers, delta_param_ratio, args)
        now_model_params = sum([_.numel() for _ in model.parameters()])

        print(
            f"finish {iter_count} round compress, now_model_params={now_model_params}, raw_tot_params={raw_tot_params}, ratio={now_model_params / raw_tot_params}"
        )
        ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)

        logger.info(
            f"finish {iter_count} round compress, current ppl {ppl}, now_model_params={now_model_params}, raw_tot_params={raw_tot_params}, ratio={now_model_params / raw_tot_params}"
        )
        iter_count += 1
        if now_model_params / raw_tot_params < model_param_ratio_target:
            break

    full_name_dict = {module: name for name, module in model.named_modules()}
    param_ratio_dict = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                param_ratio_dict[full_name] = 1
            elif isinstance(raw_linear, SVDLinear):
                full_name = full_name_dict[raw_linear]
                param_ratio_dict[full_name] = raw_linear.param_ratio
            else:
                modules.append(raw_linear)

    torch.save(param_ratio_dict, cache_file)

    print("transform SVDLinear layers to TrainableDecompLinear layers")
    layers = get_layers(model, [])
    for layer in layers:
        if isinstance(layer, SVDLinear):
            if layer.ALinear.bias != None:
                bias = layer.ALinear.bias.data
            else:
                bias = None
            svd_linear = TrainableDecompLinear(
                layer.ALinear.weight.data,
                layer.BLinear.weight.data,
                layer.truncation_rank,
                bias,
                args.train_frac_beta,
            )
            father = layer._father[0]
            name = layer._name
            setattr(father, name, svd_linear)
