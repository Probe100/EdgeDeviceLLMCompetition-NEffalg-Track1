# EdgeDeviceLLMCompetition-NICSEffalg
EdgeDeviceLLMCompetition final submission

## Submission Checklist

### The configuration and checkpoints of the original HuggingFace model 

Llama 3.1: [p2o6e100/ASVDLlama3.1-4B-Instruct-v1 · Hugging Face](https://huggingface.co/p2o6e100/ASVDLlama3.1-4B-Instruct-v1)

Qwen2: [p2o6e100/ASVDQwen2-4B-Instruct-v1 · Hugging Face](https://huggingface.co/p2o6e100/ASVDQwen2-4B-Instruct-v1)

Phi2: [p2o6e100/Phi2-lm-pruned-v1 · Hugging Face](https://huggingface.co/p2o6e100/Phi2-lm-pruned-v1)

### Code for converting the model to MLC

google drive link

### Converted MLC model files

Llama 3.1: [p2o6e100/ASVDLlama3.1-4B-Instruct-v1-q0f16-MLC · Hugging Face](https://huggingface.co/p2o6e100/ASVDLlama3.1-4B-Instruct-v1-q0f16-MLC)

Qwen2: [p2o6e100/ASVDQwen2-4B-Instruct-v1-q0f16-MLC · Hugging Face](https://huggingface.co/p2o6e100/ASVDQwen2-4B-Instruct-v1-q0f16-MLC)

Phi2: [p2o6e100/Phi2-lm-pruned-v1 · Hugging Face](https://huggingface.co/p2o6e100/Phi2-lm-pruned-v1-q0f16-MLC)

### APK file

Google drive option: [Link](https://drive.google.com/file/d/1eqA2wtcTD0xq6tLOng5gczoE8nymLPa1/view?usp=sharing)

OneDrive option: [Link](https://1drv.ms/u/c/f545d3fc68499cef/EbcYsl2TdExChpHyamvvVgYBMSTmo_ymqbBQrtGuNK3abQ?e=quUnIG)

### Script to package the MLC model file into the APK

Google drive option: [Link](https://drive.google.com/file/d/1llJfNneBJBtipt1EQww8XMEMZ2qadzyJ/view?usp=sharing)

OneDrive option: [Link](https://1drv.ms/u/c/f545d3fc68499cef/EUMG3dkTXnNJsVQg9dV-XCMByIqQdhILWoC8sisrOYQKbw?e=mmnCx4)

### Screenshot

Check `screenshot_llama3_1.jpg`, `screenshot_qwen2.jpg`, and `screenshot_phi2.jpg` in `assets` folder

### CSV file

Check Result.csv

## Compression Instruction

### Final Recipe

Iterative pruning that combine LMHead pruning, width pruning and weight decomposition

### Dataset construction

Construct a chat format English & Chinese alpaca dataset

`python chat_dataset_construct.py`

### Llama3.1-8B-Instruct Compression

#### LMHead Pruning

Reduce vocab size by 40% and prune lm_head accordingly

`python tokenizer_shrink.py --model_id Meta-Llama-3.1-8B-Instruct --drop_ratio 0.4 --save_path Llama-3.1-lm-pruned-0.6`

Manually change the bos_token_id and eos_token_id in config.py and generation_config.py. Add pad_token_id in genration_config.py

Recovery finetuning

`CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch decomp_llm/recovery_finetune_chat.py --model Llama-3.1-lm-pruned-0.6 --output_dir Llama-3.1-lm-pruned-0.6-ft --use_lora True`

Choose the model that perform best (we choose to evaluate only on human_eval for efficiency and the select model is the one with iteration = 3000)

Merge LoRA

`python lora_merge.py --model_id Llama-3.1-lm-pruned-0.6-ft/checkpoint-3000/ --save_path Llama-3.1-lm-pruned-0.6-ft3000`

#### Width Pruning

For width pruning we use a revised version of LLM-Pruner

`python llama3.py --base_model Llama-3.1-lm-pruned-0.6-ft3000  --save_ckpt_log_name llama3.1 --pruning_ratio 0.20 --pruner_type taylor --block_wise --block_attention_layer_start 0 --block_attention_layer_end 32 --block_mlp_layer_start 4 --block_mlp_layer_end 28 --device cuda --eval_device cuda --save_model --save_path Llama-3.1-lm-pruned-0.6-ft3000-width-pruned --calib_dataset alpaca_bi_chat`

Recovery finetuning

`CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch decomp_llm/recovery_finetune_chat.py --model Llama-3.1-lm-pruned-0.6-ft3000-width-pruned --output_dir Llama-3.1-lm-pruned-0.6-ft3000-width-pruned-ft --use_lora True`

Choose the model that perform best (we choose to evaluate only on human_eval for efficiency and the select model is the one with iteration = 3000)

Merge LoRA

`python lora_merge.py --model_id Llama-3.1-lm-pruned-0.6-ft3000-width-pruned-ft/checkpoint-3000/ --save_path Llama-3.1-lm-pruned-0.6-ft3000-width-pruned-ft3000`

#### Weight Decomposition

Concat the linear weight in attention layer (q_proj, k_proj, and v_proj) and ffn layer (gate_proj and up_proj)

`python linear_module_concat.py --model_id Llama-3.1-lm-pruned-0.6-ft3000-width-pruned-ft3000 --save_path Llama-3.1-lm-pruned-0.6-ft3000-width-pruned-ft3000-concat`

Weight decomposition

Generate whitening matrix

`CUDA_VISIBLE_DEVICES=0 python decomp_llm/build_asvd_repo.py --model_id Llama-3.1-lm-pruned-0.6-ft3000-width-pruned-ft3000-concat --save_path decomp_llm/profile_save --param_ratio_target 0.62 --train_frac_beta 0.1 --topk 5 --delta -0.15 --sensitivity_metric topk --calib_dataset alpaca_bi_chat --norm True`

Plan search & decomposition

`CUDA_VISIBLE_DEVICES=0 python decomp_llm/build_asvd_repo.py --model_id Llama-3.1-lm-pruned-0.6-ft3000-width-pruned-ft3000-concat --whitening_profiling_path decomp_llm/profile_save/Llama_3.1_lm_pruned_0.6_width_pruned_concat_profiling_alpaca_bi_chat_233_is_norm.pt --param_ratio_target 0.62 --train_frac_beta 0.1 --topk 5 --delta -0.15 --sensitivity_metric topk --calib_dataset alpaca_bi_chat --norm True`

Recovery finetuning

`CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch decomp_llm/recovery_finetune_chat.py --model huggingface_repos/Llama-3.1-lm-pruned-0.6-ft3000-width-pruned-a0m4-ft3000-concat-alpaca_bi_chat-0.15-param-ratio-0.62-top5_delta0.15 --output_dir Llama-3.1-lm-pruned-0.6-ft3000-width-pruned-a0m4-ft3000-concat-alpaca_bi_chat-0.15-param-ratio-0.62-top5_delta0.15-ft`

Choose the model that perform best (we choose to evaluate only on commonsenseQA for efficiency and the select model is the one with iteration = 7000)

Change layer format for higher throughput

`CUDA_VISIBLE_DEVICES=0 python combine_train_untrain.py --model_id Llama-3.1-lm-pruned-0.6-ft3000-width-pruned-a0m4-ft3000-concat-alpaca_bi_chat-0.15-param-ratio-0.62-top5_delta0.15-ft/checkpoint-7000 --save_path Llama-3.1-check`

*If encounter: TypeError: unsupported operand type(s) for //: 'list' and 'int':* Check model config.json and change "pruned_intermediate_size", "hidden_size", "pruned_hidden_size" from list to int

### Qwen2-7B-Instruct Compression

#### LMHead Pruning

Reduce vocab size by 30% and prune lm_head accordingly

`python tokenizer_shrink.py --model_id Qwen_Qwen2-7B-Instruct --drop_ratio 0.3 --save_path Qwen-2-lm-pruned-0.7`

Manually change the bos_token_id and eos_token_id in config.py and generation_config.py. Add pad_token_id in genration_config.py

#### Width Pruning

For width pruning we use a revised version of LLM-Pruner

`python qwen2.py --base_model Qwen-2-lm-pruned-0.7  --save_ckpt_log_name qwen2 --pruning_ratio 0.25 --pruner_type taylor --block_wise --block_attention_layer_start 0 --block_attention_layer_end 28 --block_mlp_layer_start 4 --block_mlp_layer_end 25 --device cuda --eval_device cuda --save_model --save_path Qwen-2-lm-pruned-0.7-width-pruned-a0m4 --calib_dataset alpaca_bi_chat`

Recovery finetuning

`CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch decomp_llm/recovery_finetune_chat.py --model Qwen-2-lm-pruned-0.7-width-pruned-a0m4 --output_dir Qwen-2-lm-pruned-0.7-width-pruned-a0m4-ft --use_lora True`

Choose the model that perform best (we choose to evaluate only on human_eval for efficiency and the selectef model is the one with iteration = 6506)

Merge LoRA

`python lora_merge.py --model_id Qwen-2-lm-pruned-0.7-width-pruned-a0m4-ft/checkpoint-6506/ --save_path Qwen-2-lm-pruned-0.7-width-pruned-a0m4-ft6506`

#### Weight Decomposition

Concat the linear weight in attention layer (q_proj, k_proj, and v_proj) and ffn layer (gate_proj and up_proj)

`python linear_module_concat.py --model_id Qwen-2-lm-pruned-0.7-width-pruned-a0m4-ft6506 --save_path Qwen-2-lm-pruned-0.7-width-pruned-a0m4-ft6506-concat`

Weight decomposition

Generate whitening matrix

`CUDA_VISIBLE_DEVICES=0 python decomp_llm/build_asvd_repo.py --model_id Qwen-2-lm-pruned-0.7-width-pruned-a0m4-ft6506-concat --save_path decomp_llm/profile_save --param_ratio_target 0.67 --train_frac_beta 0.15 --topk 5 --delta -0.15 --sensitivity_metric topk --calib_dataset alpaca_bi_chat --norm True`

Plan search & decomposition

`CUDA_VISIBLE_DEVICES=0 python decomp_llm/build_asvd_repo.py --model_id Qwen-2-lm-pruned-0.7-width-pruned-a0m4-ft6506-concat --whitening_profiling_path decomp_llm/profile_save/Llama_3.1_lm_pruned_0.6_width_pruned_concat_profiling_alpaca_bi_chat_233_is_norm.pt --param_ratio_target 0.67 --train_frac_beta 0.1 --topk 5 --delta -0.15 --sensitivity_metric topk --calib_dataset alpaca_bi_chat --norm True`

Recovery finetuning

`CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch decomp_llm/recovery_finetune_chat.py --model huggingface_repos/Qwen-2-lm-pruned-0.7-width-pruned-a0m4-ft6506-concat-alpaca_bi_chat-0.15-param-ratio-0.67-top5_delta0.15 --output_dir Qwen-2-lm-pruned-0.7-width-pruned-a0m4-ft6506-concat-alpaca_bi_chat-0.15-param-ratio-0.67-top5_delta0.15-ft`

Choose the model that perform best (we choose to evaluate only on commonsenseQA for efficiency and the select model is the one with iteration = 12000)

Change layer format for higher throughput

`CUDA_VISIBLE_DEVICES=0 python combine_train_untrain.py --model_id Qwen-2-lm-pruned-0.7-width-pruned-a0m4-ft6506-concat-alpaca_bi_chat-0.15-param-ratio-0.67-top5_delta0.15-ft/checkpoint-12000 --save_path ASVDQwen2-4B-Instruct-v1`

*If encounter: TypeError: unsupported operand type(s) for //: 'list' and 'int':* Check model config.json and change "pruned_intermediate_size", "hidden_size", "pruned_hidden_size" from list to int

### Phi2

`python tokenizer_shrink.py --model_id phi2 --drop_ratio 0.05 --save_path Phi-2-lm-pruned-v2`

## Compile Instruction

### Model Download

Dowload model from huggingface and place it into ./dist/models folder

Llama 3.1: [p2o6e100/ASVDLlama3.1-4B-Instruct-v1 · Hugging Face](https://huggingface.co/p2o6e100/ASVDLlama3.1-4B-Instruct-v1)

Qwen2: [p2o6e100/ASVDQwen2-4B-Instruct-v1 · Hugging Face](https://huggingface.co/p2o6e100/ASVDQwen2-4B-Instruct-v1)

Phi2: [p2o6e100/Phi2-lm-pruned-v1 · Hugging Face](https://huggingface.co/p2o6e100/Phi2-lm-pruned-v1)

### Weight Conversion and Compilation

#### Llama 3.1

```shell
MODEL_NAME=decomp_llama

MODEL_TYPE=decomp_llama

mlc_llm convert_weight --model-type ${MODEL_TYPE} ./dist/models/${MODEL_NAME}/ --quantization q0f16 -o dist/$MODEL_NAME/ 

mlc_llm gen_config --model-type ${MODEL_TYPE} ./dist/models/${MODEL_NAME}/ --quantization q0f16 --conv-template llama-3_1 --prefill-chunk-size 1024 --context-window-size 1024 --max-batch-size 1 -o dist/${MODEL_NAME}/
```

After gen_config, go to `.dist/models/decomp_llama/mlc-chat-config.json` and set `"system_prefix_token_ids"` under `"conv_template"` as bos_token_id, `"stop_token_ids"` as eos_token_id.

Then, compile the model

```shell
mlc_llm compile --model-type ${MODEL_TYPE} dist/${MODEL_NAME}/mlc-chat-config.json --device android -o ./dist/libs/${MODEL_NAME}-android.tar
```

#### Qwen2

```shell
MODEL_NAME=decomp_qwen2

MODEL_TYPE=decomp_qwen2

mlc_llm convert_weight --model-type ${MODEL_TYPE} ./dist/models/${MODEL_NAME}/ --quantization q0f16 -o dist/$MODEL_NAME/ 

mlc_llm gen_config --model-type ${MODEL_TYPE} ./dist/models/${MODEL_NAME}/ --quantization q0f16 --conv-template qwen2 --prefill-chunk-size 1024 --context-window-size 1024 --max-batch-size 1 -o dist/${MODEL_NAME}/
```

After gen_config, go to `.dist/models/decomp_llama/mlc-chat-config.json` and set `"system_prefix_token_ids"` under `"conv_template"` as bos_token_id, `"stop_token_ids"` as eos_token_id and set `"stop_str"` as `[]`

Then, compile the model

```shell
mlc_llm compile --model-type ${MODEL_TYPE} dist/${MODEL_NAME}/mlc-chat-config.json --device android -o ./dist/libs/${MODEL_NAME}-android.tar
```

#### Phi2

```shell
MODEL_NAME=phi2_lmpruned

MODEL_TYPE=phi

mlc_llm convert_weight --model-type ${MODEL_TYPE} ./dist/models/${MODEL_NAME}/ --quantization q0f16 -o dist/$MODEL_NAME/ 

mlc_llm gen_config --model-type ${MODEL_TYPE} ./dist/models/${MODEL_NAME}/ --quantization q0f16 --conv-template phi-2 --prefill-chunk-size 1024 --context-window-size 1024 --max-batch-size 1 -o dist/${MODEL_NAME}/
```

After gen_config, go to `.dist/models/decomp_llama/mlc-chat-config.json` and set `"stop_token_ids"` as eos_token_id

Then, compile the model

```shell
mlc_llm compile --model-type ${MODEL_TYPE} dist/${MODEL_NAME}/mlc-chat-config.json --device android -o ./dist/libs/${MODEL_NAME}-android.tar
```

### App Compile

go to ./android/MLCChat, replace mlc-package-config.json with the one we provided

```shell
mlc_llm package
```

### APK Generation

Use android studio Build/Generation Signed App Bundle/APK to generate apk

### Bundle Weight

use python bundle_weight.py --apk-path app/release/app-release.apk to transfer weight from computer to mobile phone

Our `bundle_weight.py` is the same as the original one so either script works. Our `bundle_weight.py` can be found in `assets` folder
