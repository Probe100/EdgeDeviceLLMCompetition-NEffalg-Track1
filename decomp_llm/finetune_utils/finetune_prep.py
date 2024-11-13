from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig


def prompt_no_input(row):
    return ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n{output}").format_map(row)

def prompt_input(row):
    return ("Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}").format_map(row)

def create_alpaca_prompt(row):
    return prompt_no_input(row) if row["input"] == "" else prompt_input(row)

# prepare dataset for finetuning
# Input: name of dataset
# Output: train dataset and validation dataset
def get_dataset(dataset_name:str, val_size=0.05):
    dataset = load_dataset(dataset_name)
    if "alpaca-data-gpt4-chinese" in dataset_name:
        # dataset['train']
        dataset = dataset['train']
        combined_instruction = dataset['instruction'] + dataset['instruction_zh']
        combined_input = dataset['input'] + dataset['input_zh']
        combined_output = dataset['output'] + dataset['output_zh']
        dataset = Dataset.from_dict({'instruction':combined_instruction, 
                                      'input': combined_input,
                                      'output':combined_output})
        dataset = dataset.train_test_split(test_size=val_size, shuffle=True, seed=42)
    elif "alpaca" in dataset_name:
        # dataset['train']
        dataset = dataset['train'].train_test_split(test_size=val_size)
    return dataset['train'], dataset['test']

def config_prep(use_lora, output_dir="ft/"):
    if use_lora:
        training_args = SFTConfig(
            report_to="tensorboard",
            per_device_train_batch_size=1,
            learning_rate=2e-5,
            lr_scheduler_type="cosine",
            num_train_epochs=1,
            gradient_accumulation_steps=4, # simulate larger batch sizes
            output_dir=output_dir,
            # eval_strategy="steps",
            # eval_steps=1000,
            save_steps=1000,
            bf16=True,
            bf16_full_eval=True,
            save_only_model=True
        )
        
        peft_config = LoraConfig(
            r=8,  # the rank of the LoRA matrices
            lora_alpha=16, # the weight
            lora_dropout=0.1, # dropout to add to the LoRA layers
            bias="none", # add bias to the nn.Linear layers?
            task_type="CAUSAL_LM",
            target_modules="all-linear", # the name of the layers to add LoRA
            modules_to_save=None, # layers to unfreeze and train from the original pre-trained model
        )
    else:
        training_args = SFTConfig(
            report_to="tensorboard",
            per_device_train_batch_size=1,
            learning_rate=2e-5,
            lr_scheduler_type="cosine",
            num_train_epochs=2,
            gradient_accumulation_steps=4, # simulate larger batch sizes
            output_dir=output_dir,
            # eval_strategy="steps",
            # eval_steps=1000,
            save_steps=1000,
            bf16=True,
            bf16_full_eval=True,
            save_only_model=True
        )
        
        peft_config = None

    return training_args, peft_config