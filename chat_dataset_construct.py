from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

dataset = load_dataset("silk-road/alpaca-data-gpt4-chinese")

def conversation_transform(instruction, input, output):
    if input == "":
        conversation = {"messages": [{"role": "system", "content": "You are a helpful AI assistant"}, 
                                    {"role": "user", "content": instruction}, 
                                    {"role": "assistant", "content": output}]}
    else:
        conversation = {"messages": [{"role": "system", "content": "You are a helpful AI assistant"}, 
                                    {"role": "user", "content": instruction + input}, 
                                    {"role": "assistant", "content": output}]}
    return conversation

data_list = []
for i in range(dataset['train'].num_rows):
    instruction = dataset['train'][i]['instruction_zh']
    input = dataset['train'][i]['input_zh']
    output = dataset['train'][i]['output_zh']
    data_list.append(conversation_transform(instruction, input, output))
    instruction = dataset['train'][i]['instruction']
    input = dataset['train'][i]['input']
    output = dataset['train'][i]['output']
    data_list.append(conversation_transform(instruction, input, output))
    
with open('alpaca_conversation.jsonl', 'w') as outfile:
    for entry in data_list:
        json.dump(entry, outfile)