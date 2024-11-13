import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
from tokenizers import Tokenizer
import os.path
from tqdm import tqdm
import argparse


class VocabularyPruner(object):

    def check(self, old_model_name_or_path, new_model_name_or_path, text):
        max_length = 20

        # 使用老模型对文本编码
        old_model = AutoModelForCausalLM.from_pretrained(old_model_name_or_path, trust_remote_code=True)
        old_tokenizer = AutoTokenizer.from_pretrained(old_model_name_or_path)
        old_input_ids = old_tokenizer(text, return_tensors="pt").input_ids
        old_output = old_model.generate(old_input_ids, max_length=max_length)
        old_output_text = old_tokenizer.batch_decode(old_output)
        print("old_output:{}".format(old_output_text))

        # 使用新模型对文本编码
        new_model = AutoModelForCausalLM.from_pretrained(new_model_name_or_path, trust_remote_code=True)
        new_tokenizer = AutoTokenizer.from_pretrained(new_model_name_or_path)
        new_input_ids = new_tokenizer(text, return_tensors="pt").input_ids
        new_output = new_model.generate(new_input_ids, max_length=max_length)
        new_output_text = new_tokenizer.batch_decode(new_output)
        print("new_output:{}".format(new_output_text))

        if old_output_text == new_output_text:
            print("output is same, succeed to prune.")
        else:
            print("output is not same, fail to prune.")

    def update_embeddings(self, model, new2old_token_id, new_embeds, new_lm_head):
        for token_id, old_token_id in tqdm(new2old_token_id.items()):
            new_embeds.weight.data[token_id] = model.model.embed_tokens.weight.data[old_token_id]
            new_lm_head.weight.data[token_id] = model.lm_head.weight.data[old_token_id]
            if  model.lm_head.bias is not None:
                new_lm_head.bias.data[token_id] = model.lm_head.bias.data[old_token_id]
        model.model.embed_tokens.weight = new_embeds.weight
        model.lm_head.weight = new_lm_head.weight
        if  model.lm_head.bias is not None:
            model.lm_head.bias = new_lm_head.bias

    def prune(self, model_name_or_path, new_tokenizer_name_or_path, save_path, new_name_or_path=None):
        # 创建输出目录
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 加载新词表。如果是中文，就是中文的词表
        new_tokenizer = AutoTokenizer.from_pretrained(new_tokenizer_name_or_path)
        # 加载原词表。一般为多语言模型的词表
        old_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # 检查新词表是否为原词表的子集
        old_vocab = old_tokenizer.vocab
        new_vocab = new_tokenizer.vocab
        for token in tqdm(new_vocab.keys()):
            if token not in old_vocab:
                raise Exception("{} not exist".format(token))
        print("new_tokenizer is subset of old_tokenizer")

        # 获得新词表中每个token_id到原词表的token_id的映射
        new2old_token_id = {}
        for token, token_id in tqdm(new_vocab.items()):
            old_token_id = old_vocab[token]
            new2old_token_id[token_id] = old_token_id

        # 加载多语言模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True
        ).to("cuda")
        print(model)
        # 计算原模型的参数量
        old_params = sum(p.numel() for p in model.parameters())
        print("Total params of original model: %.2fM" % (old_params / 1e6))

        # 对于新词表中的每个token，取出其对应的权重，复制到新模型中
        vocab_size = len(new_tokenizer)
        hidden_size = model.config.hidden_size

        new_embeds = torch.nn.Embedding(vocab_size, hidden_size, dtype=model.dtype)
        if model.lm_head.bias is not None:
            has_bias = True
        else:
            has_bias = False
        new_lm_head = torch.nn.Linear(in_features=hidden_size, out_features=vocab_size, bias=has_bias, dtype=model.dtype)
        # 更新词表权重
        self.update_embeddings(model, new2old_token_id, new_embeds, new_lm_head)

        model.config.__dict__["vocab_size"] = vocab_size
        if new_name_or_path is not None:
            model.config.__dict__["_name_or_path"] = new_name_or_path

        # 计算新模型的参数量
        new_params = sum(p.numel() for p in model.parameters())
        print("Total params of new model : %.2fM" % (new_params / 1e6))

        print("词表缩小为原来的:{}%".format(round(len(new_tokenizer) / len(old_tokenizer), 4) * 100))
        print("模型参数量缩小为原来的:{}%".format(round(new_params / old_params, 4) * 100))
        model.save_pretrained(save_path)
        new_tokenizer.save_pretrained(save_path)


def main(args):
    mname = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(mname)
    keep_frac = 1 - args.drop_ratio

    vocab_keep_items = int(len(tokenizer.vocab.keys()) * keep_frac)
    assert tokenizer.is_fast, "This only works for fast tokenizers."
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    vocab = tokenizer_json["model"]["vocab"]
    if tokenizer_json["model"]["type"] == "BPE":
        new_vocab = {token: i for token, i in vocab.items() if i < vocab_keep_items}
        merges = tokenizer_json["model"]["merges"]
        new_merges = []
        for i in range(len(merges)):
            a, b = merges[i].split()
            new_token = "".join((a, b))
            if a in new_vocab and b in new_vocab and new_token in new_vocab:
                new_merges.append(merges[i])
        tokenizer_json["model"]["merges"] = new_merges
    elif tokenizer_json["model"]["type"] == "Unigram":
        new_vocab = vocab[:vocab_keep_items]
    elif tokenizer_json["model"]["type"] == "WordPiece" or tokenizer_json["model"]["type"] == "WordLevel":
        new_vocab = {token: i for token, i in vocab.items() if i < vocab_keep_items}
    else:
        raise ValueError(f"don't know how to handle {tokenizer_json['model']['type']}")
    tokenizer_json["model"]["vocab"] = new_vocab
    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))

    tokenizer.save_pretrained("pruned_tokenizer")

    pruner = VocabularyPruner()
    pruner.prune(mname, "pruned_tokenizer", args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--drop_ratio",
        type=float,
        default=0.4,
        help="proportion of word want to drop from the tokenizer",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="lm_head_pruned_model",
        help="save path",
    )
    args = parser.parse_args()
    main(args)
