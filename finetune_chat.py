# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.



import json
import math
import logging
import os
import torch
import transformers
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from typing import Dict, Optional, List
from transformers import Trainer, GPTQConfig, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
from typing import (
    Any, 
    Callable, 
    NewType, 
    Optional, 
    Tuple, 
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/mnt/public/algm/models/Qwen1.5-7B")
    model_type: str = field(
        default="Qwen", metadata={"help": "Qwen or llama3"}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str


Dialog = Sequence[Message]

# DeepSpeed ZeRO 优化模型训练内存
def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# 部分错误修改
# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


# sources:[{"from":"user", "value": "...."}, {"from":"assistant", "value": "...."}]

def preprocess_Qwen(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        # system全部忽略
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)

        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            # user全部忽略
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            # assistant答案只保留句子
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)

        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)
    
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        pad_token_id = tokenizer.pad_token_id,
    )

def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"user": "<|start_header_id|>user", "assistant": "<|start_header_id|>assistant"}

    im_start = tokenizer.bos_token_id
    im_end = tokenizer.eos_token_id
    
    # llama3中 nl_tokens长度为2，所以要-4
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-4) + [im_end] + nl_tokens
        assert len(input_id) == len(target)

        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|start_header_id|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-4) + [im_end] + nl_tokens
            elif role == '<|start_header_id|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids)+1:-3] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        pad_token_id = tokenizer.pad_token_id,
    )

def preprocess_newllama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    def encode_header(tokenizer, message: Message) -> List[int]:
        tokens = []
        tokens.append(tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(tokenizer.encode(message["role"], bos=False, eos=False))
        tokens.append(tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode_message(tokenizer, message: Message) -> List[int]:
        tokens = encode_header(message)
        tokens.extend(
            tokenizer.encode(message["content"].strip(), bos=False, eos=False)
        )
        tokens.append(tokenizer.special_tokens["<|eot_id|>"])
        return tokens
    
    def encode_dialog_prompt(tokenizer, dialog: Dialog) -> List[int]:
        tokens = []
        tokens.append(tokenizer.special_tokens["<|begin_of_text|>"])
        for message in dialog:
            tokens.extend(encode_message(message))
        # Add the start of an assistant message for the model to complete.
        tokens.extend(encode_header({"role": "assistant", "content": ""}))
        return tokens
    
    _system_ids = tokenizer.encode(system_message, add_special_tokens=False)

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if source[0]["from"] != "user":
            source = source[1:]

        input_id, target = [], []
        #system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id = tokenizer.encode("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",add_special_tokens=False)
        input_id.extend(_system_ids)
        input_id.extend(tokenizer.encode("<|eot_id|>",add_special_tokens=False))

        target = tokenizer.encode("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",add_special_tokens=False) 
        target.extend([IGNORE_TOKEN_ID] * len(_system_ids)) 
        target.extend(tokenizer.encode("<|eot_id|>",add_special_tokens=False))
        
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            
            _user_ids = tokenizer.encode(sentence["from"],add_special_tokens=False)
            _value_ids = tokenizer.encode(sentence["value"],add_special_tokens=False)
            
            _input_id = tokenizer.encode("<|start_header_id|>",add_special_tokens=False) 
            _input_id.extend(_user_ids)
            _input_id.extend(tokenizer.encode("<|end_header_id|>",add_special_tokens=False))
            _input_id.extend(tokenizer.encode("\n\n",add_special_tokens=False))

            _input_id.extend(_value_ids)
            _input_id.extend(tokenizer.encode("<|eot_id|>",add_special_tokens=False))
            
            input_id += _input_id
            
            if sentence["from"] == "user":
                # _target = tokenizer.special_tokens["<|start_header_id|>"] + [IGNORE_TOKEN_ID] * len(_user_ids) + tokenizer.special_tokens["<|end_header_id|>"] + tokenizer.encode("\n\n", bos=False, eos=False)
                # _target += _value_ids + tokenizer.special_tokens["<|eot_id|>"]
                _target = tokenizer.encode("<|start_header_id|>",add_special_tokens=False)
                _target.extend([IGNORE_TOKEN_ID] * len(_user_ids))
                _target.extend(tokenizer.encode("<|end_header_id|>",add_special_tokens=False))
                _target.extend(tokenizer.encode("\n\n",add_special_tokens=False))
                _target.extend([IGNORE_TOKEN_ID] * len(_value_ids))
                _target.extend(tokenizer.encode("<|eot_id|>",add_special_tokens=False))


            elif sentence["from"] == "assistant":
                # _target = tokenizer.special_tokens["<|start_header_id|>"] + [IGNORE_TOKEN_ID] * len(_user_ids) + tokenizer.special_tokens["<|end_header_id|>"] + tokenizer.encode("\n\n", bos=False, eos=False)
                # _inpu_targett_id += [IGNORE_TOKEN_ID] * len(_value_ids) + tokenizer.special_tokens["<|eot_id|>"]
                _target = tokenizer.encode("<|start_header_id|>",add_special_tokens=False)
                _target.extend([IGNORE_TOKEN_ID] * len(_user_ids))
                _target.extend(tokenizer.encode("<|end_header_id|>",add_special_tokens=False))
                _target.extend(tokenizer.encode("\n\n",add_special_tokens=False))
                _target.extend(_value_ids)
                _target.extend(tokenizer.encode("<|eot_id|>",add_special_tokens=False))

            else:
                raise NotImplementedError
           
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        pad_token_id = tokenizer.pad_token_id,
    )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int, model_type: str):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]

        if model_type =="Qwen":
            data_dict = preprocess_Qwen(sources, tokenizer, max_len)
        else:
            data_dict = preprocess_newllama3(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int, model_type: str):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model_type = model_type

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.raw_data = raw_data
        self.cached_data_dict = {}
        print(f"--------[init LazySupervisedDataset]")
        
    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        #print(f"-------- 正在处理第 [{i}]条",flush=True)
        if self.model_type =="Qwen":
            ret = preprocess_Qwen([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)
        else:
            ret = preprocess_newllama3([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)

        #print(f"---_getitem_ {i} {ret}")

        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            pad_token_id=ret["pad_token_id"],
        )
        self.cached_data_dict[i] = ret

        return ret
    

class data_collator_withbatchmaxlength:
    tokenizer: transformers.AutoTokenizer  
    max_length: Optional[int] = None
    return_tensors: str = "pt"

    def __init__(
        self,
        tokenizer: transformers.AutoTokenizer,
        max_length: Optional[int] = None,
        return_tensors: str = "pt"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors

    def __call__(self, features: List) -> Dict:
        # print(f"=======len [{len(features)}]")


        # print(f"--------{features[0]}")
        # print(f"{len(features[0]['input_ids'])}")
        # print(f"{len(features[0]['labels'])}")
        # print(f"{len(features[0]['attention_mask'])}")

        # print(f"padding {features}")

        # print(f"{self.tokenizer.pad_token_id}")
        # os._exit(0)
       
        # if "label" in batch:
        #     batch["labels"] = batch["label"]
        #     del batch["label"]
        # if "label_ids" in batch:
        #     batch["labels"] = batch["label_ids"]
        #     del batch["label_ids"]
        
        #print(f"features ------- {features}")
        batch = transformers.default_data_collator(features, self.return_tensors)
        #print(f"batch ------- {batch}")
        batch['input_ids'] = batch['input_ids'].to(torch.long)
        batch['labels'] = batch['labels'].to(torch.long)
        
        max_find_index = 0
        for j in range(len(batch['input_ids'])):
            find_index = len(batch['input_ids'][j])
            for i in range(len(batch['input_ids'][j]) - 1, 0, -1):
                if batch['input_ids'][j][i] == self.tokenizer.pad_token_id:
                    find_index = i
                    continue
                else:
                    break
            #print("")
            if find_index > max_find_index:
                max_find_index = find_index
        if max_find_index == 0 or len(batch['input_ids']) == 0:
            raise ValueError(f"No valid input_ids found in batch: {batch}")
        #print(f"find_index {max_find_index} --- batch ------- {batch}")    
        batch['input_ids'] = batch['input_ids'][:, :max_find_index]
        batch['labels'] = batch['labels'][:, :max_find_index]
        batch['attention_mask'] = batch['attention_mask'][:, :max_find_index]
        #print(f"new batch --- {batch}")
        #os._exit(0)
        return batch
    
# def data_collator_paddingwithbatchmaxlength(features: list) -> dict:
#     print(f"=======len [{len(features)}]")


#     print(f"--------{features[0]}")
#     print(f"{len(features[0]['input_ids'])}")
#     print(f"{len(features[0]['labels'])}")
#     print(f"{len(features[0]['attention_mask'])}")

#     print(f"padding {features}")
#     os._exit(0)
#     # 序列长度: [36, 106]
#     len_ids = [len(feature["input_ids"]) for feature in features]
#     # 取最长的序列长度: 106
#     longest = max(len_ids)
#     input_ids = []
#     labels_list = []
#     # 降序排列
#     # for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
#     #     ids = feature["input_ids"]  # tokenIds
#     #     seq_len = feature["seq_len"]  # seqLen
#     #     # len(prompt) x [-100] + Target + [longest - len(prompt)] * [-100]
#     #     labels = ([-100] * seq_len + ids[seq_len:] + [-100] * (longest - ids_l))
#     #     ids = ids + [pad_token_id] * (longest - ids_l)
 
#     #     _ids = torch.LongTensor(ids)
#     #     labels_list.append(torch.LongTensor(labels))
#     #     input_ids.append(_ids)
#     # # tensor([[], []])
#     input_ids = torch.stack(input_ids)
#     labels = torch.stack(labels_list)
#     return {
#         "input_ids": input_ids,
#         "labels": labels,
#     }

def user_pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:

        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded

@dataclass
class user_DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: transformers.AutoTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:


        batch = user_pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        print(f"features {features}")
        print(f"batch {batch}")
        return batch

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    # train_json = json.load(open(data_args.data_path, "r"))
    # train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

    train_data = []
    with open(data_args.data_path, "r") as f:
        for line in f:
            train_data.append(json.loads(line))
            # json_d = json.loads(line)
            # d = []
            # for dx in json_d['conversations']:
            #     if dx['role'] in ["user","assistant"]:
            #         d.append({"from":dx['role'], "value":dx['content']})
            # train_data.append({"conversations": d})
    print(f"本次SFT数据条数： {len(train_data)}")
    train_dataset = dataset_cls(train_data, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

def make_supervised_data_module_debug(
    tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if True else SupervisedDataset
    )
    rank0_print("Loading data...")

    # train_json = json.load(open(data_args.data_path, "r"))
    # train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

    train_data = []
    with open('/mnt/public/xuhaiyang/SFT_DATA/merge_files/shuati_core_tiny.jsonl', "r") as f:
        for line in f:
            train_data.append(json.loads(line))
            # json_d = json.loads(line)
            # d = []
            # for dx in json_d['conversations']:
            #     if dx['role'] in ["user","assistant"]:
            #         d.append({"from":dx['role'], "value":dx['content']})
            # train_data.append({"conversations": d})
    print(f"本次SFT数据条数： {len(train_data)}")
    train_dataset = dataset_cls(train_data, tokenizer=tokenizer, max_len=max_len)
    print(f"----------[{train_dataset[0]}]")
    return dict(train_dataset=train_dataset, eval_dataset=None)

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # This serves for single-gpu qlora.
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1))==1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are incompatible with QLoRA."
            )

    is_chat_model = 'chat' in model_args.model_name_or_path.lower()
    if (
            training_args.use_lora
            and not lora_args.q_lora
            and deepspeed.is_deepspeed_zero3_enabled()
            and not is_chat_model
    ):
        raise RuntimeError("ZeRO3 is incompatible with LoRA when finetuning on base model.")

    model_load_kwargs = {
        'low_cpu_mem_usage': not deepspeed.is_deepspeed_zero3_enabled(),
    }

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=GPTQConfig(
            bits=4, disable_exllama=True
        )
        if training_args.use_lora and lora_args.q_lora
        else None,
        **model_load_kwargs,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    if model_args.model_type == "Qwen":
        tokenizer.pad_token_id = tokenizer.eod_id
    else:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if training_args.use_lora:
        if lora_args.q_lora or is_chat_model:
            modules_to_save = None
        else:
            modules_to_save = ["wte", "lm_head"]
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        # Print peft trainable params
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    data_collator_user = data_collator_withbatchmaxlength(tokenizer,
						   								 max_length=512,
						   								 return_tensors="pt")
    
    test_data_collator = user_DataCollatorWithPadding(tokenizer, 
						   								 padding="max_length",
						   								 max_length=512,
						   								 return_tensors="pt")

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module, data_collator = data_collator_user
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)

def train_debug():
    print("=============haha start!!!!")

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=None,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    if model_args.model_type == "Qwen":
        tokenizer.pad_token_id = tokenizer.eod_id
    else:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # if training_args.use_lora:
    #     if lora_args.q_lora or is_chat_model:
    #         modules_to_save = None
    #     else:
    #         modules_to_save = ["wte", "lm_head"]
    #     lora_config = LoraConfig(
    #         r=lora_args.lora_r,
    #         lora_alpha=lora_args.lora_alpha,
    #         target_modules=lora_args.lora_target_modules,
    #         lora_dropout=lora_args.lora_dropout,
    #         bias=lora_args.lora_bias,
    #         task_type="CAUSAL_LM",
    #         modules_to_save=modules_to_save  # This argument serves for adding new tokens.
    #     )
    #     if lora_args.q_lora:
    #         model = prepare_model_for_kbit_training(
    #             model, use_gradient_checkpointing=training_args.gradient_checkpointing
    #         )

    #     model = get_peft_model(model, lora_config)

    #     # Print peft trainable params
    #     model.print_trainable_parameters()

    #     if training_args.gradient_checkpointing:
    #         model.enable_input_require_grads()

    # Load data
    data_module = make_supervised_data_module_debug(
        tokenizer=tokenizer, data_args=None, max_len=4096
    )

    print("=============haha finish!!!!")
    # # Start trainner
    # trainer = Trainer(
    #     model=model, tokenizer=tokenizer, args=training_args, **data_module
    # )

    # trainer.train()
    # trainer.save_state()

    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)

if __name__ == "__main__":
    train()
    #train_debug()
