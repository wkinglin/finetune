#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
# DIR=$(pwd)
DIR=`pwd`

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training).
# Please set the options below according to the comments.
# For multi-gpu workers training, these options should be manually set for each worker.
# After setting the options, please run the script on each worker.

# Number of GPUs per GPU worker
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

# Number of GPU workers, for single-worker training, please set to 1
NNODES=${NNODES:-1}

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
NODE_RANK=${NODE_RANK:-0}

# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# The port for communication
MASTER_PORT=${MASTER_PORT:-6001}

MODEL="/mnt/public/xuhaiyang/model_zoo/Qwen-7B/" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA='/mnt/public/xuhaiyang/SFT_DATA/merge_files/chat.jsonl'
DATA='/mnt/public/xuhaiyang/SFT_DATA/merge_files/shuati_core.jsonl'

DATA='/mnt/public/xuhaiyang/SFT_DATA/merge_files/chat_shuati_core.jsonl'
output_dir='/mnt/public/xuhaiyang/SFT_MODEL/MODEL_20240425_09_44/'

#先执行 chat，在执行刷题
MODEL="/mnt/public/xuhaiyang/SFT_MODEL/MODEL_20240425_09_44_chat/"
output_dir='/mnt/public/xuhaiyang/SFT_MODEL/MODEL_20240426_00_14_chat_shuati/'
DATA='/mnt/public/xuhaiyang/SFT_DATA/merge_files/shuati_core.jsonl'


#测试小数据量
MODEL="/mnt/public/xuhaiyang/model_zoo/Qwen-7B/"
output_dir='/mnt/public/xuhaiyang/SFT_MODEL/MODEL_20240428_12_06_tiny_chat/'
DATA='/mnt/public/xuhaiyang/SFT_DATA/merge_files/shuati_core_tiny.jsonl'

#测试minibatch
MODEL="/mnt/public/xuhaiyang/model_zoo/Qwen-7B/"
output_dir='/mnt/public/xuhaiyang/SFT_MODEL/MODEL_20240428_14_45_minibatch/'
DATA='/mnt/public/xuhaiyang/SFT_DATA/merge_files/shuati_core_tiny.jsonl'

#测试新版shuti_core
MODEL="/mnt/public/xuhaiyang/model_zoo/Qwen-7B/"
DATA='/mnt/public/xuhaiyang/SFT_DATA/merge_files/shuati_core.jsonl'
output_dir='/mnt/public/xuhaiyang/SFT_MODEL/MODEL_20240430_20_07_oldbatch/'


#测试新版shuti_core_newbatch_1_16
MODEL="/mnt/public/xuhaiyang/model_zoo/Qwen-7B/"
DATA='/mnt/public/xuhaiyang/SFT_DATA/merge_files/shuati_core.jsonl'
output_dir='/mnt/public/xuhaiyang/SFT_MODEL/MODEL_20240501_15_07_newbatch/'

#测试先对话，在刷题
MODEL="/mnt/public/xuhaiyang/SFT_MODEL/MODEL_20240424_10_49/"
DATA='/mnt/public/xuhaiyang/SFT_DATA/merge_files/shuati.jsonl'
output_dir='/mnt/public/xuhaiyang/SFT_MODEL/MODEL_20240501_17_29_newbatch_chat_shuati_1_16/'


#测试llama3
#修改<|im_start|>user -> <|start_header_id|>user
#测试新版shuti_core_4_16
MODEL="/mnt/public/xuhaiyang/model_zoo/Meta-Llama-3-8B/"
DATA='/mnt/public/xuhaiyang/SFT_DATA/merge_files/shuati_core.jsonl'
output_dir='/mnt/public/xuhaiyang/SFT_MODEL/MODEL_20240508_20_46_Meta-Llama-3-8B/'


#测试新版shuti_core_4_16 修改拼接算法
MODEL="/mnt/public/xuhaiyang/model_zoo/Meta-Llama-3-8B/"
DATA='/mnt/public/xuhaiyang/SFT_DATA/merge_files/shuati_core.jsonl'
output_dir='/mnt/public/xuhaiyang/SFT_MODEL/MODEL_20240508_20_46_Meta-Llama-3-8B_2/'


#测试新版shuti_core_4_16 修改拼接算法,仅chat
MODEL="/mnt/public/xuhaiyang/model_zoo/Meta-Llama-3-8B/"
DATA='/mnt/public/xuhaiyang/SFT_DATA/merge_files/chat.jsonl'
output_dir='/mnt/public/xuhaiyang/SFT_MODEL/MODEL_20240508_20_46_Meta-Llama-3-8B_chat/'

#测试新版shuti_core_4_16 修改拼接算法,仅chat
MODEL="/mnt/public/xuhaiyang/SFT_MODEL/MODEL_20240508_20_46_Meta-Llama-3-8B_chat/"
DATA='/mnt/public/xuhaiyang/SFT_DATA/merge_files/shuati.jsonl'
output_dir='/mnt/public/xuhaiyang/SFT_MODEL/MODEL_20240508_20_46_Meta-Llama-3-8B_chat_shuati/'

#在h800第一次训练
#测试新版shuti_core_4_16 修改拼接算法,chat + shuati_core
MODEL="/mnt/public/xuhaiyang/SFT_MODEL/MODEL_20240508_20_46_Meta-Llama-3-8B_chat/"
DATA='/mnt/public/xuhaiyang/SFT_DATA/merge_files/shuati_core.jsonl'
output_dir='/mnt/public/xuhaiyang/SFT_MODEL/MODEL_20240508_20_46_Meta-Llama-3-8B_chat_shuati_core/'


#测试新版shuti_core_4_16 修改拼接算法,混合 shuati_core
#gradient_accumulation_steps = 1
#对标 MODEL_20240508_20_46_Meta-Llama-3-8B_2
MODEL="/mnt/public/xuhaiyang/model_zoo/Meta-Llama-3-8B/"
DATA='/mnt/public/xuhaiyang/SFT_DATA/merge_files/shuati_core.jsonl'
output_dir='/mnt/public/xuhaiyang/SFT_MODEL/MODEL_20240511_09_00_Meta-Llama-3-8B_shuati_core/'


#测试新版shuti_core_4_16 修改拼接算法,混合 shuati_core
#gradient_accumulation_steps = 32
#对标 MODEL_20240508_20_46_Meta-Llama-3-8B_2
#epoche_5
MODEL="/mnt/public/xuhaiyang/model_zoo/Meta-Llama-3-8B/"
DATA='/mnt/public/xuhaiyang/SFT_DATA/merge_files/shuati_core.jsonl'
output_dir='/mnt/public/xuhaiyang/SFT_MODEL/MODEL_20240511_10_28_Meta-Llama-3-8B_shuati_core_32_epoche_5/'

#测试新版shuti_core_4_16 使用llama3的newprompt,仅chat

#MODEL="/mnt/public/xuhaiyang/model_zoo/Meta-Llama-3-8B/"
#MODEL="/mnt/public/xuhaiyang/model_zoo/Meta-Llama-3-8B-instuct/"
MODEL="/mnt/public/xuhaiyang/model_zoo/Meta-Llama-3-8B/"
DATA='/mnt/public/xuhaiyang/SFT_DATA/merge_files/chat.jsonl'
output_dir='/mnt/public/xuhaiyang/SFT_MODEL/MODEL_20240515_08_59_Meta-Llama-3-8B_chat/'


#测试新版shuti_core_4_16 使用llama3的newprompt,仅chat

#MODEL="/mnt/public/xuhaiyang/model_zoo/Meta-Llama-3-8B/"
MODEL="/mnt/public/algm/yuantao_home/public_models/Meta-Llama-3-8B-Instruct/"
#MODEL="/mnt/public/xuhaiyang/model_zoo/Meta-Llama-3-8B/"
DATA='/mnt/public/xuhaiyang/SFT_DATA/merge_files/chat.jsonl'
output_dir='/mnt/public/xuhaiyang/SFT_MODEL/MODEL_20240515_08_59_Meta-Llama-3-8B-instuct_chat/'




MODEL="/mnt/public/algm/models/Qwen1.5-7B"
DATA='/mnt/public/xuhaiyang/SFT_DATA/merge_files/shuati_core.jsonl'
output_dir='/mnt/public/jinweilin/SFT_MODEL/MODEL_20240529_Qwen_7B_shuati/'

MODEL="/mnt/public/algm/models/Qwen1.5-7B-Chat"
DATA='/mnt/public/xuhaiyang/SFT_DATA/merge_files/shuati_core.jsonl'
output_dir='/mnt/public/jinweilin/SFT_MODEL/MODEL_20240529_Qwen_7B-Chat_shuati/'

MODEL="/mnt/public/algm/models/Qwen1.5-14B"
DATA='/mnt/public/xuhaiyang/SFT_DATA/merge_files/shuati_core.jsonl'
output_dir='/mnt/public/jinweilin/SFT_MODEL/MODEL_20240529_Qwen1.5_14B_shuati/'

MODEL="/mnt/public/algm/models/Qwen1.5-14B-Chat"
DATA='/mnt/public/xuhaiyang/SFT_DATA/merge_files/shuati_core.jsonl'
output_dir='/mnt/public/jinweilin/SFT_MODEL/MODEL_20240530_Qwen1.5_14B-Chat_shuati/'

MODEL="/mnt/public/algm/models/Meta-Llama-3-8B-Instruct"
DATA="/mnt/public/jinweilin/code/shuati_core_compress.jsonl"
output_dir='/mnt/public/jinweilin/SFT_MODEL/MODEL_20240530_Meta-Llama-3-8B-Instruct_shuati_compress/'

MAX_LENGTH=4096
NUM_TRAIN_EPOCHS=1
MODEL_TYPE="llama3"

function usage() {
    echo '
Usage: bash finetune/finetune_ds.sh [-m MODEL_PATH] [-d DATA_PATH]
'
}

while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model )
            shift
            MODEL=$1
            ;;
        -d | --data )
            shift
            DATA=$1
            ;;
        -o | --output )
            shift
            output_dir=$1
            ;;
        -t | --modelType )
            shift
            MODEL_TYPE=$1
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

#GPUS_PER_NODE=1
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


#测试默认per_device_train_batch_size 1  gradient_accumulation_steps 16
#测试per_device_train_batch_size 4  gradient_accumulation_steps 8 
torchrun $DISTRIBUTED_ARGS finetune_chat.py \
    --model_name_or_path $MODEL \
    --model_type $MODEL_TYPE  \
    --data_path $DATA \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length $MAX_LENGTH \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed finetune/ds_config_zero3.json
