import subprocess

sft_list = [
    {
        "model": "/mnt/public/algm/models/Meta-Llama-3-8B",
        "data": '/mnt/public/xuhaiyang/SFT_DATA/merge_files/shuati_core.jsonl',
        "output_dir": '/mnt/public/jinweilin/SFT_MODEL/MODEL_20240530_Meta-Llama-3-8B_shuati/',
        "model_type": "llama3"
    },
    {
        "model": "/mnt/public/algm/models/Meta-Llama-3-8B-Instruct",
        "data": '/mnt/public/xuhaiyang/SFT_DATA/merge_files/shuati_core.jsonl',
        "output_dir": '/mnt/public/jinweilin/SFT_MODEL/MODEL_20240530_Meta-Llama-3-8B-Instruct_shuati/',
        "model_type": "llama3"
    }
]

for index, item in enumerate(sft_list):
    model = item['model']
    data = item['data']
    output_dir = item['output_dir']
    model_type = item['model_type']
    result = subprocess.run(['./finetune/finetune_chat.sh', '-m', model, '-d', data, '-o', output_dir, '-t', model_type], stdout=subprocess.PIPE, text=True)

