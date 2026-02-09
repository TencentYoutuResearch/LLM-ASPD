# Serial-Parallel Training

## Code and Config Preparation

Replace `LLaMA-Factory/src/llamafactory/data/collator.py` with ours:
```bash
cp collator.py LLaMA-Factory/src/llamafactory/data/
```
Add training configs to `LLaMA-Factory/expr`:
```bash
cp -r expr LLaMA-Factory/
```


## Data Preparation
The save format of training data should be like (Alpaca Format):
```json
{
    {
        "unique_id": "XXX",
        "system": "",
        "instruction": "User Input Here",
        "input": "",
        "history": [["Q1", "A1"], ["Q2", "A2"]],
        "output": "Model Output Here",
  },
  {
        "unique_id": "YYY",
        "system": "",
        "instruction": "User Input Here",
        "input": "",
        "history": [],
        "output": "Model Output Here",
  }
}
```

Field Description:
- `"unique_id"`: Not required.
- `"history"`: history messages, a list where each element is prompt-response pairs like ["user prompt", "model response"].
- `"system"`: System prompt. If empty, default one will be used in LLama-Factory.
- `"instruction"`: User prompt.
- `"output"`: Model response.

Once all train data are ready, add data info to `LLaMA-Factory/data/dataset_info.json`:
```json
{
  // ...
  "APAR_ASPD": {
    "file_name": "../../../data_ppl/train_data/ParaTrain_apar_qw.jsonl"
  },
  "APAR_SFT": {
    "file_name": "../../../data_ppl/train_data/SFTTrain_apar_qw.jsonl"
  }
}
```
You can use `APAR_ASPD` and `APAR_SFT` to refer these training datasets in config files.

## Training Configs
All training config files are placed in `LLaMA-Factory/expr`. There are two config files for Serial and Parallel Trianing, respectively:
- `qw_7b_instruct_sft_apar_aspd.yaml`: Parallel training for ASPD model.
- `qw_7b_instruct_sft_apar_sft.yaml`: SFT training for SFT model.

Now, let's take a detailed look at the content of `qw_7b_instruct_sft_apar_aspd.yaml`:
```yaml
### model
model_name_or_path: ../../data_ppl/para_model/Qwen2.5-7B-Instruct-Async
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json  # choices: [ds_z0_config.json, ds_z2_config.json, ds_z3_config.json]
# flash_attn: fa2
# use_unsloth: true

### dataset
dataset: APAR_ASPD
template: qwen
cutoff_len: 8096
# max_samples: 1
overwrite_cache: true
preprocessing_num_workers: 512

### output[]
output_dir: ../saves/Qwen25-7B/full/Qwen7B_ASPD
logging_steps: 2
# save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_strategy: epoch

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2.0e-5
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
# eval_dataset: alpaca_en_demo
# val_size: 0.1
# per_device_eval_batch_size: 1
# eval_strategy: steps
# eval_steps: 500
```
The only items you need to modify are follows:
- `model_name_or_path`: The saved model path outputed by `../data_ppl/step0_add_sptokens.py`
- `dataset`: The reference name of the training dataset in `LLaMA-Factory/data/dataset_info.json`. It should be `APAR_ASPD` here.
- `output_dir`: The save path of tranined models. The final save path will be `saves/Qwen25-7B/full/Qwen7B_ASPD`

# Launching

## ASPD Model
For DDP (Distributed Data Parallel) training:
```bash
# cd to ../ASPD/train/LLaMA-Factory
export PARA_TRAIN=true
export PARA_MASK=cantsee
export PARA_ST_TOKEN=151669
export PARA_ED_TOKEN=151670
export POSID_MODE=normal
FORCE_TORCHRUN=1 NNODES=${WORLD_SIZE} NODE_RANK=${RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} llamafactory-cli train expr/qw_7b_instruct_sft_apar_aspd.yaml
```

For Stand-Alone training:
```bash
# cd to ../ASPD/train/LLaMA-Factory
export PARA_TRAIN=true
export PARA_MASK=cantsee
export PARA_ST_TOKEN=151669
export PARA_ED_TOKEN=151670
export POSID_MODE=normal
llamafactory-cli train expr/qw_7b_instruct_sft_apar_aspd.yaml
```

## SFT Model
For DDP (Distributed Data Parallel) training:
```bash
# cd to ../ASPD/train/LLaMA-Factory
export PARA_TRAIN=false
FORCE_TORCHRUN=1 NNODES=${WORLD_SIZE} NODE_RANK=${RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} llamafactory-cli train expr/qw_7b_instruct_sft_apar_aspd.yaml
```

For Stand-Alone training:
```bash
# cd to ../ASPD/train/LLaMA-Factory
export PARA_TRAIN=false
llamafactory-cli train expr/qw_7b_instruct_sft_apar_sft.yaml
```

All saved models can be found in `saves/Qwen25-7B/full`.