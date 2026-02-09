# Hybrid Decoding Engine

## Inference
Do inference on all benchmarks with ASPD and SFT model:
```bash
export CUDA_VISIBLE_DEVICES=0 # Run each model with one GPU
# use -pm to specify a parallel model
# use -rm to specify a raw/SFT model
# Replace the -pm/-rm model name with your ASPD and SFT save_path (located in `../train/saves/Qwen25-7B/full`):
python quick_eval_qw.py -b mt_bench -pm Qwen7B_ASPD
python quick_eval_qw.py -b mt_bench -rm Qwen7B_SFT

python quick_eval_qw.py -b vicuna_bench -rm Qwen7B_SFT
python quick_eval_qw.py -b vicuna_bench -pm Qwen7B_ASPD

python quick_eval_qw.py -b rag_bench -pm Qwen7B_ASPD
python quick_eval_qw.py -b rag_bench -rm Qwen7B_SFT


# You can make a soft link to baseline model in ../train/saves/Qwen25-7B/fulls
# To run the baseline:
python quick_eval_qw.py -b mt_bench -rm Qwen2.5-7B-Instruct
python quick_eval_qw.py -b vicuna_bench -rm Qwen2.5-7B-Instruct
python quick_eval_qw.py -b rag_bench -rm Qwen2.5-7B-Instruct
```

All running results will save in `APAR_BENCH_Qwen25-7B_RES`.

## Evaluation

Before start, please follow instruction `data_ppl/readme.md` to setup a OpenAI-API-Endpoint.

Do evaluation and show results:
```bash
# mt bench
python gen_judgment.py --parallel 60 --bench-name mt_bench # specify a --parallel num to set max threads for evaluation
python show_result.py --bench-name mt_bench
# vicuna bench
python gen_judgment.py --parallel 60 --bench-name vicuna_bench
python show_result.py --bench-name vicuna_bench
# rag bench
python gen_judgment.py --parallel 60 --bench-name mt_bench
python show_result.py --bench-name mt_bench
python gen_judgment.py --parallel 60 --bench-name vicuna_bench
python show_result.py --bench-name vicuna_bench
python gen_judgment.py --parallel 60 --bench-name rag_bench
python show_result.py --bench-name rag_bench
```

All running results will save in `eval`.