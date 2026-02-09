# <img src="imgs/logo/logo.svg" alt="Youtu Logo" height="24px"> ASPD: Unlocking Adaptive Serial-Parallel Decoding by Exploring Intrinsic Parallelism in LLMs

<p align="center">
| <a href="https://arxiv.org/abs/2508.08895"><b>📄 Paper</b></a>
| <a href="#-installation"><b>⚒️ Installation</b></a> 
| <a href="#-getting-started"><b>🚀 Getting Started</b> </a> 
| <a href="#-license"><b>⚖️ License</b> </a> 
| <a href="#-citation"><b>📚 Citation</b></a> 
| 
</p>

`ASPD` (Adaptive Serial-Parallel Decoding) is an end-to-end framework for **accelerating LLM inference** by exploiting **intrinsic parallelism** in autoregressive outputs. Instead of strictly generating tokens one-by-one, ASPD identifies **parallelizable branches** in model responses and decodes them **simultaneously**, achieving significant speedups while maintaining generation quality. ⚡️

---

## ✨ What’s Inside

`ASPD` provides a full parallel decoding framework that consists of 1. Non-Invasive Parallel Data Transformation Pipeline, 2. Serial-Parallel Training, 3. Inference with Hybrid Decoding Engine (Using transformers), 4. Evaluation.

A typical workflow looks like this:

1. **Parallel Data Transformation (Non-Invasive)** 🧩  
   Automatically extract and validate parallelizable structures from standard autoregressive model outputs—no model internals modification required.

2. **Serial-Parallel Training** 🏋️  
   Train models to better follow and generate parallelizable structures.

3. **Hybrid Decoding Engine for Inference** 🚀  
   Seamlessly switch between **serial** and **parallel** decoding modes while **reusing KV cache** to maximize efficiency.

4. **Evaluation** 📊  
   Evaluate both **effectiveness** (quality) and **efficiency** (latency/speedup) across diverse tasks (general, RAG, math reasoning).

---

## ⚒️ Installation

We use [uv](https://github.com/astral-sh/uv) to setup a clean installation:

```bash
# make sure current dir is under the code root
# For base env
uv venv -p 3.11
source .venv/bin/activate
uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install transformers==4.49.0 tqdm xlsxwriter shortuuid pandas accelerate==1.2.1 setuptools deepspeed==0.17.5 openai==1.104.2
# For evaluation
cd infer
git clone --branch v0.2.36 https://github.com/lm-sys/FastChat.git
cd FastChat
uv pip install -e ".[model_worker]"
cd ../../
# For training
cd train
git clone --branch v0.9.2 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
uv pip install -e ".[metrics]" --no-build-isolation
cd ../../
```

After installation, you can activate the venv anytime just by running following commands:
```bash
# make sure current dir is under the code root
source .venv/bin/activate
```
Or exit it:
```bash
deactivate
```

---

## 🚀 Getting Started

We provice details guidences in following dirs:
 1. `data_ppl`: Non-Invasive Parallel Data Transformation Pipeline
 2. `train`: Serial-Parallel Training (powered by [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) with [Apache-2.0 license](https://github.com/hiyouga/LLaMA-Factory?tab=Apache-2.0-1-ov-file))
 3. `infer`: Inference with Hybrid Decoding Engine (using [transformers](https://github.com/huggingface/transformers) with [Apache-2.0 license](https://github.com/huggingface/transformers?tab=Apache-2.0-1-ov-file)), Evaluation (powered by [FastChat](https://github.com/lm-sys/FastChat) with [Apache-2.0 license](https://github.com/lm-sys/FastChat?tab=Apache-2.0-1-ov-file))

Recommended run order (to match the full ASPD pipeline):  
1) `data_ppl` → 2) `train` → 3) `infer`

Please goto above dirs in order to start your parallel trip~😊

---

## ⚖️ License

ASPD is released under the [License](LICENSE).

---

## 📚 Citation

Please consider citing our work if you find it helpful:

```bibtex
@article{chen2025aspd,
  title={ASPD: Unlocking Adaptive Serial-Parallel Decoding by Exploring Intrinsic Parallelism in LLMs},
  author={Chen, Keyu and Shen, Zhifeng and Yu, Daohai and Wu, Haoqian and Wen, Wei and He, Jianfeng and Qiao, Ruizhi and Sun, Xing},
  journal={arXiv preprint arXiv:2508.08895},
  year={2025}
}
```