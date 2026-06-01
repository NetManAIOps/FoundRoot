# FoundRoot: Towards Foundation Model for Root Cause Analysis via Structured Deep Thinking

The official implementation of **ICSE 2026** paper: **FoundRoot: Towards Foundation Model for Root Cause Analysis via Structured Deep Thinking**.

You can download the model checkpoints from huggingface: [FoundRoot-14B](https://huggingface.co/xiezhe22/FoundRoot-14B).

## TODO
1. Upload dataset preprocessing scripts.

**(Update)** Model checkpoints have been uploaded!

## Install
- Install `python>3.11`. We recommend using `Linux` environment with `8 x Nvidia A100 GPUs` for training and evaluation.
- Run `pip install -r requirements.txt`.

## Data Generation
- We have provided all the training and evaluation datasets under the `data/` folder (datasets A - J).
- Data Augmentation: You need to set the `SOURCE_DATASET_DIR`, `OUTPUT_DATASET_DIR` and path to a local LLM `PATH_TO_LLM` in `utils/augment_training_datasets.py` first. Please refer to the content under `datasets/` folder to prepare your own datasets. Then use `python3 -m utils.augment_training_datasets` to augment the datasets.
- Genenerate WarmUp-SFT Datasets: You need to set `LLM_MODEL_PATH` in `utils/generate_warmup_dataset.py` first. Then use `python3 -m utils.generate_warmup_dataset -d [PATH_TO_INPUT_DATA_DIR]` to generate the dataset.

## Model Training
- You can use [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) for WarmUp SFT Training.
- You can use [trl](https://github.com/huggingface/open-r1) for RL training with the provided receipe: `train/rl_receipe.yaml`.

## Evaluation
Download the model checkpoints from huggingface: [FoundRoot-14B](https://huggingface.co/xiezhe22/FoundRoot-14B).

We have provided the evaluation code for both local LLMs and APIs:
- Local LLMs: Set the LLM model path in `evaluation/evaluate_local_llms.py`. Then run `python3 -m evaluation.evaluate_local_llms` to get the evalution results.
- LLM APIs: Set the model name, base_url and api_key in `evaluation/evaluate_api.py`. Then run `python3 -m evaluation.evaluate_api` to get the evaluation results.

---

## Security

If you discover a potential security issue, please contact ByteDance Security via the [security center](https://security.bytedance.com/src) or email **[sec@bytedance.com](mailto:sec@bytedance.com)**.
**Do not** open public GitHub issues for vulnerabilities.

---

## License

This project is licensed under the **MIT License** (see `LICENSE`).

---

## Third-Party Dependencies

* Qwen ([https://github.com/QwenLM/Qwen2.5](https://github.com/QwenLM/Qwen2.5))
* DeepSpeed ([https://www.deepspeed.ai/](https://www.deepspeed.ai/))
* vLLM ([https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm))
* Flash-Attention ([https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention))
