# ARFBench - Anomaly Reasoning Framework Benchmark

Paper | [ARFBench Dataset Card](https://huggingface.co/datasets/Datadog/ARFBench) | [ARFBench Leaderboard](https://huggingface.co/spaces/Datadog/ARFBench) | [Toto-1.0-QA-Experimental Model Card](https://huggingface.co/Datadog/Toto-1.0-QA-Experimental)

## Overview
ARFBench (Anomaly Reasoning Framework Benchmark) is a multimodal time-series reasoning benchmark with 750 question-answer pairs grounded in real-world software incidents from Datadog. In this repository, we provide the code for evaluating all models on ARFBench.

## Installation
### Toto-1.0-QA-Experimental Installation
Running Toto-1.0-QA-Experimental requires a minimum of 4 NVIDIA A100 (40GB) GPUs to run. For users that hit OOMs, you may consider decreasing the max-ts-length hyperparameter.
```bash
git clone git@github.com:DataDog/ARFBench.git
cd ARFBench/toto-1.0-qa-experimental
python -m venv .venv
bash setup.sh
```
For `Toto-1.0-QA-Experimental`, the dataset download can be skipped.

### Third-party Evaluation Installation
We use `uv` to run the third-party evaluation scripts. This can be installed by running
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
After installing uv, sync the dependencies.
```bash
cd ARFBench
# If you only plan to use API-based evaluations on CPU
uv sync
# If you plan to use vLLM or Transformers-based evaluations on GPU
uv sync --extra gpu
```
If you prefer to use pip, the dependencies can be installed by running
```
pip install -r requirements.txt
```

## Getting Started
### Running Toto-1.0-QA-Experimental
[Toto-1.0-QA-Experimental](https://huggingface.co/Datadog/Toto-1.0-QA-Experimental) combines Toto time series embeddings with a Qwen3-VL vision-language model via LoRA adapters. Both the model weights and the ARFBench dataset are automatically downloaded from Hugging Face on first run.

**Run the evaluation:**
```bash
python toto-1.0-qa-experimental/evaluation/eval_toto_vlm.py \
    --model-path Datadog/Toto-1.0-QA-Experimental \
    --benchmark-source huggingface \
    --allow-multi-gpu
```
This downloads `Datadog/Toto-1.0-QA-Experimental` (LoRA + time series modules), the base models (`Qwen/Qwen3-VL-32B-Instruct`, `Datadog/Toto-Open-Base-1.0`), and the `Datadog/ARFBench` dataset automatically.

### Dataset Installation for Third-Party Evaluations
Using the given third-party evaluation scripts also requires the download of the [ARFBench data](https://huggingface.co/datasets/Datadog/ARFBench). This can be done by running
```bash
hf download Datadog/ARFBench --repo-type dataset
```
After download, the data should be placed in the same folder as these scripts. Note that if you decide to change the dataset directory structure, the scripts will need to be updated accordingly.

## Model-Specific Configuration
If you plan to use proprietary models such as OpenAI or Anthropic, you must set the environment variables before running the scripts.

### Running OpenAI models
You can run OpenAI models with the following command, e.g.
```bash
uv run python -m evaluation.eval_openai --model gpt-5 --max-concurrent 10
```

### Running Anthropic models
You can run Anthropic models with the following command, e.g.
```bash
uv run python -m evaluation.eval_anthropic --model claude-4-5-sonnet --max-concurrent 10
```

### Running other models via vLLM
You can run vision-language vLLM models (by default, Qwen3-VL) with
```bash
uv run python -m evaluation.eval_qwen3_vl
```
This will run the set of vision-language models included in the ARFBench paper. You can customize the evaluated models within the script.

### Running ChatTS
You can run ChatTS on ARFBench by downloading the model and adjusting the model path.
```bash
hf download bytedance-research/ChatTS-8B
uv run python -m evaluation.eval_chatts --model_path <PATH_TO_CHATTS_MODEL>
```

### Running OpenTSLM
OpenTSLM evaluation requires cloning the upstream OpenTSLM repository and placing
the ARFBench dataset adapter files into the OpenTSLM source tree.

#### One-command setup
From the repository root:
```bash
bash scripts/setup_opentslm.sh
```
This will:
- clone [OpenTSLM](https://github.com/StanfordBDHG/OpenTSLM) into `third_party/OpenTSLM` (if missing),
- copy `opentslm_custom/ARFBenchQADataset.py` to `third_party/OpenTSLM/src/time_series_datasets/arfbench/ARFBenchQADataset.py`,
- copy `opentslm_custom/arfbench_loader.py` to `third_party/OpenTSLM/src/time_series_datasets/arfbench/arfbench_loader.py`.

#### Manual setup (equivalent)
```bash
mkdir -p third_party
git clone https://github.com/StanfordBDHG/OpenTSLM.git third_party/OpenTSLM
cp opentslm_custom/ARFBenchQADataset.py third_party/OpenTSLM/src/time_series_datasets/arfbench/ARFBenchQADataset.py
cp opentslm_custom/arfbench_loader.py third_party/OpenTSLM/src/time_series_datasets/arfbench/arfbench_loader.py
```

#### Run evaluation
```bash
uv run python -m evaluation.eval_opentslm \
  --opentslm-src third_party/OpenTSLM/src \
  --benchmark data/arfbench-qa.csv \
  --data-dir arfbench-data \
  --model-id OpenTSLM/llama-3.2-1b-ecg-sp
```