# CERES

Online learning platform for ophthalmology.

## Repository Core Contents

CERES is an online learning platform for ophthalmology. This repository currently contains the model training part (`model/`), which performs multi-turn teaching GRPO/SFT fine-tuning of Qwen2.5-VL based on ms-swift 3.4.1: it constructs multi-turn rollouts via a virtual student simulator, and uses a two-layer reward of "sequence reward + anchor credit" to train the model to produce structured teaching actions.

## Project Structure

```text
CERES/
├── README.md                # this file
├── requirements.txt         # complete dependencies for the training environment (Python 3.10)
├── .gitignore               # excludes run artifacts and secrets
└── model/
    └── grpo/
        ├── ceres_plugin/    # multi-turn environment teacher_env, reward, student simulator, trajectory storage
        ├── data_pipeline/   # data preparation pipeline (8 CLI modules)
        ├── eval/            # inference demo and offline evaluation
        ├── scripts/         # training / inference Bash launch scripts
        ├── data/            # sample datasets
        ├── README.md        # detailed explanation of the training code
        └── requirements.txt # the project's own dependencies
```

## How to Launch Training

### 0. Environment

The training environment (Python 3.10 + ms-swift) lives outside the repository and is not committed to Git:

```bash
export PY_BIN="$PWD/../model/my-env/swift/bin/python"
export PATH="$(dirname "$PY_BIN"):$PATH"
cd model/grpo
```

On a brand-new machine, just recreate the Python 3.10 environment according to the repository root [`requirements.txt`](requirements.txt).

### 1. Prepare Data

```bash
python -m data_pipeline.prepare_datasets --type cmexam \
  --input RAW/CMExam/data/train.csv
python -m data_pipeline.extract_questions \
  --input data/manifest.jsonl --output data/raw_pool.jsonl
python -m data_pipeline.rewrite_question \
  --input data/raw_pool.jsonl --output data/queries_candidates.jsonl
python -m data_pipeline.build_query_dataset \
  --input data/queries_candidates.jsonl \
  --output data/ceres_oph_queries.jsonl
```

The repository already ships a small amount of sample data, which can be used directly for pipeline verification.

### 2. SFT Warmup + GRPO Training

```bash
export CERES_MODEL=/path/to/Qwen2.5-VL-32B-Instruct
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4

# SFT warmup
bash scripts/run_sft_warmup.sh

# GRPO training
export CERES_STUDENT_API_KEY="<your-api-key>"
export CERES_ADAPTERS=output/ceres-oph-sft-warmup
bash scripts/run_grpo_oph_lora.sh
```

### 3. Inference

```bash
CERES_MODEL=/path/to/Qwen2.5-VL-7B-Instruct \
bash scripts/run_demo_dialogue.sh \
  --input data/ceres_oph_queries.jsonl \
  --num 8 \
  --out-dir output/demo/baseline
```

Mount the trained LoRA adapter:

```bash
CERES_MODEL=/path/to/Qwen2.5-VL-7B-Instruct \
bash scripts/run_demo_dialogue.sh \
  --adapters output/ceres-oph-grpo-lora-v1/checkpoint-xxx \
  --out-dir output/demo/grpo
```

For more parameters, data pipeline details, and the reward mechanism, see [`model/grpo/README.md`](model/grpo/README.md).
