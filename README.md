# FAMMA Benchmark Scripts
Scripts for preparing, uploading, and evaluating the `FAMMA` (Financial Multilingual Multi-Modal Question Answering) benchmark dataset.


## Overview

`FAMMA` is a multi-modal financial analysis benchmark dataset that includes:
- Questions in multiple languages (English, Chinese and French.)
- Financial charts and diagrams
- Multiple choice and open-ended questions
- Detailed explanations and answers

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

## Downloading Dataset

To download the dataset for use:

```bash
python step_1_download_dataset.py --hf_dir "weaverbirdllm/famma" --split "release_v2406"
```

Options:
- `--hf_dir`: HuggingFace repository name
- `--split`: Specific version to download (optional)
- `--save_dir`: Local directory to save the dataset (default: "./hf_data")


# Acknowledgment

The following repositories are used in `FAMMA benchmark`, either in close to original form or as an inspiration:


- [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)
- [EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess)