# FAMMA Benchmark Scripts
Scripts for preparing, uploading, and evaluating the `FAMMA` (Financial Domain Multilingual Multimodal Question Answering) benchmark dataset.

🏠 [Home Page](https://huggingface.co/weaverbirdllm/famma) • 💻 [Dataset](https://huggingface.co/datasets/weaverbirdllm/famma) 



** STILL UNDER DEVELOPMENT **

## Introduction

`FAMMA` provides a "live" benchmark for evaluating financial analysis capabilities of LLMs. The benchmark continuously collects new questions from real-world financial scenarios, ensuring up-to-date and contamination-free evaluation. Questions are sourced from various financial domains including accounting, economics, investment, and more.

### Live Benchmarking Co ncept

The "live" nature of FAMMA means:
1. **Expert-Sourced Questions**: New questions are continuously proposed by financial experts, ensuring they have never been made public before and reflect real-world financial analysis scenarios.
2. **Contamination Prevention**: Questions in the live set (`release_v2501`) have non-public answers and explanations.
3. **Time-Based Evaluation**: Models can be evaluated on questions from specific time periods.
4. **Domain Coverage**: Questions span across different financial topics and complexity levels, curated by domain experts.

`FAMMA` is a multi-modal financial Q&A benchmark dataset that includes:
- Questions in multiple languages (English, Chinese and French.)
- Financial charts and diagrams
- Multiple choice and open-ended questions
- Detailed explanations and answers

## Setup

## Installation

```bash
git clone https://github.com/weaverbirdllm/bench-script.git
pip install -r requirements.txt
```

## Dataset Versions

FAMMA is continuously updated with new questions. We provide different versions of the dataset:

* `release_v2406`: The initial release containing xxxx questions.
* `release_v2501`: Latest release with additional questions and improvements

You can specify the dataset version when downloading:
```bash
python step_1_download_dataset.py --hf_dir "weaverbirdllm/famma" --split "release_v2406"
```

## Usage

### Downloading Dataset

```bash
python step_1_download_dataset.py \
    --hf_dir "weaverbirdllm/famma" \
    --split "release_v2406" \
    --save_dir "./hf_data"
```

Options:
- `--hf_dir`: HuggingFace repository name
- `--split`: Specific version to download (optional)
- `--save_dir`: Local directory to save the dataset (default: "./hf_data")

### Dataset Structure

Each sample in the dataset contains:
- idx: a unique identifier for the index of the question in the dataset.
- question_id: a unique identifier for the question across the whole dataset: {language}{main_question_id}{sub_question_id}_{release_version}.
- context: relevant background information related to the question.
- question: the specific query being asked.
- options: the specific query being asked.
- image_1- image_7: directories of images referenced in the context or question.
- image_type: type of the image, e.g., chart, table, screenshot.
- answers: a concise and accurate response. (public on release v2406, non-public on the live set release v2501)
- explanation: a detailed justification for the answer. (public on release v2406, non-public on the live set release v2501)
- topic_difficulty: a measure of the question's complexity based on the level of reasoning required.
- question_type: categorized as either multiple-choice or open-ended.
- subfield: the specific area of expertise to which the question belongs, categorized into eight subfields.
- language: the language in which the question text is written.
- main_question_id: a unique identifier for the question within its context; questions with the same context share the same ID.
- sub_question_id: a unique identifier for the question within its corresponding main question.
- ans_image_1 - ans_image_6: (public on release v2406, non-public on the live set release v2501)

## Custom Evaluation

You can evaluate model outputs using a custom file. Format your outputs as:
```json
[
    {
        "question_id": "en_1_1_v2406",
        "answer": "B",
        "explanation": "Based on the chart..."
    },
    ...
]
```

## Acknowledgments

The following repositories were used in developing FAMMA:

- [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)
- [EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess)

## Citation

If you use FAMMA in your research, please cite the following paper:

```bibtex
@article{xue2024famma,
  title={FAMMA: A Benchmark for Financial Domain Multilingual Multimodal Question Answering},
  author={Siqiao Xue, Tingting Chen, Fan Zhou, Qingyang Dai, Zhixuan Chu, and Hongyuan Mei},
  journal={arXiv preprint arXiv:2410.04526},
  year={2024},
  url={https://arxiv.org/abs/2410.04526}
}
```
