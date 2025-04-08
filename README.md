# FAMMA Benchmark Scripts
Scripts for evaluating the `FAMMA` (Financial Domain Multilingual Multimodal Question Answering) benchmark dataset.

<div align="center">
<a href="https://famma-bench.github.io/famma/"><img alt="Home Page" src="https://img.shields.io/badge/🌐-Home_Page-blue"></a> • <a href="https://huggingface.co/datasets/weaverbirdllm/famma"><img alt="Hugging Face" src="https://img.shields.io/badge/🤗-Dataset-ffce44?logo=huggingface"></a>
</div>




<div align="center">
<a href="https://github.com/famma-bench/bench-script">
    <img alt="Code License" src="https://img.shields.io/badge/license-Apache-000000.svg?&color=f5de53">
  </a>
  <a href="commit">
    <img alt="Last Commit" src="https://img.shields.io/github/last-commit/famma-bench/bench-script">
  </a>
  <a href="https://github.com/famma-bench/bench-script/issues">
  <img alt="Open Issues" src="https://img.shields.io/github/issues-raw/famma-bench/bench-script" />
</a>
</div>


## NEWS

🔥 **Latest Updates**:
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png)[2025/04] Release of an interactive notebook [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/famma-bench/bench-script/blob/main/notebooks/FAMMA_1_dataset_inspection.ipynb) for exploring and analyzing the FAMMA dataset.
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png)[2025/03] Release of `release_basic_txt` and `release_livepro_txt`, two purely textual datasets that utilize OCR to extract multimodal information and convert it into textual context for each question in `release_basic` and `release_livepro`, respestively.
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png)[2025/02] Release of `release_livepro` dataset.
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png)[2025/01] Release of `release_basic` dataset, now including answers and explanations with enhanced quality.
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png) [2024/06] Initial public release of `FAMMA` benchmark (based on the `release_basic` dataset), along with our paper: [FAMMA: A Benchmark for Financial Domain Multilingual Multimodal Question Answering](https://arxiv.org/abs/2410.04526).



## Introduction

`FAMMA` is a multi-modal financial Q&A benchmark dataset. The questions encompass three heterogeneous image types - tables, charts and text & math screenshots - and span eight subfields in finance, comprehensively covering topics across major asset classes. Additionally, all the questions are categorized by three difficulty levels — easy, medium, and hard - and are available in three languages — English, Chinese, and French. Furthermore, the questions are divided into two types: multiple-choice and open questions.


### Live Benchmarking Concept

In addition to the baseline dataset (`release_basic` that contains 1935 questions), `FAMMA` provides a "live" benchmark for evaluating financial analysis capabilities of LLMs. The benchmark continuously collects new questions from real-world financial professionals, ensuring up-to-date and contamination-free evaluation. 

The "live" nature of FAMMA means:
1. **Expert-Sourced Questions**: New questions are continuously proposed by financial experts, ensuring they have never been made public before and reflect real-world financial analysis scenarios. See [contributors](https://github.com/famma-bench/bench-script/blob/main/contributors.md).
2. **Contamination Prevention**: Questions in the live set (at the moment `release_livepro`) have non-public answers and explanations.
3. **Time-Based Evaluation**: Models can be evaluated on questions from specific time periods.
4. **Domain Coverage**: Questions span across different financial topics and complexity levels, curated by domain experts.


## Dataset Versions

FAMMA is continuously updated with new questions. We provide different versions of the dataset:

- `release_basic`: The release containing 1935 questions, collected from online sources. Apart from the questions, both answers and explanations are provided.
- `release_basic_txt`: A textual version of `release_basic`, where OCR has been used to extract multimodal information and convert it into contextual text for each question.
- `release_livepro`: The release containing 103 questions, created by invited experts. Only the questions are provided.
- `release_livepro_txt`: A textual version of `release_livepro`, generated using OCR to extract and represent the questions in text format.

You can specify the dataset version when downloading, see [Downloading Dataset](#downloading-dataset) for more details.

## Usage


### Installation

Clone the repository and install the dependencies:
```bash
git clone https://github.com/famma-bench/bench-script.git
cd bench-script
pip install -r requirements.txt
pip install -e .
```

### Downloading Dataset


To download the dataset, run the following command:


```bash
python step_1_download_dataset.py \
    --hf_dir "weaverbirdllm/famma" \
    --split "release_basic" \ # or "release_livepro" or None to download the whole set
    --save_dir "./hf_data"
```

Options:
- `--hf_dir`: HuggingFace repository name
- `--split`: Specific version to download (optional)
- `--save_dir`: Local directory to save the dataset (default: "./hf_data")


After downloading, the dataset will be saved in the local directory `./data` in json format. 

### Dataset Structure

Each sample in the dataset contains:
- idx: a unique identifier for the index of the question in the dataset.
- question_id: a unique identifier for the question across the whole dataset: {language}{main_question_id}{sub_question_id}_{release_version}.
- context: relevant background information related to the question.
- question: the specific query being asked.
- options: the specific query being asked.
- image_1- image_7: directories of images referenced in the context or question.
- image_type: type of the image, e.g., chart, table, screenshot.
- answers: a concise and accurate response. (public on `release_basic`, non-public on the live set `release_livepro`)
- explanation: a detailed justification for the answer. (public on `release_basic`, non-public on the live set `release_livepro`)
- topic_difficulty: a measure of the question's complexity based on the level of reasoning required.
- question_type: categorized as either multiple-choice or open-ended.
- subfield: the specific area of expertise to which the question belongs, categorized into eight subfields.
- language: the language in which the question text is written.
- main_question_id: a unique identifier under the same language subset for the question within its context; questions with the same context share the same ID.
- sub_question_id: a unique identifier for the question within its corresponding main question.
- is_arithmetic: whether the question is an arithmetic question that needs heavy calculation.
- ans_image_1 - ans_image_6: (public on `release_basic`, non-public on the live set `release_livepro`)

## Custom LLM Evaluation

One can customized a LLM and evaluate it on the `FAMMA` dataset. It involves two steps:
1. Define a `generate` function for the LLM to generate the answer and explanation. See [custom_llm.py](main_scripts/custom_llm.py) for more details.
2. Define a yaml config for the LLM to load the model. See [custom_gen.yaml](./configs/custom_gen.yaml) for more details.

Then the evaluation can be done with the same script (except specify the config file) as the default one.



## Colab Tutorials

Explore the following tutorials that can be opened directly in Google Colab:

- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/famma-bench/bench-script/blob/main/notebooks/FAMMA_1_dataset_inspection.ipynb) Tutorial 1: Dataset inspection in FAMMA.


## ERRATA
We maintain a list of known issues and updates in the [ERRATA.md](./ERRATA.md) file. Particularly, we document issues regarding erroneous tests and problems not amenable to autograding. We are constantly using this feedback to improve our problem selection heuristics as we update `FAMMA`.


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
